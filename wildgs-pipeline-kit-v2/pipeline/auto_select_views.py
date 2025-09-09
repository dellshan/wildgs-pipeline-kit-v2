#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_select_views.py

从一堆渲染帧里，自动挑选一条（或一组）“好视角”。
打分由以下项组成（都是 0~1 之间再加权）：
  • 覆盖度 A：同帧所有实例掩码（union）的像素占比（越大越好）
  • 清晰度/显著性 B：Laplacian 方差归一化（越大越好，避免虚焦/模糊帧）
  • 语义多样性 D：与已选帧的 CLIP 相似度的 1 - max（越大越不同）
  • 平滑/间隔约束：与已选帧的距离（位姿或索引）过近则跳过

CLIP 依赖策略：
  先尝试 open_clip（LAION ViT-B/32），失败则尝试 OpenAI CLIP（ViT-B/32），
  两者都不可用或 --beta=0 则自动禁用多样性项。

输入：
  --frames      必选，帧图像目录
  --masks_csv   可选，由实例提取脚本生成的 CSV（含列：image, mask, conf …）
  --poses       可选，位姿 JSON（可多种宽松格式，见 parse_poses）
  --out_dir     必选，输出目录

可调权重：
  --top_k       选取多少帧（默认 60）
  --alpha       覆盖度权重（默认 0.6）
  --beta        多样性权重（默认 0.3；设为 0 则完全禁用 CLIP）
  --gamma       清晰度权重（默认 0.1）

约束：
  --min_gap     帧索引最小间隔（无位姿时生效，默认 10）
  --min_move    位姿最小欧氏距离（有位姿时生效，默认 0.50 米）

输出：
  out_dir/selected.csv         选中清单（分项得分 + 总分）
  out_dir/selected_list.txt    仅文件名列表（便于后续脚本）
  out_dir/previews/…           覆盖可选：叠加 union mask 的预览（--save_previews）
"""

import argparse
import os
import glob
import json
import math
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

# 可选依赖：只有在需要保存预览或 Laplacian 时才用到
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pandas as pd
except Exception:
    pd = None


# ---------------------------
# 工具函数
# ---------------------------

def natural_key(s: str):
    """以“自然数顺序”排序文件名 path_0002.png < path_0010.png"""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def list_frames(frames_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG")
    files = [p for ext in exts for p in glob.glob(os.path.join(frames_dir, ext))]
    files = sorted(files, key=natural_key)
    return files


def load_masks_csv(masks_csv):
    """读取实例 CSV。要求列至少包含: image, mask, conf。
       返回 dict[image_name] -> [mask_path, ...]（按 conf 从高到低）"""
    if masks_csv is None:
        return defaultdict(list)
    if pd is None:
        print("[WARN] 未安装 pandas，跳过读取 masks_csv。", flush=True)
        return defaultdict(list)

    df = pd.read_csv(masks_csv)
    for col in ("image", "mask", "conf"):
        if col not in df.columns:
            raise ValueError(f"[ERR] masks_csv 缺失列: {col}")

    # 只保留存在的掩码文件
    df = df[df["mask"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))]
    # 按同帧内置信度降序
    df = df.sort_values(by=["image", "conf"], ascending=[True, False])

    groups = defaultdict(list)
    for _, row in df.iterrows():
        groups[str(row["image"])].append(str(row["mask"]))
    return groups


def union_mask_area(mask_paths, size_wh):
    """把同帧 mask 做 union，返回( union_ratio, union_bitmap[H,W] )。
       若没有 mask，则返回 (0, None)。"""
    W, H = size_wh
    if not mask_paths:
        return 0.0, None

    union = None
    for mp in mask_paths:
        try:
            m = Image.open(mp).convert("L")
        except Exception:
            continue
        if m.size != (W, H):
            m = m.resize((W, H), Image.NEAREST)
        m_np = (np.array(m) > 0)
        if union is None:
            union = m_np
        else:
            union |= m_np

    if union is None:
        return 0.0, None

    ratio = float(union.sum()) / float(W * H)
    # 转成 0/255 的 uint8 bitmap（后续预览用）
    union_u8 = (union.astype(np.uint8) * 255)
    return ratio, union_u8


def laplacian_sharpness(image_path):
    """Laplacian 方差作为清晰度指标，并做一个经验归一化映射到 0~1。"""
    try:
        im = Image.open(image_path).convert("RGB")
        im_np = np.array(im)
        if cv2 is None:
            # 简单后备：用 Sobel 近似，归一化较粗糙
            gx = np.abs(np.gradient(im_np.astype(np.float32), axis=1)).mean()
            gy = np.abs(np.gradient(im_np.astype(np.float32), axis=0)).mean()
            val = float(gx + gy) / 255.0
            return max(0.0, min(1.0, val))
        gray = cv2.cvtColor(im_np, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = lap.var()  # 典型场景 0~4000+，按经验尺度归一化
        # 使用一个压缩的映射：x / (x + c)，c 控制拐点，避免极端值
        c = 1500.0
        score = float(var) / (float(var) + c)
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


# ---------------------------
# CLIP 加载（open_clip / openai clip / none）
# ---------------------------

def load_clip(device="cuda"):
    # 1) open_clip
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model = model.to(device)
        model.eval()

        def embed_fn(pil_img: Image.Image):
            import torch
            x = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.squeeze(0).detach().cpu().numpy()

        print("[CLIP] open_clip: ViT-B-32 (LAION2B)", flush=True)
        return "open_clip", embed_fn
    except Exception:
        pass

    # 2) OpenAI CLIP
    try:
        import clip as clip_oa  # 若失败可：pip install git+https://github.com/openai/CLIP.git
        model, preprocess = clip_oa.load("ViT-B/32", device=device, jit=False)
        model.eval()

        def embed_fn(pil_img: Image.Image):
            import torch
            x = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.squeeze(0).detach().cpu().numpy()

        print("[CLIP] openai-clip: ViT-B/32", flush=True)
        return "openai_clip", embed_fn
    except Exception:
        print("[CLIP] 未找到可用的 CLIP（open_clip / openai-clip），将禁用语义多样性项。", flush=True)
        return "none", None


# ---------------------------
# 位姿解析（尽量宽松）
# ---------------------------

def parse_poses(poses_json_path):
    """支持几种常见格式，统一返回 dict[image_name] -> np.array([x,y,z])。"""
    if poses_json_path is None or not os.path.isfile(poses_json_path):
        return {}

    with open(poses_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    # 可能是列表
    if isinstance(data, list):
        for it in data:
            # { "image": "xxx.png", "T": [x,y,z] } 或 { "name":…, "position": [x,y,z] }
            name = it.get("image") or it.get("name") or it.get("file") or it.get("frame")
            pos = it.get("T") or it.get("position") or it.get("pos")
            if isinstance(name, str) and isinstance(pos, (list, tuple)) and len(pos) >= 3:
                out[os.path.basename(name)] = np.array(pos[:3], dtype=float)
    # 也可能是字典：{ "xxx.png": [x,y,z], ... } 或 { "xxx.png": {"position":[x,y,z]} }
    elif isinstance(data, dict):
        for k, v in data.items():
            name = os.path.basename(k)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                out[name] = np.array(v[:3], dtype=float)
            elif isinstance(v, dict):
                pos = v.get("T") or v.get("position") or v.get("pos")
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    out[name] = np.array(pos[:3], dtype=float)

    return out


# ---------------------------
# 选择逻辑
# ---------------------------

def greedy_select(frames, features, clip_feats, poses_xyz, args):
    """
    frames: [str] 图像路径
    features: dict
        - cov[i] 覆盖度 (0..1)
        - sal[i] 清晰度/显著性 (0..1)
        - idx[i] 原始索引
        - name[i] 文件名（仅名不带路径）
        - union_mask[i] (可选) union mask 的 0/255 uint8 图（预览用）
        - size[i] (W,H)
    clip_feats: np.ndarray [N, D] 或 None
    poses_xyz: dict[name] -> np.array([x,y,z]) 或 {}
    """
    N = len(frames)
    selected = []  # list of tuples: (i, score, parts_dict)
    selected_indices = []

    # 预处理：排序候选（先按覆盖度 + 清晰度合成一个基础分，从大到小遍历更快收敛）
    base_w = args.alpha + args.gamma + 1e-12
    prelim = []
    for i in range(N):
        a = float(features["cov"][i])
        b = float(features["sal"][i])
        prelim_score = (args.alpha * a + args.gamma * b) / base_w
        prelim.append((i, prelim_score))
    prelim.sort(key=lambda x: x[1], reverse=True)

    # 函数：检查与已选的“间距”是否满足（位姿或索引）
    def ok_spacing(i):
        if not selected_indices:
            return True
        my_name = features["name"][i]
        # 有位姿时用位移阈值，否则用索引间隔
        if poses_xyz:
            if my_name not in poses_xyz:
                # 没有该帧位姿，则退回用索引间隔
                my_idx = features["idx"][i]
                for j in selected_indices:
                    if abs(my_idx - features["idx"][j]) < args.min_gap:
                        return False
                return True
            my_p = poses_xyz[my_name]
            for j in selected_indices:
                sj_name = features["name"][j]
                if sj_name in poses_xyz:
                    dist = float(np.linalg.norm(my_p - poses_xyz[sj_name]))
                    if dist < args.min_move:
                        return False
            return True
        else:
            my_idx = features["idx"][i]
            for j in selected_indices:
                if abs(my_idx - features["idx"][j]) < args.min_gap:
                    return False
            return True

    # 贪心：每次挑得分最高且满足约束的
    while len(selected) < args.top_k and prelim:
        picked = None
        picked_score = -1e9
        picked_parts = None

        # 为了效率，每轮只看前若干个候选（例如 5*N^(1/2)），也可以遍历全部
        limit = max(64, int(math.sqrt(len(prelim)) * 8))
        for t, (i, _) in enumerate(prelim[:limit]):
            if not ok_spacing(i):
                continue
            # 分项分数
            A = float(features["cov"][i])
            B = float(features["sal"][i])
            # 多样性
            if args.beta > 1e-8 and clip_feats is not None and len(selected_indices) > 0:
                sims = []
                for j, _, _ in selected:
                    # 余弦相似度：特征已归一化
                    s = float(clip_feats[i].dot(clip_feats[j]))
                    sims.append(s)
                max_sim = max(sims) if sims else 0.0
                D = 1.0 - max_sim
            elif args.beta > 1e-8 and clip_feats is not None and len(selected_indices) == 0:
                D = 1.0
            else:
                D = 0.0

            score = args.alpha * A + args.gamma * B + args.beta * D

            if score > picked_score:
                picked = i
                picked_score = score
                picked_parts = dict(A=A, B=B, D=D)

        if picked is None:
            # 放宽：从剩余候选里，按 prelim 顺序挑第一个满足 spacing 的
            for i, _ in prelim:
                if ok_spacing(i):
                    A = float(features["cov"][i])
                    B = float(features["sal"][i])
                    if args.beta > 1e-8 and clip_feats is not None and len(selected_indices) > 0:
                        sims = [float(clip_feats[i].dot(clip_feats[j])) for j, _, _ in selected]
                        max_sim = max(sims) if sims else 0.0
                        D = 1.0 - max_sim
                    elif args.beta > 1e-8 and clip_feats is not None:
                        D = 1.0
                    else:
                        D = 0.0
                    picked = i
                    picked_score = args.alpha * A + args.gamma * B + args.beta * D
                    picked_parts = dict(A=A, B=B, D=D)
                    break

        if picked is None:
            break  # 没有满足约束的了

        selected.append((picked, picked_score, picked_parts))
        selected_indices.append(picked)
        # 从候选中剔除
        prelim = [(ii, sc) for (ii, sc) in prelim if ii != picked]

    # 按得分降序输出（也可保留时间顺序，视需求）
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected


# ---------------------------
# 预览叠加（可选）
# ---------------------------

def save_overlay_preview(img_path, union_mask_u8, out_path, color=(255, 0, 0, 120)):
    """把 union mask 叠到原图上，保存 RGBA 预览。"""
    try:
        im = Image.open(img_path).convert("RGBA")
        W, H = im.size
        if union_mask_u8 is None:
            im.save(out_path)
            return
        if union_mask_u8.shape != (H, W):
            um = Image.fromarray(union_mask_u8).resize((W, H), Image.NEAREST)
            union_mask_u8 = np.array(um)

        overlay = Image.new("RGBA", (W, H), (color[0], color[1], color[2], 0))
        alpha = Image.fromarray(union_mask_u8)
        overlay.putalpha(alpha)
        out = Image.alpha_composite(im, overlay)
        out.save(out_path)
    except Exception as e:
        print(f"[WARN] 预览失败 {img_path}: {e}", flush=True)


# ---------------------------
# 主流程
# ---------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    frames = list_frames(args.frames)
    if not frames:
        print(f"[ERR] 没有在 {args.frames} 找到图像。", file=sys.stderr)
        sys.exit(1)

    # 读掩码列表
    masks_group = load_masks_csv(args.masks_csv)

    # 读位姿（可选）
    poses_xyz = parse_poses(args.poses)

    # CLIP（可选）
    use_clip = args.beta > 1e-8
    clip_kind = "none"
    clip_embed_fn = None
    if use_clip:
        dev = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or "cuda" in args.device.lower() else args.device
        clip_kind, clip_embed_fn = load_clip(dev)
        if clip_kind == "none" or clip_embed_fn is None:
            use_clip = False
            args.beta = 0.0
    else:
        print("[INFO] beta=0，禁用 CLIP 多样性项。", flush=True)

    # 逐帧计算特征
    names = [os.path.basename(p) for p in frames]
    cov = np.zeros(len(frames), dtype=np.float32)
    sal = np.zeros(len(frames), dtype=np.float32)
    union_bitmaps = [None] * len(frames)
    sizes = []

    for i, fp in enumerate(frames):
        try:
            with Image.open(fp) as im:
                W, H = im.size
            sizes.append((W, H))
        except Exception:
            sizes.append((0, 0))

        # 覆盖度 & union mask
        mps = masks_group.get(names[i], [])
        a_ratio, u_mask = union_mask_area(mps, sizes[i]) if sizes[i] != (0, 0) else (0.0, None)
        cov[i] = float(a_ratio)
        union_bitmaps[i] = u_mask

        # 清晰度（Laplacian）
        sal[i] = laplacian_sharpness(fp)

        if (i + 1) % 200 == 0:
            print(f"[feat] {i+1}/{len(frames)}  cov_mean={cov[:i+1].mean():.3f}  sal_mean={sal[:i+1].mean():.3f}", flush=True)

    # 预计算 CLIP 特征（若启用）
    clip_feats = None
    if use_clip:
        clip_feats = []
        for i, fp in enumerate(frames):
            try:
                im = Image.open(fp).convert("RGB")
                feat = clip_embed_fn(im)
            except Exception:
                feat = None
            if feat is None:
                # 用 0 向量占位（归一化过的特征不应为零；此处仅容错）
                feat = np.zeros(512, dtype=np.float32)
            clip_feats.append(feat)
            if (i + 1) % 200 == 0:
                print(f"[clip] {i+1}/{len(frames)}", flush=True)
        clip_feats = np.stack(clip_feats).astype(np.float32)

    features = {
        "cov": cov,
        "sal": sal,
        "idx": np.arange(len(frames), dtype=int),
        "name": names,
        "union_mask": union_bitmaps,
        "size": sizes,
    }

    selected = greedy_select(frames, features, clip_feats, poses_xyz, args)

    # 输出 CSV / 列表
    sel_csv = os.path.join(args.out_dir, "selected.csv")
    sel_list = os.path.join(args.out_dir, "selected_list.txt")
    with open(sel_csv, "w", encoding="utf-8") as f:
        f.write("rank,image,score,coverage,saliency,diversity\n")
        for rank, (i, sc, parts) in enumerate(selected, 1):
            f.write(f"{rank},{names[i]},{sc:.6f},{parts['A']:.6f},{parts['B']:.6f},{parts['D']:.6f}\n")
    with open(sel_list, "w", encoding="utf-8") as f:
        for i, _, _ in selected:
            f.write(f"{names[i]}\n")

    print(f"[OK] 选中 {len(selected)} / {len(frames)} 帧：\n  -> {sel_csv}\n  -> {sel_list}", flush=True)

    # 预览（可选）
    if args.save_previews:
        prev_dir = os.path.join(args.out_dir, "previews")
        os.makedirs(prev_dir, exist_ok=True)
        for rank, (i, sc, parts) in enumerate(selected, 1):
            outp = os.path.join(prev_dir, f"{rank:03d}_{names[i]}")
            try:
                save_overlay_preview(frames[i], features["union_mask"][i], outp)
            except Exception as e:
                print(f"[WARN] 预览失败 {names[i]}: {e}", flush=True)
        print(f"[OK] 预览已保存 -> {prev_dir}", flush=True)


# ---------------------------
# CLI
# ---------------------------

def build_args():
    p = argparse.ArgumentParser("Auto Select Views (coverage + sharpness + CLIP diversity)")
    p.add_argument("--frames", required=True, help="帧图片目录")
    p.add_argument("--masks_csv", default=None, help="实例掩码 CSV（可选）")
    p.add_argument("--poses", default=None, help="位姿 JSON（可选）")
    p.add_argument("--out_dir", required=True, help="输出目录")

    p.add_argument("--top_k", type=int, default=60, help="选取数量")
    p.add_argument("--alpha", type=float, default=0.6, help="覆盖度权重")
    p.add_argument("--beta", type=float, default=0.3, help="多样性权重（0 则禁用 CLIP）")
    p.add_argument("--gamma", type=float, default=0.1, help="清晰度权重")

    p.add_argument("--min_gap", type=int, default=10, help="无位姿时：帧索引最小间隔")
    p.add_argument("--min_move", type=float, default=0.50, help="有位姿时：相邻选择最小位移（米）")

    p.add_argument("--device", type=str, default="cuda", help="CLIP 推理设备（cuda/cpu）")
    p.add_argument("--save_previews", action="store_true", help="保存 union mask 叠加预览")

    return p.parse_args()


if __name__ == "__main__":
    main(build_args())
