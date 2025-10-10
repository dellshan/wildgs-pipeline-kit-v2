#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoView-Lite: 从帧目录与可选 transforms.json 生成：
- per-frame 指标表 metrics.csv（清晰度/亮度/对比度/边缘/熵/可选语义）
- 归一化后的联合得分 + 多样性（余弦相似度）约束的 Top-K 精选 selection.csv
- 直方图/散点图：hist_sharpness.png, hist_brightness.png, scatter_sharpness_vs_entropy.png
- 顶视相机轨迹图：topdown_path.png（需 transforms_train.json）
- 精选帧拼贴：storyboard_selected.jpg
- 精选视频：selected_reel.mp4（OpenCV 写出）
"""

import os, sys, json, math, glob, argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 可选：CLIP 语义（若不可用则自动跳过） ----------
def try_load_clip():
    try:
        import open_clip
        import torch
        model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        return dict(ok=True, open_clip=open_clip, torch=torch, model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)
    except Exception as e:
        return dict(ok=False, err=str(e))

def clip_score(img_bgr, clip_ctx, text_prompt):
    if not clip_ctx or not clip_ctx.get("ok", False): return None
    open_clip, torch = clip_ctx["open_clip"], clip_ctx["torch"]
    model, preprocess, tokenizer, device = clip_ctx["model"], clip_ctx["preprocess"], clip_ctx["tokenizer"], clip_ctx["device"]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    with torch.no_grad():
        image = preprocess(pil).unsqueeze(0).to(device)
        text = tokenizer([text_prompt]).to(device)
        image_features = model.encode_image(image)
        text_features  = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
        sim = (image_features @ text_features.T).squeeze().item()
        return float(sim)  # [-1, 1] 左右
    return None

# ---------- 基础图像指标 ----------
def image_metrics(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    shp  = cv2.Laplacian(gray, cv2.CV_64F).var()
    bri  = float(gray.mean())
    con  = float(gray.std())
    # Shannon entropy
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p = hist / (hist.sum() + 1e-8)
    ent = float(-(p*(np.log2(p+1e-12))).sum())
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float((edges>0).mean())
    # HSV sat mean
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = float(hsv[...,1].mean())
    return dict(sharpness=shp, brightness=bri, contrast=con, entropy=ent, edge_ratio=edge_ratio, sat_mean=sat)

# ---------- 读 transforms_*（Nerfstudio 标准：camera_to_world） ----------
def yaw_from_c2w(c2w):
    # yaw around up-axis (assuming y-up or z-up? Nerfstudio默认y-up，此处简单取水平朝向)
    R = np.array(c2w, dtype=np.float64)[:3,:3]
    # 以世界坐标的前向为 -Z（常见渲染约定），取相机朝向向量
    fwd = -R[:,2]
    yaw = math.degrees(math.atan2(fwd[0], fwd[2] + 1e-9))  # x vs z
    return yaw

def load_transforms_json(path):
    if not path or not os.path.isfile(path): return {}
    data = json.load(open(path,'r'))
    frames = data.get("frames", [])
    table = {}
    for f in frames:
        fp = f.get("file_path","")
        base = os.path.basename(fp)
        c2w  = np.array(f.get("transform_matrix"), dtype=np.float64)
        pos  = c2w[:3,3].tolist()
        yaw  = yaw_from_c2w(c2w)
        table[base] = dict(pos_x=pos[0], pos_y=pos[1], pos_z=pos[2], yaw_deg=float(yaw))
    return table

# ---------- 简单选择：归一化加权 + 多样性（余弦相似度） ----------
def minmax_norm(x):
    a = np.array(x, dtype=np.float64)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi - lo < 1e-9: return np.zeros_like(a)
    return (a - lo) / (hi - lo)

def greedy_select(feats, base_score, K=40, lambda_div=0.6):
    # feats: ndarray [N, D]（用于相似度）
    # base_score: ndarray [N]
    N = len(base_score)
    selected, mask = [], np.zeros(N, dtype=bool)
    # 预归一化特征
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True)+1e-9)
    sims  = feats @ feats.T  # 余弦相似度
    for _ in range(min(K, N)):
        best_i, best_v = -1, -1e9
        for i in range(N):
            if mask[i]: continue
            if len(selected)==0:
                v = base_score[i]
            else:
                # 惩罚与已选的最大相似
                penalty = lambda_div * np.max(sims[i, selected])
                v = base_score[i] - penalty
            if v > best_v:
                best_v, best_i = v, i
        if best_i < 0: break
        selected.append(best_i)
        mask[best_i] = True
    return selected

# ---------- 拼贴与视频 ----------
def make_contact_sheet(paths, out_path, tile_cols=6, tile_rows=None, resize=(640,480), caption=False):
    if tile_rows is None:
        tile_rows = math.ceil(len(paths)/tile_cols)
    W, H = resize
    sheet = Image.new("RGB", (tile_cols*W, tile_rows*H), (255,255,255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = None
    for idx, p in enumerate(paths[:tile_cols*tile_rows]):
        try:
            img = Image.open(p).convert("RGB").resize((W,H))
        except:
            img = Image.new("RGB",(W,H),(200,200,200))
        x = (idx % tile_cols)*W
        y = (idx // tile_cols)*H
        sheet.paste(img, (x,y))
        if caption:
            text = os.path.basename(p)
            draw.text((x+6,y+6), text, fill=(0,0,0), font=font)
    sheet.save(out_path, quality=95)

def write_video(paths, out_mp4, fps=15):
    # 用第一帧定尺寸
    for p in paths:
        if os.path.isfile(p):
            f0 = cv2.imread(p)
            if f0 is not None: break
    if f0 is None: return
    h,w = f0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (w,h))
    for p in paths:
        img = cv2.imread(p)
        if img is None: continue
        vw.write(img)
    vw.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="目录：.../renders/dataset_train/train/rgb")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--transforms_json", default="", help="可选：.../exports/dataset_meta/transforms_train.json")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--wc", type=float, default=0.0, help="覆盖权重（若无mask则为0）")
    ap.add_argument("--ws", type=float, default=0.6, help="清晰度权重")
    ap.add_argument("--wm", type=float, default=0.4, help="语义权重（若无CLIP则忽略）")
    ap.add_argument("--lambda_div", type=float, default=0.6, help="多样性惩罚系数")
    ap.add_argument("--prompt", default="indoor navigation obstacles: chair, table, cable, toy, step, threshold, clutter",
                    help="可选：CLIP 文本提示")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # 帧列表（按文件名序）
    exts = ("*.jpg","*.png","*.jpeg")
    frames = []
    for e in exts: frames += glob.glob(os.path.join(args.frames_dir, e))
    frames = sorted(frames)
    if not frames:
        print("No frames found in", args.frames_dir); sys.exit(1)

    # 读位姿
    pose_tbl = load_transforms_json(args.transforms_json)

    # CLIP（可选）
    clip_ctx = try_load_clip()
    if not clip_ctx.get("ok", False):
        print("[Info] CLIP unavailable:", clip_ctx.get("err",""))
    else:
        print("[Info] CLIP ready.")

    rows, emb_list = [], []
    for idx, fpath in enumerate(frames):
        img = cv2.imread(fpath)
        if img is None: continue
        m = image_metrics(img)
        # 可选语义（归一化时再一起做）
        sem = clip_score(img, clip_ctx, args.prompt) if clip_ctx.get("ok", False) else None

        base = os.path.basename(fpath)
        pose = pose_tbl.get(base, {})
        row = dict(
            index=idx, file=base, path=fpath,
            width=img.shape[1], height=img.shape[0],
            sharpness=m["sharpness"], brightness=m["brightness"], contrast=m["contrast"],
            entropy=m["entropy"], edge_ratio=m["edge_ratio"], sat_mean=m["sat_mean"],
            sem_clip=sem,
            pos_x=pose.get("pos_x",np.nan), pos_y=pose.get("pos_y",np.nan),
            pos_z=pose.get("pos_z",np.nan), yaw_deg=pose.get("yaw_deg",np.nan)
        )
        rows.append(row)

        # 作为多样性特征：优先用 CLIP 图像特征，不可用则用颜色直方图
        if clip_ctx.get("ok", False):
            # 轻量化：缩图+归一化灰度直方图以避免额外算子——此处简化为SIFT替代将过重；这里就用直方图
            img_small = cv2.resize(img, (64,64))
            hist = cv2.calcHist([img_small],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]).ravel()
            emb = hist / (np.linalg.norm(hist)+1e-9)
        else:
            img_small = cv2.resize(img, (64,64))
            hist = cv2.calcHist([img_small],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]).ravel()
            emb = hist / (np.linalg.norm(hist)+1e-9)
        emb_list.append(emb)

    df = pd.DataFrame(rows)
    # 归一化若干列
    cols_norm = ["sharpness","brightness","contrast","entropy","edge_ratio","sat_mean"]
    for c in cols_norm:
        df[c+"_n"] = minmax_norm(df[c].values)
    if "sem_clip" in df.columns and df["sem_clip"].notna().any():
        df["sem_clip_n"] = minmax_norm(df["sem_clip"].fillna(df["sem_clip"].min()).values)
    else:
        df["sem_clip_n"] = 0.0

    # 基础打分（不含覆盖项，覆盖在这个lite版里没有mask就=0）
    base_score = (args.ws*df["sharpness_n"].values +
                  args.wm*df["sem_clip_n"].values +
                  args.wc*0.0)

    feats = np.stack(emb_list, axis=0)
    sel_idx = greedy_select(feats, base_score, K=args.topk, lambda_div=args.lambda_div)
    df["selected"] = False
    df.loc[sel_idx, "selected"] = True
    df["score"] = base_score

    # 保存表
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)
    df_sel = df[df["selected"]].sort_values("index")
    df_sel.to_csv(os.path.join(args.out_dir, "selection.csv"), index=False)
    print("✅ CSV saved:", csv_path)

    # 统计图
    plt.figure(); plt.hist(df["sharpness"].values, bins=60); plt.xlabel("Laplacian variance (sharpness)"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"hist_sharpness.png")); plt.close()

    plt.figure(); plt.hist(df["brightness"].values, bins=60); plt.xlabel("brightness (gray mean)"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"hist_brightness.png")); plt.close()

    plt.figure(); plt.scatter(df["sharpness_n"].values, df["entropy_n"].values, s=6, alpha=0.5)
    plt.xlabel("sharpness (norm)"); plt.ylabel("entropy (norm)"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"scatter_sharpness_vs_entropy.png")); plt.close()

    # 顶视路径（需位姿）
    if df["pos_x"].notna().any() and df["pos_z"].notna().any():
        plt.figure()
        plt.plot(df["pos_x"], df["pos_z"], '-', lw=1, alpha=0.6, label="path")
        sel = df["selected"].values
        plt.scatter(df["pos_x"][sel], df["pos_z"][sel], s=12, label="selected")
        plt.axis('equal'); plt.xlabel("x"); plt.ylabel("z"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir,"topdown_path.png")); plt.close()

    # 精选拼贴 + 视频
    sel_paths = df_sel["path"].tolist()
    make_contact_sheet(sel_paths, os.path.join(args.out_dir,"storyboard_selected.jpg"),
                       tile_cols=6, resize=(640,480), caption=True)
    write_video(sel_paths, os.path.join(args.out_dir,"selected_reel.mp4"), fps=15)
    print("✅ Figures & reel saved to:", args.out_dir)

if __name__ == "__main__":
    main()
