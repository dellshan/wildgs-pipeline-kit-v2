#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, glob, argparse, math, uuid
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# --- LSeg wrapper (沿用你已有的 pipeline/lseg/ )
from pipeline.lseg.lseg_net import LSegNet

# --------- Utils ----------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def to_numpy_img(pil_img: Image.Image) -> np.ndarray:
    # SAM/SAM2 期望 RGB np.uint8
    return np.array(pil_img.convert("RGB"))

def color_palette(n):
    rng = np.random.default_rng(0)
    cols = rng.integers(64, 255, size=(n, 3), dtype=np.uint8)
    return [tuple(map(int, c)) for c in cols]

def overlay_mask(img: Image.Image, mask: np.ndarray, color=(255,0,0), alpha=0.5):
    """mask: [H,W] bool"""
    ov = img.copy()
    draw = ImageDraw.Draw(ov, "RGBA")
    H, W = mask.shape
    # 快速画法：把整幅图按照 mask 着色
    overlay = Image.new("RGBA", (W, H), color + (int(255*alpha),))
    ov.paste(overlay, (0,0), Image.fromarray((mask*255).astype(np.uint8)))
    return ov

# --------- LSeg: heatmap per prompt ----------
def load_lseg(ckpt_path: str, backbone: str="clip_vitl16_384", im_size: int=512, device="cuda"):
    model = LSegNet(backbone=backbone, features=256, crop_size=im_size,
                    arch_option=0, block_depth=0, activation="lrelu")
    # 允许 ckpt 内缺少 logit_log（你之前已见到 Missing keys 的提示）
    try:
        model.load(ckpt_path)
    except Exception as e:
        print(f"[WARN] load ckpt with torch.load fallback: {e}")

    model.eval().to(device)
    tfm = T.Compose([
        T.Resize(im_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    return tfm, model

@torch.no_grad()
def lseg_heatmaps_for_prompts(pil_img: Image.Image,
                              prompts_en: List[str],
                              preproc, model: LSegNet,
                              device="cuda") -> torch.Tensor:
    """
    返回: [K,H,W] 的 logits（或相对分数），K=len(prompts_en)
    之后会按需做 sigmoid/softmax；这里当作多标签用 sigmoid。
    """
    x = preproc(pil_img).unsqueeze(0).to(device)  # [1,3,S,S]
    # LSeg.forward 支持 labelset=list[str]，返回 [B,K,h,w]
    out = model(x, labelset=prompts_en)  # logits
    out = F.interpolate(out, size=pil_img.size[::-1], mode="bilinear", align_corners=False)  # -> [1,K,H,W]
    out = out[0]  # [K,H,W]
    return out

def find_peaks_from_heatmap(hm: torch.Tensor, k: int=5, min_val: float=0.35, nms_ks: int=9) -> List[Tuple[int,int,float]]:
    """
    hm: [H,W] torch, 已是 [0,1] 概率
    返回 top-k 峰值点 (x,y,score)
    """
    H, W = hm.shape
    # 简单 NMS: 与 maxpool 相等的位置视为峰
    mp = F.max_pool2d(hm.unsqueeze(0).unsqueeze(0), kernel_size=nms_ks, stride=1, padding=nms_ks//2)[0,0]
    peaks = (hm == mp) & (hm >= min_val)
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    scores = hm[ys, xs]
    if scores.numel() == 0:
        return []
    topv, topi = torch.topk(scores, k=min(k, scores.numel()))
    pts = []
    for v, idx in zip(topv.tolist(), topi.tolist()):
        y = int(ys[idx].item()); x = int(xs[idx].item())
        pts.append((x,y,float(v)))
    return pts

# --------- SAM / SAM2 wrapper ----------
class AnySAMPredictor:
    def __init__(self, backend: str, ckpt: str, sam2_cfg: str|None, device="cuda"):
        self.backend = backend
        self.device = device

        if backend == "sam2":
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except Exception as e:
                raise RuntimeError(f"SAM2 import failed: {e}\n请先安装 SAM2，或改用 --sam_backend sam。")
            if sam2_cfg is None:
                raise RuntimeError("使用 sam2 时需要提供 --sam2_cfg（官方 yaml 配置）。")
            model = build_sam2(checkpoint=ckpt, model_cfg=sam2_cfg, device=device)
            self.predictor = SAM2ImagePredictor(model)
            self.kind = "sam2"
        else:
            # SAM v1 兜底
            try:
                from segment_anything import sam_model_registry, SamPredictor
            except Exception as e:
                raise RuntimeError(f"SAM import failed: {e}\n请 pip install git+https://github.com/facebookresearch/segment-anything")
            # 自动从 ckpt 名推测型号；不行就用 vit_h
            model_type = "vit_h"
            for t in ["vit_h","vit_l","vit_b"]:
                if t in os.path.basename(ckpt):
                    model_type = t; break
            sam = sam_model_registry[model_type](checkpoint=ckpt).to(device)
            self.predictor = SamPredictor(sam)
            self.kind = "sam"

    def set_image(self, image_rgb: np.ndarray):
        self.predictor.set_image(image_rgb)

    def predict_from_points(self, point_coords: np.ndarray, point_labels: np.ndarray):
        """
        返回 (masks[...], scores[...])
        SAM2/SAM1 的接口都类似：point_coords: [N,2] (x,y), point_labels: [N] 1=fg
        """
        if self.kind == "sam2":
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords.astype(np.float32),
                point_labels=point_labels.astype(np.int32),
                multimask_output=True
            )
        else:
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords.astype(np.float32),
                point_labels=point_labels.astype(np.int32),
                multimask_output=True
            )
        # masks: [M,H,W] bool
        return masks, scores

# --------- Main pipeline ----------
def zh_en_default_map():
    # 与你 lseg_describe 中的标签基本一致，可按需增删
    return {
        "桌子":"table","书桌":"desk",
        "杯子":"cup","马克杯":"mug","瓶子":"bottle",
        "书":"book","笔记本":"notebook","纸张":"paper",
        "手机":"phone","遥控器":"remote control",
        "笔":"pen","马克笔":"marker","胶带":"tape","剪刀":"scissors",
        "工具箱":"toolbox","盒子":"box","包":"bag","背包":"backpack",
        "电脑":"computer","笔记本电脑":"laptop","显示器":"monitor","屏幕":"screen",
        "相机":"camera","三脚架":"tripod","键盘":"keyboard","鼠标":"mouse",
        "人":"person"
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", default="output/inst")
    # LSeg
    ap.add_argument("--lseg_ckpt", required=True)
    ap.add_argument("--lseg_backbone", default="clip_vitl16_384")
    ap.add_argument("--lseg_size", type=int, default=512)
    # 类别选择（中文逗号分隔），为空则用默认
    ap.add_argument("--labels", type=str, default="杯子,马克杯,瓶子,书,纸张,笔,马克笔,工具箱,盒子,包,背包,笔记本电脑,相机,三脚架")
    ap.add_argument("--table_label", type=str, default="桌子")
    # 阈值 & 点数
    ap.add_argument("--table_th", type=float, default=0.40)
    ap.add_argument("--peak_k", type=int, default=5)
    ap.add_argument("--peak_th", type=float, default=0.35)
    ap.add_argument("--min_area", type=int, default=200)
    # SAM / SAM2
    ap.add_argument("--sam_backend", choices=["sam2","sam"], default="sam2")
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--sam2_cfg", type=str, default=None, help="SAM2 的模型 yaml，比如 sam2_hiera_t.yaml")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 准备输出目录
    vis_dir = ensure_dir(os.path.join(args.out_dir, "vis"))
    mask_dir = ensure_dir(os.path.join(args.out_dir, "masks"))
    ensure_dir(args.out_dir)

    # 读帧
    frames = []
    for e in ("*.png","*.jpg","*.jpeg","*.JPG","*.PNG"):
        frames += glob.glob(os.path.join(args.frames_dir, e))
    frames = sorted(frames)
    if not frames:
        print(f"[ERR] no images under {args.frames_dir}"); sys.exit(1)

    # LSeg
    preproc, lseg = load_lseg(args.lseg_ckpt, args.lseg_backbone, args.lseg_size, device=device)

    # 类别映射
    ZH2EN = zh_en_default_map()
    target_zh = [x.strip() for x in args.labels.split(",") if x.strip()]
    target_en = [ZH2EN.get(z, z) for z in target_zh]
    table_en = ZH2EN.get(args.table_label, args.table_label)

    # SAM / SAM2
    predictor = AnySAMPredictor(args.sam_backend, args.sam_ckpt, args.sam2_cfg, device=device)

    # 输出 JSON
    js_out = {"frames": []}
    palette = color_palette(max(1, len(target_en)+1))

    for idx, fp in enumerate(frames):
        pil = Image.open(fp).convert("RGB")
        np_img = to_numpy_img(pil)

        # 1) LSeg 热图
        prompts = [table_en] + target_en
        logits = lseg_heatmaps_for_prompts(pil, prompts, preproc, lseg, device=device)  # [K,H,W]
        probs = torch.sigmoid(logits)  # 多标签：Sigmoid
        table_prob = probs[0]
        obj_probs = probs[1:]  # 对应 target_en

        # 2) 桌子掩码
        table_mask = (table_prob >= args.table_th)
        table_mask_np = table_mask.cpu().numpy().astype(np.uint8)

        # 3) 选峰值点（限制在桌面上）
        predictor.set_image(np_img)
        instances = []
        vis = pil.copy()
        color_i = 0

        for cls_i, (zh, en) in enumerate(zip(target_zh, target_en)):
            hm = obj_probs[cls_i]
            hm_on_table = hm * table_mask.float()
            pts = find_peaks_from_heatmap(hm_on_table, k=args.peak_k, min_val=args.peak_th)

            for (x,y,score) in pts:
                point_coords = np.array([[x, y]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)
                masks, scores = predictor.predict_from_points(point_coords, point_labels)

                if masks is None or len(masks)==0:
                    continue
                # 取最高分那一张
                best = int(np.argmax(scores))
                m = masks[best].astype(np.uint8)
                area = int(m.sum())
                if area < args.min_area:
                    continue
                # 只保留与桌面相交的部分（可选）
                m = (m & table_mask_np).astype(np.uint8)
                if m.sum() < args.min_area:
                    continue

                ins_id = str(uuid.uuid4())[:8]
                mask_fp = os.path.join(mask_dir, f"{os.path.basename(fp)}.{zh}.{ins_id}.png")
                Image.fromarray((m*255).astype(np.uint8)).save(mask_fp)

                # 叠加可视化
                vis = overlay_mask(vis, m.astype(bool), color=palette[(color_i+cls_i)%len(palette)], alpha=0.45)

                # bbox
                ys, xs = np.where(m>0)
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

                instances.append({
                    "cls_zh": zh,
                    "cls_en": en,
                    "point": [int(x), int(y)],
                    "seed_score": float(score),
                    "sam_score": float(scores[best]),
                    "area": int(area),
                    "bbox": [x0, y0, x1, y1],
                    "mask": os.path.relpath(mask_fp, start=args.out_dir)
                })

        vis_fp = os.path.join(vis_dir, os.path.basename(fp))
        vis.save(vis_fp)

        js_out["frames"].append({
            "frame": os.path.basename(fp),
            "instances": instances,
            "vis": os.path.relpath(vis_fp, start=args.out_dir)
        })

        print(f"[{idx+1}/{len(frames)}] {os.path.basename(fp)} -> {len(instances)} instances")

    out_json = os.path.join(args.out_dir, "instances.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(js_out, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved: {out_json}")
    print(f"[OK] vis dir: {vis_dir}")
    print(f"[OK] masks  : {mask_dir}")

if __name__ == "__main__":
    main()
