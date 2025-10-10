#!/usr/bin/env python3
import os, os.path as op, argparse, glob, csv
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

def load_img_size(p):
    with Image.open(p) as im:
        return im.size  # (W,H)

def yolo_line_to_px(ln, W, H):
    parts = ln.strip().split()
    if len(parts) < 5:
        return None
    c = int(float(parts[0]))
    cx,cy,w,h = map(float, parts[1:5])
    conf = float(parts[5]) if len(parts) >= 6 else 1.0
    # 归一化 -> 像素
    px, py = cx*W, cy*H
    pw, ph = w*W, h*H
    x0 = max(0, int(round(px - pw/2)))
    y0 = max(0, int(round(py - ph/2)))
    x1 = min(W-1, int(round(px + pw/2)))
    y1 = min(H-1, int(round(py + ph/2)))
    if x1 <= x0 or y1 <= y0:
        return None
    area = (x1-x0+1)*(y1-y0+1)
    return c, conf, (x0,y0,x1,y1), area

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir of RGBs")
    ap.add_argument("--labels", required=True, help="dir of YOLO txts")
    ap.add_argument("--out",    required=True, help="output dir for masks")
    ap.add_argument("--min_area", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = [("image","inst_id","mask","cls","conf","x0","y0","x1","y1","area_px")]

    txts = sorted(glob.glob(op.join(args.labels, "*.txt")))
    for p in tqdm(txts, desc="boxes->masks"):
        stem = op.splitext(op.basename(p))[0]
        # 找同名图片（常见 .jpg/.png）
        for ext in (".jpg",".JPG",".png",".PNG",".jpeg",".JPEG"):
            imgp = op.join(args.images, stem+ext)
            if op.isfile(imgp): break
        else:
            # 没在 images 里，尝试递归找
            matches = glob.glob(op.join(args.images, "**", stem+".*"), recursive=True)
            imgp = matches[0] if matches else None
        if not imgp or not op.isfile(imgp):
            continue

        W,H = load_img_size(imgp)
        with open(p) as f:
            lines = [ln for ln in f if ln.strip()]

        inst_k = 0
        for ln in lines:
            item = yolo_line_to_px(ln, W, H)
            if item is None: 
                continue
            c, conf, (x0,y0,x1,y1), area = item
            if area < args.min_area:
                continue

            mask = Image.new("L", (W,H), 0)
            ImageDraw.Draw(mask).rectangle([x0,y0,x1,y1], fill=255)
            mname = f"{stem}_inst{inst_k:04d}.png"
            mask.save(op.join(args.out, mname))
            rows.append((stem, inst_k, mname, c, conf, x0,y0,x1,y1, area))
            inst_k += 1

    # 写 instances.csv（和之前 Mip360 流程一致）
    csv_out = op.join(op.dirname(args.out), "instances.csv")
    with open(csv_out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[OK] wrote {csv_out} | instances= {len(rows)-1}")

if __name__ == "__main__":
    main()
