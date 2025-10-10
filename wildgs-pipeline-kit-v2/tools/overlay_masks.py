#!/usr/bin/env python3
import os, os.path as op, argparse, glob, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_boxes_from_txt(txtp, W, H):
    boxes=[]
    if not op.isfile(txtp): return boxes
    for ln in open(txtp):
        ps=ln.strip().split()
        if len(ps)<5: continue
        cx,cy,w,h=map(float,ps[1:5])
        px,py, pw,ph = cx*W, cy*H, w*W, h*H
        x0,y0 = int(round(px-pw/2)), int(round(py-ph/2))
        x1,y1 = int(round(px+pw/2)), int(round(py+ph/2))
        boxes.append((x0,y0,x1,y1))
    return boxes

def load_mask_bboxes(mask_dir, stem):
    bbs=[]
    for mp in glob.glob(op.join(mask_dir, f"{stem}_inst*.png")):
        m = (Image.open(mp).convert("L"))
        a = np.array(m)
        ys,xs = np.where(a>127)
        if xs.size==0: continue
        bbs.append((int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())))
    return bbs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True, help="YOLO txt dir")
    ap.add_argument("--masks",  required=True, help="rect masks dir")
    ap.add_argument("--out",    required=True)
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for ip in sorted(glob.glob(op.join(args.images,"*.jpg"))):
        im=Image.open(ip).convert("RGB")
        W,H=im.size
        stem=op.splitext(op.basename(ip))[0]
        # 优先用 mask 的 bbox，兜底用 YOLO 框
        bbs = load_mask_bboxes(args.masks, stem)
        if not bbs:
            bbs = load_boxes_from_txt(op.join(args.labels, stem+".txt"), W,H)

        overlay = Image.new("RGBA",(W,H),(0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        for (x0,y0,x1,y1) in bbs:
            draw.rectangle([x0,y0,x1,y1], outline=(255,255,255,255), width=2)
            draw.rectangle([x0,y0,x1,y1], fill=(0,128,255,60))
        out = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        out.save(op.join(args.out, os.path.basename(ip)))
    print("[OK] wrote:", args.out)

if __name__=="__main__":
    main()
