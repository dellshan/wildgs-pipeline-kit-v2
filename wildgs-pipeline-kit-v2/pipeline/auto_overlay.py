# pipeline/auto_overlay.py
import os, csv, numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def main(frames_dir, masks_csv, selected_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mask_map = {}
    with open(masks_csv,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if float(r["conf"])<0.8: continue
            mask_map.setdefault(r["image"],[]).append(r["mask"])

    with open(selected_csv,newline='',encoding='utf-8') as f:
        rows=[r for r in csv.DictReader(f)]
    for r in rows:
        img=r["image"]; fp=os.path.join(frames_dir, img)
        im=Image.open(fp).convert("RGBA"); W,H=im.size
        overlay=np.zeros((H,W), np.uint8)
        for m in mask_map.get(img,[]):
            overlay=np.maximum(overlay, np.array(Image.open(m)))
        color=Image.new("RGBA",(W,H),(255,0,0,0)); alpha=Image.fromarray(overlay)
        color.putalpha(alpha)
        out=Image.alpha_composite(im, color)
        out.save(os.path.join(out_dir, f"ov_{Path(img).stem}.png"))

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks_csv", required=True)
    ap.add_argument("--selected_csv", required=True)
    ap.add_argument("--out_dir", default="output/auto_views/overlays")
    a=ap.parse_args()
    main(a.frames, a.masks_csv, a.selected_csv, a.out_dir)
