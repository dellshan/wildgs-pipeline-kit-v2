#!/usr/bin/env python3
import os, os.path as op, argparse, glob, csv
import numpy as np
from PIL import Image
from tqdm import tqdm

def lap_var(gray):
    # 3x3 Laplacian kernel
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    g = gray.astype(np.float32)
    pad = np.pad(g, 1, mode="edge")
    L = (pad[:-2,1:-1]*k[0,0] + pad[:-2,2:]*k[0,1] + pad[:-2,:-2]*k[0,2] +
         pad[1:-1,1:-1]*k[1,1] +
         pad[2:,1:-1]*k[2,0] + pad[2:,2:]*k[2,1] + pad[2:,:-2]*k[2,2] +
         pad[1:-1,2:]*k[1,2] + pad[1:-1,:-2]*k[1,0])  # expanded for speed
    return float(L.var())

def unite_masks(mask_paths, size):
    if not mask_paths: return 0.0
    H,W = size[1], size[0]
    acc = np.zeros((H,W), dtype=np.uint8)
    for mp in mask_paths:
        m = np.array(Image.open(mp).convert("L"))
        acc |= (m > 127).astype(np.uint8)
    return float(acc.sum()) / (H*W)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    imgs = sorted(glob.glob(op.join(args.images, "*.jpg")) + glob.glob(op.join(args.images, "*.png")))
    rows = [("image","coverage","sharpness")]
    shps = []
    covs = []
    tmp = []
    for ip in tqdm(imgs, desc="features"):
        name = os.path.basename(ip)
        stem = os.path.splitext(name)[0]
        im = Image.open(ip).convert("RGB")
        W,H = im.size
        gray = np.array(im.convert("L"))
        shp = lap_var(gray)
        mps = sorted(glob.glob(op.join(args.masks, f"{stem}_inst*.png")))
        cov = unite_masks(mps, im.size)
        tmp.append((name, cov, shp))
        covs.append(cov); shps.append(shp)

    # min-max normalize sharpness for stability; keep coverage as-is
    smin, smax = (min(shps), max(shps)) if shps else (0,1)
    for name, cov, shp in tmp:
        nshp = 0.0 if smax==smin else (shp - smin)/(smax - smin)
        rows.append((name, f"{cov:.6f}", f"{nshp:.6f}"))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv,"w",newline="") as f:
        csv.writer(f).writerows(rows)
    print("[OK] wrote", args.out_csv, "| frames=", len(rows)-1)

if __name__ == "__main__":
    main()
