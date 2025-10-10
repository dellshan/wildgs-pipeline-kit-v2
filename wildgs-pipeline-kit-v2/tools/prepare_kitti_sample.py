#!/usr/bin/env python3
import os, os.path as op, argparse, glob, shutil
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/kitti/raw/training/image_2")
    ap.add_argument("--out", default="outputs/kitti/sample")
    ap.add_argument("--num", type=int, default=200)
    args = ap.parse_args()

    imgs = sorted(glob.glob(op.join(args.root, "*.png")) + glob.glob(op.join(args.root, "*.jpg")))
    if not imgs:
        raise SystemExit(f"No images under {args.root}")
    imgs = imgs[:args.num]
    os.makedirs(op.join(args.out, "images"), exist_ok=True)

    rows = []
    for p in tqdm(imgs, desc="copy"):
        dst = op.join(args.out, "images", op.basename(p))
        shutil.copy2(p, dst)
        rows.append((op.basename(p), 0, "KITTI", "image_2", 0, 0, 0, 0))

    with open(op.join(args.out, "frames_views.csv"), "w") as f:
        f.write("image,ts,scene,cam,tx,ty,tz,yaw_deg\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print("[OK] KITTI sample ->", args.out, "| images=", len(rows))

if __name__ == "__main__":
    main()