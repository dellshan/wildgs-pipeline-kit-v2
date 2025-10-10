#!/usr/bin/env python3
import os, os.path as op, argparse, glob, cv2

def denorm(x, w): return max(0, min(int(round(x*w)), w-1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img_paths = {op.splitext(op.basename(p))[0]: p
                 for p in glob.glob(op.join(args.images, "*.*"))}

    for lp in sorted(glob.glob(op.join(args.labels, "*.txt"))):
        stem = op.splitext(op.basename(lp))[0]
        ip = img_paths.get(stem)
        if not ip: continue
        im = cv2.imread(ip)
        H, W = im.shape[:2]
        for ln in open(lp):
            parts = ln.strip().split()
            if len(parts) < 5: continue
            # cls cx cy w h [conf]
            cx, cy, bw, bh = map(float, parts[1:5])
            x = denorm(cx - bw/2, W); y = denorm(cy - bh/2, H)
            x2 = denorm(cx + bw/2, W); y2 = denorm(cy + bh/2, H)
            cv2.rectangle(im, (x,y), (x2,y2), (0,255,0), 2)
        cv2.imwrite(op.join(args.out, op.basename(ip)), im)
    print("[OK] wrote vis to", args.out)

if __name__ == "__main__":
    main()
