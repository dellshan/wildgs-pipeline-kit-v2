#!/usr/bin/env python3
import os, os.path as op, argparse, glob, cv2, numpy as np
from tqdm import tqdm

def load_img_size(img_path):
    im = cv2.imread(img_path)
    if im is None: raise RuntimeError(f"fail to read {img_path}")
    h, w = im.shape[:2]
    return w, h

def yolo_to_px(box, W, H):
    # YOLO: cx cy w h (normalized) -> x1,y1,x2,y2 (pixel, clamped)
    cx, cy, bw, bh = box
    x1 = max(0, min(int(round((cx - bw/2) * W)), W-1))
    y1 = max(0, min(int(round((cy - bh/2) * H)), H-1))
    x2 = max(0, min(int(round((cx + bw/2) * W)), W-1))
    y2 = max(0, min(int(round((cy + bh/2) * H)), H-1))
    return x1, y1, x2, y2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)  # will write masks/ and instances.csv
    ap.add_argument("--min_conf", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mdir = op.join(args.out, "masks")
    os.makedirs(mdir, exist_ok=True)

    img_map = {op.splitext(op.basename(p))[0]: p
               for p in glob.glob(op.join(args.images, "*"))}

    rows = [("image","inst_id","cls","conf","x1","y1","x2","y2","mask_path")]
    for lp in tqdm(sorted(glob.glob(op.join(args.labels, "*.txt"))), desc="boxes->masks"):
        stem = op.splitext(op.basename(lp))[0]
        ip = img_map.get(stem)
        if not ip: continue
        W,H = load_img_size(ip)
        inst_id = 0
        for ln in open(lp):
            parts = ln.strip().split()
            if len(parts) < 5: continue
            cls = int(parts[0])
            cx,cy,bw,bh = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            if conf < args.min_conf: continue
            x1,y1,x2,y2 = yolo_to_px((cx,cy,bw,bh), W,H)
            # rectangle mask
            mask = np.zeros((H,W), dtype=np.uint8)
            mask[y1:y2+1, x1:x2+1] = 255
            mname = f"{stem}_{inst_id:03d}.png"
            mpath = op.join(mdir, mname)
            cv2.imwrite(mpath, mask)
            rows.append((op.basename(ip), inst_id, cls, conf, x1,y1,x2,y2, op.relpath(mpath, args.out)))
            inst_id += 1

    # write instances.csv
    with open(op.join(args.out, "instances.csv"), "w") as f:
        f.write(",".join(rows[0])+"\n")
        for r in rows[1:]:
            f.write(",".join(map(str, r))+"\n")
    print("[OK] wrote", op.join(args.out, "instances.csv"), "| instances=", len(rows)-1)

if __name__ == "__main__":
    main()
