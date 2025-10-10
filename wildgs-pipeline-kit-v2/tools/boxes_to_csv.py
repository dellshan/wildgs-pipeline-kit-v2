#!/usr/bin/env python3
import os, os.path as op, argparse, glob, csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = [("image","cls","conf","cx","cy","w","h")]
    for p in sorted(glob.glob(op.join(args.labels, "*.txt"))):
        stem = op.splitext(op.basename(p))[0]
        for ln in open(p):
            parts = ln.strip().split()
            if len(parts) < 6:  # 兼容无 conf 的格式
                continue
            c,cx,cy,w,h,conf = parts[0],parts[1],parts[2],parts[3],parts[4],parts[5]
            rows.append((stem, int(c), float(conf), float(cx), float(cy), float(w), float(h)))

    os.makedirs(op.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("[OK] wrote", args.out_csv, "| dets=", len(rows)-1)

if __name__ == "__main__":
    main()