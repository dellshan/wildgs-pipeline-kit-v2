#!/usr/bin/env python3
import os, os.path as op, argparse
from glob import glob
from ultralytics import YOLO

DEF_WEIGHTS = ["yolov12m.pt", "yolov12n.pt", "yolo12m.pt", "yolo11n.pt", "yolov8n.pt"]


def pick_weights():
    for w in DEF_WEIGHTS:
        if op.isfile(w):
            return w
    return DEF_WEIGHTS[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    os.makedirs(op.join(args.out, "labels"), exist_ok=True)
    w = pick_weights()
    m = YOLO(w)
    results = m.predict(source=args.images, imgsz=640, conf=args.conf, device=args.device, stream=True, verbose=False)

    for r in results:
        name = op.basename(r.path)
        stem = op.splitext(name)[0]
        # YOLO TXT: class cx cy w h (normalized)
        lines=[]
        if r.boxes is not None and len(r.boxes) > 0:
            xywhn = r.boxes.xywhn.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for (cx,cy,w,h),c,p in zip(xywhn, cls, conf):
                lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {p:.4f}")
        with open(op.join(args.out, "labels", stem+".txt"), "w") as f:
            f.write("\n".join(lines))
    print("[OK] YOLO predictions ->", args.out)

if __name__ == "__main__":
    main()