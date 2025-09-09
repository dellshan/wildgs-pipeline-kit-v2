#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: $0 --traj <est_poses_full.txt> --rgb <rgb_dir> --out <out_dir> [options]"
  echo "Options: --max-kf N --svd-gpu ID --svd-frames N --svd-fps N --svd-w W --svd-h H"
}
MAX_KF=30; SVD_GPU=0; SVD_FRAMES=14; SVD_FPS=7; SVD_W=768; SVD_H=432
TRAJ=""; RGB=""; OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --traj) TRAJ="$2"; shift 2;;
    --rgb) RGB="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --max-kf) MAX_KF="$2"; shift 2;;
    --svd-gpu) SVD_GPU="$2"; shift 2;;
    --svd-frames) SVD_FRAMES="$2"; shift 2;;
    --svd-fps) SVD_FPS="$2"; shift 2;;
    --svd-w) SVD_W="$2"; shift 2;;
    --svd-h) SVD_H="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done
[ -n "$TRAJ" ] && [ -n "$RGB" ] && [ -n "$OUT" ] || { usage; exit 1; }

mkdir -p "$OUT" keyframes

echo "[1/3] Keyframes..."
python pipeline/select_keyframes.py "$TRAJ" "$RGB" keyframes --max-kf "$MAX_KF"

IMG=$(ls keyframes/kf_000.* 2>/dev/null | head -n1); [ -n "$IMG" ] || IMG=$(ls keyframes/*.* | head -n1)

echo "[2/3] CLIP..."
python pipeline/clip_describe.py --frames_dir keyframes --out "$OUT/clip_results.json" --csv "$OUT/clip_results.csv"

echo "[3/3] SVD... (GPU $SVD_GPU)"
export CUDA_VISIBLE_DEVICES="$SVD_GPU"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python pipeline/svd_generate.py --image "$IMG" --out "$OUT/svd_out.mp4" \
  --frames "$SVD_FRAMES" --fps "$SVD_FPS" --width "$SVD_W" --height "$SVD_H"

echo "Done -> $OUT"
