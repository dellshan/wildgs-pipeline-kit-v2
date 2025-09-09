#!/usr/bin/env bash
# Run SVD-XT on 2 GPUs by splitting frames and stitching.
# Usage: scripts/svd_two_gpu.sh <image> <out.mp4> <frames> <fps> <width> <height>

set -euo pipefail
IMG="${1:?image}"
OUT="${2:-output/svd_xt_2gpu.mp4}"
FRAMES="${3:-20}"
FPS="${4:-7}"
W="${5:-896}"
H="${6:-504}"

# tuneables (lower mem if needed)
DCS="${DCS:-16}"             # decode_chunk_size
MOTION="${MOTION:-160}"      # 127~200; higher = more motion
NOISE="${NOISE:-0.10}"       # 0.08~0.18 sensible
SEED="${SEED:-0}"
EXTRA_FLAGS="${EXTRA_FLAGS:---no-xformers --sequential-offload}"  # safest VRAM

# split
A=$((FRAMES/2))
B=$((FRAMES - A))

echo "[2GPU] chunk A: frames=${A} on GPU0"
CUDA_VISIBLE_DEVICES=0 python pipeline/svd_generate.py \
  --image "${IMG}" --out output/_partA.mp4 \
  --frames "${A}" --fps "${FPS}" --width "${W}" --height "${H}" \
  --decode-chunk-size "${DCS}" --motion-bucket-id "${MOTION}" \
  --noise-aug-strength "${NOISE}" --seed "${SEED}" ${EXTRA_FLAGS} &

echo "[2GPU] chunk B: frames=${B} on GPU1"
CUDA_VISIBLE_DEVICES=1 python pipeline/svd_generate.py \
  --image "${IMG}" --out output/_partB.mp4 \
  --frames "${B}" --fps "${FPS}" --width "${W}" --height "${H}" \
  --decode-chunk-size "${DCS}" --motion-bucket-id "${MOTION}" \
  --noise-aug-strength "${NOISE}" --seed "$((SEED+1))" ${EXTRA_FLAGS} &

wait

echo "[2GPU] stitching…"
ffmpeg -y -loglevel error \
  -i output/_partA.mp4 -i output/_partB.mp4 \
  -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0" \
  -r "${FPS}" -pix_fmt yuv420p "${OUT}"

rm -f output/_partA.mp4 output/_partB.mp4
echo "Done -> ${OUT}"

