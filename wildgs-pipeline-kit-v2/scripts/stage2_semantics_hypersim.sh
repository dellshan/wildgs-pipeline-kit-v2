#!/usr/bin/env bash
set -euo pipefail
BASE="outputs/hypersim"
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=(ai_001_001)

for S in "${SCENES[@]}"; do
  IMG_DIR="${BASE}/${S}/images"
  LAB_DIR="${BASE}/${S}/labels"
  MSK_DIR="${BASE}/${S}/masks"
  CSV_BOX="${BASE}/${S}/yolo_preds.csv"

  mkdir -p "${MSK_DIR}"

  echo ">>> [${S}] box -> rectangle masks (fallback)"
  python tools/boxes_to_rect_masks.py \
    --images "${IMG_DIR}" \
    --labels "${LAB_DIR}" \
    --out    "${MSK_DIR}" \
    --min_area 20

  echo ">>> [${S}] CLIP naming"
  python tools/clip_name_from_masks.py \
    --images "${IMG_DIR}" \
    --masks  "${MSK_DIR}" \
    --yolo_csv "${CSV_BOX}" \
    --out_csv "${BASE}/${S}/objects_views.csv" \
    --lang en
done
echo "[DONE] stage2 for: ${SCENES[*]}"
