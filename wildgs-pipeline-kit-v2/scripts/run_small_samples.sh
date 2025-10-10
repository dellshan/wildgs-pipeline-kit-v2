#!/usr/bin/env bash
set -euo pipefail

# 1) nuScenes mini
python tools/prepare_nuscenes_mini.py \
  --dataroot data/nuscenes/raw \
  --version v1.0-mini \
  --out outputs/nuscenes/mini \
  --per_scene 40

python tools/yolo_detect_dir.py \
  --images outputs/nuscenes/mini/images \
  --out outputs/nuscenes/mini

python tools/boxes_to_csv.py \
  --images outputs/nuscenes/mini/images \
  --labels outputs/nuscenes/mini/labels \
  --out_csv outputs/nuscenes/mini/yolo_preds.csv

# 2) KITTI sample
python tools/prepare_kitti_sample.py \
  --root data/kitti/raw/training/image_2 \
  --out outputs/kitti/sample \
  --num 200

python tools/yolo_detect_dir.py \
  --images outputs/kitti/sample/images \
  --out outputs/kitti/sample

python tools/boxes_to_csv.py \
  --images outputs/kitti/sample/images \
  --labels outputs/kitti/sample/labels \
  --out_csv outputs/kitti/sample/yolo_preds.csv

# 3) Mip-NeRF 360（默认场景：bicycle；可传参覆盖）
SCENE=${1:-bicycle}
python tools/prepare_mipnerf360_sample.py ${SCENE} \
  --root data/mipnerf360/raw \
  --out_root outputs/mipnerf360 \
  --num 80

python tools/yolo_detect_dir.py \
  --images outputs/mipnerf360/${SCENE}/images \
  --out outputs/mipnerf360/${SCENE}

python tools/boxes_to_csv.py \
  --images outputs/mipnerf360/${SCENE}/images \
  --labels outputs/mipnerf360/${SCENE}/labels \
  --out_csv outputs/mipnerf360/${SCENE}/yolo_preds.csv

echo "[DONE] All small-sample pipelines finished."
