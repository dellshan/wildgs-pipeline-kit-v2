#!/usr/bin/env bash
set -euo pipefail

# 1) 准备三个“toy”桶（只要能跑通后续即可）
mkdir -p outputs/nuscenes/mini/images \
         outputs/kitti/sample/images \
         outputs/mipnerf360/bicycle/images

# 2) 从仓库里抓 10 张现成图片（不扫 outputs/）
readarray -t CANDS < <(find . -maxdepth 2 -type f \( -iname '*.jpg' -o -iname '*.png' \) \
  | grep -v './outputs/' | head -n 10)
if [ ${#CANDS[@]} -eq 0 ]; then
  echo "没找到本地 jpg/png，请放几张图再跑"; exit 1
fi

i=0
for p in "${CANDS[@]}"; do
  bn=$(printf "%06d.jpg" $i)
  for d in outputs/nuscenes/mini/images outputs/kitti/sample/images outputs/mipnerf360/bicycle/images; do
    cp "$p" "$d/$bn"
  done
  i=$((i+1))
done

# 3) 生成最小 frames_views.csv（位姿用 0 先占位）
for d in outputs/nuscenes/mini outputs/kitti/sample outputs/mipnerf360/bicycle; do
  printf "image,ts,scene,cam,tx,ty,tz,yaw_deg\n" > "$d/frames_views.csv"
  for f in "$d/images"/*.jpg; do
    printf "%s,0,TOY,cam,0,0,0,0\n" "$(basename "$f")" >> "$d/frames_views.csv"
  done
done

# 4) YOLO 仅打框 + 导出 CSV
python tools/yolo_detect_dir.py --images outputs/nuscenes/mini/images --out outputs/nuscenes/mini
python tools/boxes_to_csv.py --images outputs/nuscenes/mini/images --labels outputs/nuscenes/mini/labels \
  --out_csv outputs/nuscenes/mini/yolo_preds.csv

python tools/yolo_detect_dir.py --images outputs/kitti/sample/images --out outputs/kitti/sample
python tools/boxes_to_csv.py --images outputs/kitti/sample/images --labels outputs/kitti/sample/labels \
  --out_csv outputs/kitti/sample/yolo_preds.csv

python tools/yolo_detect_dir.py --images outputs/mipnerf360/bicycle/images --out outputs/mipnerf360/bicycle
python tools/boxes_to_csv.py --images outputs/mipnerf360/bicycle/images --labels outputs/mipnerf360/bicycle/labels \
  --out_csv outputs/mipnerf360/bicycle/yolo_preds.csv

echo "[DONE] Toy smoke test ready under outputs/**"
