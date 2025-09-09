#!/usr/bin/env bash
set -euo pipefail
[ $# -ge 1 ] || { echo "Usage: $0 <WGS_DIR>"; exit 1; }
WGS_DIR="$1"; cd "$WGS_DIR"; [ -e datasets ] || ln -s Datasets datasets || true
mkdir -p Datasets/Wild_SLAM_Mocap/scene1
for d in crowd umbrella stones person_tracking table_tracking1; do
  [ -d "$d" ] && ln -sfn "$(pwd)/$d" "Datasets/Wild_SLAM_Mocap/scene1/$d" && echo "Linked $d"
done
