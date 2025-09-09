#!/usr/bin/env bash
set -euo pipefail
[ $# -ge 2 ] || { echo "Usage: $0 <WGS_DIR> <base_name>"; exit 1; }
WGS_DIR="$1"; BASE="$2"; CFG_DIR="$WGS_DIR/configs/Dynamic/Wild_SLAM_Mocap"
cp "$CFG_DIR/${BASE}.yaml" "$CFG_DIR/${BASE}_tiny.yaml"
sed -i 's/^\(\s*H_out:\s*\).*/ 240/' "$CFG_DIR/${BASE}_tiny.yaml" || true
sed -i 's/^\(\s*W_out:\s*\).*/ 400/' "$CFG_DIR/${BASE}_tiny.yaml" || true
echo "Created ${BASE}_tiny.yaml"
