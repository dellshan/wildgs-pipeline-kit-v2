#!/usr/bin/env bash
set -euo pipefail
[ $# -ge 1 ] || { echo "Usage: $0 <WGS_DIR>"; exit 1; }
WGS_DIR="$1"
[ -d "$WGS_DIR" ] || { echo "Not a dir: $WGS_DIR"; exit 2; }
for f in run.py requirements.txt thirdparty; do [ -e "$WGS_DIR/$f" ] || { echo "Missing $f"; exit 3; }; done
if [ ! -f "$WGS_DIR/pretrained/droid.pth" ]; then echo "WARN: missing pretrained/droid.pth"; fi
python - <<'PY'
mods=["torch","lietorch","simple_knn","diff_gaussian_rasterization"]
for m in mods:
    try: __import__(m); print("[OK] import",m)
    except Exception as e: print("[FAIL] import",m,"->",e)
import torch; print("Torch:",torch.__version__,"CUDA avail:",torch.cuda.is_available(),"CUDA:",torch.version.cuda)
PY
