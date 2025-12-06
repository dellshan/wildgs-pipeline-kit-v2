Advanced WildGS‑SLAM → CLIP → SVD Pipeline (v2)
Advanced WildGS‑SLAM → CLIP → SVD Pipeline (v2)
This repository contains an end‑to‑end PyTorch implementation that:

Runs WildGS‑SLAM (monocular 3D Gaussian SLAM with RAFT optical flow),
Performs multi‑view scene understanding with CLIP,
Generates short videos with Stable‑Video‑Diffusion.
Structure
wildgs_pipeline/
├── wildgs/
│   ├── __init__.py
│   └── pipeline.py   # main implementation
├── requirements.txt  # pip install -r requirements.txt
└── README.md         # this file
Quick start
# clone or unzip, then:
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# run the demo (will create ./out/ with results)
python - <<'PY'
from wildgs import WildGSCLIPSVDPipeline, Path
pipe = WildGSCLIPSVDPipeline()
out_dir = Path('./out')
out_dir.mkdir(exist_ok=True)
pipe.process_video('your_video.mp4', out_dir)
PY
If you have no video at hand, the script will automatically create a 3‑second dummy clip and process it.
