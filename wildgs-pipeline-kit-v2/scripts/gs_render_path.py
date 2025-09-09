#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, math, json, argparse
import numpy as np
import torch
from PIL import Image

# --- Robust path injection (repo root, then thirdparty) ---
ROOT = os.path.dirname(os.path.abspath(__file__))
CANDS = [
    os.path.abspath(os.path.join(ROOT, "..", "deps", "WildGS-SLAM")),
    os.path.abspath(os.path.join(ROOT, "..", "deps", "WildGS-SLAM", "thirdparty")),
]
for p in CANDS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Prefer fork's namespace, fall back to upstream
try:
    from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
    from thirdparty.gaussian_splatting.gaussian_renderer import render
    from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix
    try:
        from thirdparty.gaussian_splatting.arguments import PipelineParams as _PP  # optional
    except Exception:
        _PP = None
except ModuleNotFoundError:
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.utils.graphics_utils import getProjectionMatrix
    try:
        from gaussian_splatting.arguments import PipelineParams as _PP  # optional
    except Exception:
        _PP = None

from types import SimpleNamespace


# ---------- Minimal camera compatible with renderer ----------
class SimpleCamera:
    def __init__(self, c2w, fx, fy, w, h, znear=0.01, zfar=100.0, device="cuda"):
        self.image_width  = int(w)
        self.image_height = int(h)

        c2w = np.asarray(c2w, dtype=np.float32)
        self.camera_center = torch.tensor(c2w[:3, 3], dtype=torch.float32, device=device)

        # world->view (column-major expected by CUDA kernels)
        w2c = np.linalg.inv(c2w).T
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32, device=device)

        # FoVs from intrinsics
        self.FoVx = 2.0 * math.atan(self.image_width  / (2.0 * float(fx)))
        self.FoVy = 2.0 * math.atan(self.image_height / (2.0 * float(fy)))
        self.tanfovx = math.tan(self.FoVx / 2.0)
        self.tanfovy = math.tan(self.FoVy / 2.0)

        # projection (column-major) and composed transform
        P = getProjectionMatrix(znear, zfar, self.FoVx, self.FoVy)
        if isinstance(P, torch.Tensor):
            P = P.T.to(dtype=torch.float32, device=device)
        else:
            P = torch.tensor(np.asarray(P).T, dtype=torch.float32, device=device)
        self.projection_matrix   = P
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

        # fields renderer accesses
        self.cam_rot_delta   = torch.zeros(3, dtype=torch.float32, device=device)
        self.cam_trans_delta = torch.zeros(3, dtype=torch.float32, device=device)


def detect_sh_degree_from_ply(ply_path: str) -> int:
    """Infer SH degree from number of f_rest_* properties in PLY header."""
    try:
        n_extra = 0
        with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith("end_header"):
                    break
                ls = ln.strip()
                if ls.startswith("property") and "f_rest_" in ls:
                    n_extra += 1
        if n_extra == 0:
            return 0
        return max(0, int(round((n_extra / 3.0 + 1.0) ** 0.5 - 1.0)))
    except Exception:
        return 0


def make_pipeline(sh_degree: int, gs_obj) -> object:
    """Create a PipelineParams-like object (attributes, not dict)."""
    active = int(getattr(gs_obj, "active_sh_degree", sh_degree))
    params = dict(
        convert_SHs_to_RGB=True,      # keep RGB conversion in renderer
        compute_cov3D_python=False,   # use CUDA kernels
        convert_SHs_python=False,     # <- add this: renderer checks for it
        sh_degree=active,
        debug=False,
    )
    if _PP is not None:
        try:
            return _PP(**params)
        except Exception:
            pass
    from types import SimpleNamespace
    return SimpleNamespace(**params)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--transforms", required=True, help="transforms.json with fl_x, fl_y, w, h, frames[].transform_matrix")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bg", choices=["black","white"], default="black")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Gaussians
    sh_deg = detect_sh_degree_from_ply(args.ply)
    print(f"[INFO] Detected SH degree from PLY: {sh_deg}")
    gs = GaussianModel(sh_degree=int(sh_deg))
    gs.load_ply(args.ply)  # NOTE: this fork's model isn't nn.Module, so no .cuda()

    # Transforms/intrinsics
    tj = json.load(open(args.transforms, "r"))
    fx, fy = float(tj["fl_x"]), float(tj["fl_y"])
    w, h   = int(tj["w"]), int(tj["h"])
    frames = tj["frames"]
    N = len(frames)
    i0 = max(0, args.start)
    i1 = N if args.end < 0 else min(N, args.end)
    print(f"[INFO] Rendering frames [{i0}, {i1}) / {N}")

    # Pipeline + background
    pipe = make_pipeline(sh_deg, gs)
    background = torch.ones(3, device=device) if args.bg == "white" else torch.zeros(3, device=device)

    # Render along path
    for i in range(i0, i1):
        rec = frames[i]
        c2w = np.array(rec["transform_matrix"], dtype=np.float32)
        cam = SimpleCamera(c2w, fx=fx, fy=fy, w=w, h=h, device=device)
        out = render(cam, gs, pipe, background)
        rgb = (out["render"] * 255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
        Image.fromarray(rgb).save(os.path.join(args.out_dir, f"path_{i:05d}.png"))

    os.system(
        f"ffmpeg -y -framerate {args.fps} -pattern_type glob -i '{args.out_dir}/path_*.png' "
        f"-c:v libx264 -pix_fmt yuv420p {args.out_dir}/path.mp4"
    )
    print(f"[OK] wrote {args.out_dir}/path.mp4")


if __name__ == "__main__":
    main()
