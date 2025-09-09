#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, argparse, json, re
import numpy as np
import torch
from PIL import Image

# ---------- Path setup ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
CANDS = [
    os.path.abspath(os.path.join(ROOT, "..", "deps", "WildGS-SLAM")),
    os.path.abspath(os.path.join(ROOT, "..", "deps", "WildGS-SLAM", "thirdparty")),
]
for p in CANDS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# ---------- GS imports (fork优先，失败回退上游包) ----------
try:
    from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
    from thirdparty.gaussian_splatting.gaussian_renderer import render
    from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix
    try:
        from thirdparty.gaussian_splatting.arguments import PipelineParams
        HAVE_PIPE = True
    except Exception:
        HAVE_PIPE = False
    HAS_THIRDPARTY = True
except ModuleNotFoundError:
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.utils.graphics_utils import getProjectionMatrix
    try:
        from gaussian_splatting.arguments import PipelineParams
        HAVE_PIPE = True
    except Exception:
        HAVE_PIPE = False
    HAS_THIRDPARTY = False


# ---------- Minimal camera ----------
def _look_at(eye, target, up=(0, 1, 0)):
    eye = np.array(eye, float)
    target = np.array(target, float)
    up = np.array(up, float)

    z = target - eye
    z /= (np.linalg.norm(z) + 1e-8)
    x = np.cross(z, up); x /= (np.linalg.norm(x) + 1e-8)
    y = np.cross(x, z)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack([x, y, z], axis=1)
    c2w[:3, 3] = eye
    return c2w

class SimpleCamera:
    """
    满足 gaussian_renderer.render 所需字段：
    world_view_transform, projection_matrix, full_proj_transform,
    image_width, image_height, camera_center, FoVx, FoVy, tanfovx, tanfovy,
    以及（部分fork会读取的）cam_rot_delta, cam_trans_delta。
    另外补上一些兼容字段（image, image_name, uid, data_device 等）。
    """
    def __init__(self, c2w, fx, fy, w, h, znear=0.01, zfar=100.0, device="cuda"):
        self.image_width  = int(w)
        self.image_height = int(h)
        self.data_device  = device

        # 相机中心（c2w 的 t）
        self.camera_center = torch.from_numpy(c2w[:3, 3].astype(np.float32)).to(device)

        # world->view （列主）
        w2c = np.linalg.inv(c2w).astype(np.float32)
        self.world_view_transform = torch.from_numpy(w2c.T.copy()).to(device)

        # FOV from intrinsics
        self.FoVx = 2.0 * math.atan(0.5 * w / float(fx))
        self.FoVy = 2.0 * math.atan(0.5 * h / float(fy))
        self.tanfovx = math.tan(self.FoVx / 2.0)
        self.tanfovy = math.tan(self.FoVy / 2.0)

        # 投影矩阵（到目标 device）
        P = getProjectionMatrix(znear, zfar, self.FoVx, self.FoVy)
        if isinstance(P, torch.Tensor):
            P = P.T.to(device)
        else:
            P = torch.from_numpy(np.asarray(P, dtype=np.float32).T.copy()).to(device)
        self.projection_matrix  = P
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

        # 某些分支里会访问的增量参数名（必须存在）
        self.cam_rot_delta   = torch.zeros(3, device=device, dtype=torch.float32)
        self.cam_trans_delta = torch.zeros(3, device=device, dtype=torch.float32)

        # 兼容字段（通常不会用到，但保持齐全）
        self.image = torch.empty(1, dtype=torch.float32, device=device)
        self.image_name = "orbit"
        self.uid = -1

        # 也留一份 R、T（world->cam），以防某些调试路径读取
        self.R = torch.from_numpy(w2c[:3, :3]).to(device)
        self.T = torch.from_numpy(w2c[:3, 3]).to(device)


# ---------- Helpers ----------
def detect_sh_degree_from_ply(ply_path: str) -> int:
    """
    数 f_rest_*（或 rest_*/sh_* 非 f_dc_）的数量，反推 SH degree:
      count = 3 * ((deg+1)^2 - 1)  => deg = sqrt(count/3 + 1) - 1
    """
    hdr_names = []
    with open(ply_path, "r", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln == "end_header":
                break
            if ln.startswith("property"):
                toks = ln.split()
                if len(toks) >= 3:
                    hdr_names.append(toks[-1])

    rest_count = 0
    for n in hdr_names:
        if n.startswith("f_rest_") or n.startswith("rest_"):
            rest_count += 1
        elif n.startswith("sh_") and not n.startswith("f_dc_"):
            rest_count += 1

    if rest_count == 0:
        return 0
    deg = int(round(math.sqrt(rest_count / 3.0 + 1.0) - 1.0))
    return max(0, min(deg, 5))

def load_intrinsics(tf_json):
    with open(tf_json, "r") as f:
        j = json.load(f)
    W, H = int(j["w"]), int(j["h"])
    fx, fy = float(j["fl_x"]), float(j["fl_y"])
    return W, H, fx, fy

def sphere_eye(center, radius, yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg); pitch = math.radians(pitch_deg)
    x = center[0] + radius * math.cos(pitch) * math.cos(yaw)
    y = center[1] + radius * math.sin(pitch)
    z = center[2] + radius * math.cos(pitch) * math.sin(yaw)
    return np.array([x, y, z], dtype=np.float32)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True,
                    help="高斯 PLY（如 deps/WildGS-SLAM/output/.../final_gs.ply）")
    ap.add_argument("--transforms", required=True,
                    help="transforms.json（含 fl_x/fl_y/w/h）")
    ap.add_argument("--out_dir", default="output/wildgs_slam/orbit")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--elev", type=float, default=5.0)
    ap.add_argument("--radius_scale", type=float, default=1.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bg", choices=["black", "white"], default="black")
    ap.add_argument("--znear", type=float, default=0.01)
    ap.add_argument("--zfar",  type=float, default=100.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 自适应 SH 阶数
    deg = detect_sh_degree_from_ply(args.ply)
    print(f"[INFO] Detected SH degree from PLY: {deg}")

    # 2) 载入 3D 高斯
    gs = GaussianModel(sh_degree=deg)
    gs.load_ply(args.ply)
    if hasattr(gs, "cuda"):
        gs.cuda()

    # 3) 相机轨道中心与半径
    xyz = gs.get_xyz.detach().cpu().numpy()
    center = xyz.mean(0)
    radius = np.linalg.norm(xyz - center, axis=1).mean() * float(args.radius_scale)

    # 4) 内参
    W, H, fx, fy = load_intrinsics(args.transforms)

    # 5) 渲染管线
    if HAVE_PIPE:
        pipe = PipelineParams()
    else:
        class _Pipe: pass
        pipe = _Pipe()
        pipe.convert_SHs_python   = False
        pipe.compute_cov3D        = True
        pipe.compute_cov3D_python = False
        pipe.debug                = False

    background = torch.tensor([0, 0, 0] if args.bg == "black" else [1, 1, 1],
                              dtype=torch.float32, device=device)

    # 6) 逐帧渲染
    for i in range(args.frames):
        yaw = 360.0 * i / args.frames
        eye = sphere_eye(center, radius, yaw, args.elev)
        c2w = _look_at(eye, center, up=(0, 1, 0))

        cam = SimpleCamera(
            c2w=c2w, fx=fx, fy=fy, w=W, h=H,
            znear=args.znear, zfar=args.zfar, device=device
        )

        out = render(cam, gs, pipe, background)
        rgb = (out["render"].clamp(0, 1) * 255).permute(1, 2, 0).byte().cpu().numpy()
        Image.fromarray(rgb).save(os.path.join(args.out_dir, f"orbit_{i:04d}.png"))

    # 7) 合成视频
    os.system(
        f"ffmpeg -y -framerate {args.fps} -pattern_type glob "
        f"-i '{args.out_dir}/orbit_*.png' -c:v libx264 -pix_fmt yuv420p "
        f"{os.path.join(args.out_dir, 'orbit.mp4')}"
    )
    print(f"[OK] wrote {os.path.join(args.out_dir, 'orbit.mp4')}")

if __name__ == "__main__":
    main()
