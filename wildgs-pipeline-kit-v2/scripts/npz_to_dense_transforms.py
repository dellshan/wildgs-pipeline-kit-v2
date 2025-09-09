#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turn WildGS-SLAM video.npz keyframe poses into a dense per-frame transforms.json
- Interpolates SE(3): SLERP for rotation, linear for translation
- Uses rgb.txt timestamps as targets
"""

import os, sys, json, math, argparse
import numpy as np

# --- import shim: prefer repo root & thirdparty ---
ROOT = os.path.dirname(os.path.abspath(__file__))
CANDS = [
    os.path.abspath(os.path.join(ROOT, "..", "deps", "WildGS-SLAM")),               # repo root
    os.path.abspath(os.path.join(ROOT, "..", "deps", "WildGS-SLAM", "thirdparty")), # thirdparty path
]
for p in CANDS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# ---------- math utils ----------
def mat3_to_quat_wxyz(R):
    """Rotation matrix (3x3) → unit quaternion [w,x,y,z]."""
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        i = int(np.argmax([m[0,0], m[1,1], m[2,2]]))
        if i == 0:
            s = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    q = np.array([w,x,y,z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q

def quat_wxyz_to_mat3(q):
    """Quaternion [w,x,y,z] → 3x3 rotation matrix."""
    w,x,y,z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1-2*(xx+zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1-2*(xx+yy)]
    ], dtype=np.float64)

def slerp(q0, q1, a):
    """Spherical linear interpolation between two unit quats [w,x,y,z]."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESH = 0.9995
    if dot > DOT_THRESH:
        # nearly linear
        q = q0 + a*(q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    theta_0 = math.acos(dot)
    sin_0 = math.sin(theta_0)
    theta = theta_0 * a
    sin_t = math.sin(theta)
    s0 = math.sin(theta_0 - theta) / (sin_0 + 1e-12)
    s1 = sin_t / (sin_0 + 1e-12)
    return (s0*q0 + s1*q1)

def parse_rgb_txt(path):
    rows=[]
    with open(path, 'r') as f:
        for ln in f:
            ln=ln.strip()
            if not ln or ln.startswith('#'): continue
            ts, rel = ln.split()
            rows.append((float(ts), rel))
    return rows

def copy_intrinsics(dst, intr_src_json):
    j = json.load(open(intr_src_json, "r"))
    for k in ['fl_x','fl_y','cx','cy','w','h']:
        if k not in j:
            raise KeyError(f"Missing {k} in intrinsics source {intr_src_json}")
        dst[k] = j[k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="WildGS-SLAM output video.npz (poses, timestamps)")
    ap.add_argument("--rgb_txt", required=True, help="TUM rgb.txt with all timestamps/relative paths")
    ap.add_argument("--dataset_root", required=True, help="Folder containing rgb/ (to resolve file paths)")
    ap.add_argument("--copy_intrinsics_from", required=True, help="transforms.json with correct intrinsics")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    z = np.load(args.npz)
    poses_k = z["poses"]              # (K,4,4) camera-to-world
    t_k     = z["timestamps"].astype(np.float64)  # (K,)
    assert poses_k.shape[0] == t_k.shape[0] and poses_k.shape[1:] == (4,4)

    # prepare keyframe SE(3)
    q_k = np.zeros((len(t_k),4)); tvec_k = np.zeros((len(t_k),3))
    for i in range(len(t_k)):
        M = poses_k[i]
        q_k[i]    = mat3_to_quat_wxyz(M[:3,:3])
        tvec_k[i] = M[:3,3]

    # target timestamps from rgb.txt
    rows = parse_rgb_txt(args.rgb_txt)  # list of (ts, relpath)
    t_all = np.array([r[0] for r in rows], dtype=np.float64)

    # interpolate
    frames = []
    for ts, rel in rows:
        j = int(np.searchsorted(t_k, ts, side='right')) - 1
        if j < 0:
            j = 0; a = 0.0
        elif j >= len(t_k)-1:
            j = len(t_k)-2; a = 1.0
        else:
            t0, t1 = t_k[j], t_k[j+1]
            a = 0.0 if t1<=t0 else float((ts - t0) / (t1 - t0))

        q = slerp(q_k[j], q_k[j+1], a)
        R = quat_wxyz_to_mat3(q)
        tvec = (1.0 - a) * tvec_k[j] + a * tvec_k[j+1]

        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = R
        T[:3, 3] = tvec
        frames.append({
            "file_path": os.path.join(args.dataset_root, rel),
            "time": ts,
            "transform_matrix": T.tolist()
        })

    out = {"frames": frames}
    copy_intrinsics(out, args.copy_intrinsics_from)  # put fl_x, fl_y, cx, cy, w, h

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote {args.out_json} with {len(frames)} frames")

if __name__ == "__main__":
    main()
