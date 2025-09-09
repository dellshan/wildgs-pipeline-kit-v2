#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob, math, re
import numpy as np

# ---------------- intrinsics (robust) ----------------
def _pick(d, keys):
    for k in keys:
        if k in d: return d[k]
    return None

def _as_3x3(x):
    A = np.array(x, dtype=float)
    if A.size == 9: A = A.reshape(3,3)
    assert A.shape == (3,3)
    return A

def load_intrinsics_robust(intr_path, sample_rgb_dir=None):
    j = json.load(open(intr_path, "r"))

    def dig(d):
        fx = _pick(d, ["fx","fx_rgb","f_x"]);    fy = _pick(d, ["fy","fy_rgb","f_y"])
        cx = _pick(d, ["cx","c_x"]);            cy = _pick(d, ["cy","c_y"])
        W  = _pick(d, ["width","W","w"]);       H  = _pick(d, ["height","H","h"])
        return fx,fy,cx,cy,W,H

    fx,fy,cx,cy,W,H = dig(j)

    # nested dict
    if fx is None:
        for key in ("camera","cam","intrinsics","calibration"):
            if key in j and isinstance(j[key], dict):
                fx,fy,cx,cy,W,H = dig(j[key])
                if fx is not None: break

    # K matrix
    if fx is None:
        for key in ("K","camera_matrix","intrinsics","intrinsic"):
            if key in j:
                K = _as_3x3(j[key])
                fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
                break

    # fallback by fov
    if (fx is None or fy is None):
        fovx = _pick(j, ["fovx","FOVx","hfov"]); fovy = _pick(j, ["fovy","FOVy","vfov"])
        if (W is None or H is None) and sample_rgb_dir:
            imgs = sorted(glob.glob(os.path.join(sample_rgb_dir, "*.*")))
            if imgs:
                from PIL import Image
                W,H = Image.open(imgs[0]).size
        def _rad(v): return float(v if v<=math.pi*2 else v*math.pi/180.0)
        if fovx is not None and fovy is not None and W and H:
            fx = fx or (W / (2.0 * math.tan(_rad(float(fovx))/2.0)))
            fy = fy or (H / (2.0 * math.tan(_rad(float(fovy))/2.0)))

    if (W is None or H is None) and sample_rgb_dir:
        imgs = sorted(glob.glob(os.path.join(sample_rgb_dir, "*.*")))
        if imgs:
            from PIL import Image
            W,H = Image.open(imgs[0]).size

    if cx is None and W is not None: cx = float(W)/2.0
    if cy is None and H is not None: cy = float(H)/2.0
    if fx is None or fy is None:
        if W is not None and H is not None:
            fx = fx or (W+H)/2.0
            fy = fy or (W+H)/2.0

    if None in (fx,fy,cx,cy,W,H):
        raise RuntimeError(f"Cannot parse intrinsics from {intr_path}")

    print(f"[INFO] intrinsics -> fx={float(fx):.3f}, fy={float(fy):.3f}, cx={float(cx):.3f}, cy={float(cy):.3f}, W={int(W)}, H={int(H)}")
    return float(fx), float(fy), float(cx), float(cy), int(W), int(H)

# ---------------- TUM parsing (robust) ----------------
_num_re = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
def _to_float(tok):
    if tok is None: return None
    s = tok.strip().replace(',', ' ')
    try: return float(s)
    except:
        m=_num_re.search(s)
        return float(m.group(0)) if m else None

def load_tum_list(fn):
    out=[]
    with open(fn,"r") as f:
        for ln in f:
            ln=ln.strip()
            if not ln or ln.startswith("#"): continue
            toks=ln.split()
            out.append(toks)
    return out

def tum_txt_to_map(path, min_valid_rows=20):
    mp={}; valid=0
    with open(path,'r') as f:
        for ln in f:
            ln=ln.strip()
            if not ln or ln.startswith('#'): continue
            nums=[_to_float(t) for t in ln.split()]
            nums=[x for x in nums if x is not None]
            if len(nums)<8: continue
            ts,tx,ty,tz,qx,qy,qz,qw = nums[:8]
            mp[float(ts)]=[float(tx),float(ty),float(tz),float(qx),float(qy),float(qz),float(qw)]
            valid+=1
    return mp if valid>=min_valid_rows else {}

def q_to_R(qx,qy,qz,qw):
    q = np.array([qw,qx,qy,qz],dtype=np.float64); q/= (np.linalg.norm(q)+1e-12)
    w,x,y,z = q
    return np.array([
      [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
      [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
      [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ],dtype=np.float64)

# ---------------- pick best NPZ trajectory ----------------
def pick_best_npz_mat(npz_path, n_rgb):
    npz = np.load(npz_path, allow_pickle=True)
    best = None  # (kind, arr, name)
    # 先挑 (N,4,4)
    for k in getattr(npz,'files',[]):
        A = np.array(npz[k])
        if A.ndim==3 and A.shape[-2:]==(4,4):
            score = (A.shape[0], abs(A.shape[0]-n_rgb))
            if best is None or score > (best[1].shape[0], abs(best[1].shape[0]-n_rgb)):
                best = ("mat44", A, k)
    # 其次 (N,7/8)
    if best is None:
        for k in getattr(npz,'files',[]):
            A = np.array(npz[k])
            if A.ndim==2 and (A.shape[1]==7 or A.shape[1]==8):
                score = (A.shape[0], abs(A.shape[0]-n_rgb))
                if best is None or score > (best[1].shape[0], abs(best[1].shape[0]-n_rgb)):
                    best = ("vec7", A, k)
    return best  # or None

# ---------------- find estimated trajectory in run_dir ----------------
def find_est_traj(run_dir, n_rgb):
    # 1) TUM txt（关键帧/全帧）
    patterns = [
        "**/traj_*full*.txt", "**/*Full*Trajectory*.txt",
        "**/*CameraTrajectory*.txt", "**/*poses.txt",
        "**/traj_*key*frame*.txt", "**/Key*Frame*Trajectory*.txt",
        "**/*Trajectory*.txt"
    ]
    best_txt = (None,None)
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(run_dir, pat), recursive=True)):
            if not os.path.isfile(p): continue
            mp = tum_txt_to_map(p, min_valid_rows=20)
            if mp and (best_txt[1] is None or len(mp)>len(best_txt[1])): best_txt=(p,mp)
    if best_txt[1] is not None:
        return ("tum", best_txt[0], best_txt[1])

    # 2) NPZ/NPY：挑最接近 rgb 数量的 (N,4,4) 或 (N,7/8)
    for pat in ["**/*.npz","**/*.npy"]:
        for p in sorted(glob.glob(os.path.join(run_dir, pat), recursive=True)):
            try:
                if p.endswith(".npz"):
                    sel = pick_best_npz_mat(p, n_rgb)
                    if sel is not None:
                        kind, A, name = sel
                        print(f"[INFO] NPZ picked: {os.path.relpath(p)}[{name}] shape={A.shape}")
                        return (kind, p, (A,name))
                else:
                    A = np.load(p, allow_pickle=True)
                    if A.ndim==3 and A.shape[-2:]==(4,4):
                        print(f"[INFO] NPY picked: {os.path.relpath(p)} shape={A.shape}")
                        return ("mat44", p, (A, None))
                    if A.ndim==2 and (A.shape[1]==7 or A.shape[1]==8):
                        print(f"[INFO] NPY picked: {os.path.relpath(p)} shape={A.shape}")
                        return ("vec7", p, (A, None))
            except Exception:
                pass
    return (None, None, None)

def pair_frames(kind, payload, rgb_pairs, tol_sec=5e-3):
    frames=[]
    n_rgb = len(rgb_pairs)
    if kind=="tum":
        est_map = payload  # ts -> [t q]
        est_ts = np.array(sorted(est_map.keys()), dtype=float)
        for ts_s, rel in rgb_pairs:
            ts = float(ts_s)
            if ts in est_map:
                tx,ty,tz,qx,qy,qz,qw = est_map[ts]
            else:
                if len(est_ts)==0: continue
                i = int(np.argmin(np.abs(est_ts - ts)))
                if abs(est_ts[i]-ts) > tol_sec: continue
                tx,ty,tz,qx,qy,qz,qw = est_map[float(est_ts[i])]
            R = q_to_R(qx,qy,qz,qw); t = np.array([tx,ty,tz])
            T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
            frames.append((rel, T))
    elif kind in ("mat44","vec7"):
        A, _name = payload
        A = np.array(A)
        N = min(n_rgb, A.shape[0])
        for i in range(N):
            rel = rgb_pairs[i][1]
            if kind=="mat44":
                T = A[i].astype(np.float64)
            else:
                tx,ty,tz,qx,qy,qz,qw = A[i][:7]
                R = q_to_R(qx,qy,qz,qw); t = np.array([tx,ty,tz])
                T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
            frames.append((rel, T))
    return frames

# ---------------- main ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_json", default="output/wildgs_slam/transforms_slam.json")
    args = ap.parse_args()

    rgb_txt   = os.path.join(args.dataset_root, "rgb.txt")
    intr_json = os.path.join(args.dataset_root, "intrinsics.json")
    rgb_pairs = load_tum_list(rgb_txt)  # [ts, rel]
    n_rgb = len(rgb_pairs)

    fx,fy,cx,cy,W,H = load_intrinsics_robust(intr_json, sample_rgb_dir=os.path.join(args.dataset_root,"rgb"))

    kind, path, payload = find_est_traj(args.run_dir, n_rgb)
    if kind is None:
        print("[WARN] no estimated traj found; fallback to GT")
        gt_map = tum_txt_to_map(os.path.join(args.dataset_root,"groundtruth.txt"), min_valid_rows=5)
        payload = gt_map
        kind = "tum"

    print(f"[INFO] using trajectory: kind={kind}, path={path}")
    frames = pair_frames(kind, payload, rgb_pairs)

    out_frames=[]
    for rel, T in frames:
        out_frames.append({
            "file_path": os.path.join(args.dataset_root, rel),
            "transform_matrix": T.tolist()
        })
    out = {"fl_x":fx,"fl_y":fy,"cx":cx,"cy":cy,"w":W,"h":H,"frames":out_frames}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json,"w"), indent=2)
    print(f"[OK] wrote {args.out_json} with {len(out_frames)} frames (rgb.txt had {n_rgb})")

if __name__ == "__main__":
    main()
