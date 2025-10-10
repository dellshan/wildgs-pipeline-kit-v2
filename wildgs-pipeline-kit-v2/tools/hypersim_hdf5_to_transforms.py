#!/usr/bin/env python3
import os, os.path as op, json, glob, re, argparse
import numpy as np
from PIL import Image
import h5py

POSE_KEYS = [
    "camera_keyframe_poses","poses","cam2world","c2w",
    "camera_to_world","world_from_camera","extrinsics"
]
INTRIN_KEYS = [
    "intrinsics/focal_length_px","focal_length_px","fx_fy_px",
    "intrinsics/focal_length","focal_length"
]
FOV_KEYS = ["fov_x_deg","fov_deg","camera/fov_x_deg"]

def find_h5(scene_root):
    cands=[]
    for pat in ("**/*.hdf5","**/*.h5"):
        cands += glob.glob(op.join(scene_root, pat), recursive=True)
    return cands

def pick_pose_dataset(h5):
    for k in POSE_KEYS:
        if k in h5: return k
        for g in h5.keys():
            if isinstance(h5[g], h5py.Group) and k in h5[g]:
                return f"{g}/{k}"
    return None

def read_intrinsics(h5, W, H):
    fx = fy = None
    for k in INTRIN_KEYS:
        if k in h5:
            v = np.array(h5[k]); 
            if v.shape==(2,): fx,fy=float(v[0]),float(v[1])
            elif np.ndim(v)==0: fx=fy=float(v)
            break
        for g in h5.keys():
            if isinstance(h5[g], h5py.Group) and k in h5[g]:
                v = np.array(h5[g][k])
                if v.shape==(2,): fx,fy=float(v[0]),float(v[1])
                elif np.ndim(v)==0: fx=fy=float(v)
                break
    if fx is None or fy is None:
        fov_deg=None
        for k in FOV_KEYS:
            if k in h5:
                fov_deg=float(np.array(h5[k])); break
            for g in h5.keys():
                if isinstance(h5[g], h5py.Group) and k in h5[g]:
                    fov_deg=float(np.array(h5[g][k])); break
        if fov_deg and fov_deg>0:
            fov=np.deg2rad(fov_deg)
            fx=fy=0.5*W/np.tan(0.5*fov)
    if fx is None or fy is None:
        fx=fy=0.9*max(W,H)  # 兜底，后续 Nerfstudio 会 refine
    return fx, fy

def parse_idx(name):
    m=re.search(r'(\d+)', op.splitext(name)[0])
    return int(m.group(1)) if m else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scene-root", required=True)
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--pattern", default="*.color.jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pose-space", choices=["c2w","w2c"], default="c2w",
                    help="HDF5里的矩阵是cam2world(c2w)还是world2cam(w2c)")
    args=ap.parse_args()

    rgb_root=op.join(args.scene_root, args.rgb_dir)
    images=sorted(glob.glob(op.join(rgb_root, args.pattern)))
    if not images:
        raise SystemExit(f"[ERR] No images under {rgb_root} with {args.pattern}")

    W,H=Image.open(images[0]).size
    cx,cy=W/2.0,H/2.0

    h5_paths=find_h5(args.scene_root)
    if not h5_paths:
        raise SystemExit(f"[ERR] No HDF5 under {args.scene_root}")

    pose_mats=None; used_h5=used_ds=None
    for hp in h5_paths:
        try:
            with h5py.File(hp,"r") as h5:
                ds=pick_pose_dataset(h5)
                if not ds: continue
                M=np.array(h5[ds])
                if M.ndim==3 and ((M.shape[1],M.shape[2])==(4,4) or (M.shape[1],M.shape[2])==(3,4)):
                    if M.shape[1]==3:
                        T=np.tile(np.array([0,0,0,1.0],dtype=M.dtype),(M.shape[0],1,1)).reshape(M.shape[0],1,4)
                        M=np.concatenate([M,T],axis=1)
                    pose_mats=M; used_h5=hp; used_ds=ds; break
        except Exception:
            continue
    if pose_mats is None:
        raise SystemExit("[ERR] No usable pose dataset in any HDF5.")

    if len(pose_mats)==len(images):
        pairs=list(zip(images, pose_mats))
    else:
        pose_dict={i:pose_mats[i] for i in range(len(pose_mats))}
        tmp=[]
        for ip in images:
            idx=parse_idx(op.basename(ip))
            if idx is not None and idx in pose_dict:
                tmp.append((ip, pose_dict[idx]))
        if not tmp:
            raise SystemExit("[ERR] Cannot align images with poses by index.")
        pairs=tmp

    frames=[]
    for ip,M in pairs:
        M=M.astype(np.float64)
        if args.pose_space=="w2c":
            M=np.linalg.inv(M)
        frames.append({
            "file_path": op.relpath(ip, op.dirname(args.out)),
            "transform_matrix": M.tolist()
        })

    fx,fy=read_intrinsics(h5py.File(used_h5,"r"), W, H)
    J={"w":W,"h":H,"cx":cx,"cy":cy,"fl_x":fx,"fl_y":fy,"frames":frames}
    os.makedirs(op.dirname(args.out), exist_ok=True)
    json.dump(J, open(args.out,"w"), indent=2)
    print(f"[OK] wrote {args.out}")
    print(f"     poses from: {used_h5} :: {used_ds}")
    print(f"     frames={len(frames)} | intrinsics (fx,fy)=({fx:.1f},{fy:.1f})")

if __name__=="__main__":
    main()
