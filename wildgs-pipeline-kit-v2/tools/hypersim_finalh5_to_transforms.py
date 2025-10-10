#!/usr/bin/env python3
import os, os.path as op, json, glob, re, argparse
import numpy as np
from PIL import Image
import h5py
import shutil

POSE_HINTS = [
    "camera/world_from_camera","camera/camera_to_world",
    "world_from_camera","camera_to_world",
    "extrinsics","pose","cam2world","c2w","w2c","world_from_cam"
]
K_HINTS = ["camera/K","K","intrinsics/K","camera_matrix","camera/intrinsics/K"]
FXFY_HINTS = ["focal_length_px","camera/focal_length_px","fx_fy_px","focal_length"]
PP_HINTS = ["principal_point_px","camera/principal_point_px","pp","cx_cy_px"]

def parse_idx(p):
    m=re.search(r'(\d+)', op.splitext(op.basename(p))[0])
    return int(m.group(1)) if m else None

def find_first(h5, want_shape=None, name_hints=None):
    # 返回 (name, np.array) 最匹配的一个
    cand=[]
    def cb(name, obj):
        if not isinstance(obj, (h5py.Dataset,)): return
        arr = np.array(obj)
        sh = arr.shape
        if want_shape and sh!=want_shape: 
            # 允许 (3,4) 视为OK
            if not (want_shape==(4,4) and sh==(3,4)):
                return
        score = 0
        if name_hints:
            nm = name.lower()
            for i,kw in enumerate(name_hints):
                if kw in nm: score += (100 - i)  # 带顺序加权
        cand.append((score, name, arr))
    h5.visititems(cb)
    if not cand: return None
    cand.sort(key=lambda x: (-x[0], x[1]))
    return cand[0][1], cand[0][2]

def get_pose(h5, prefer="world_from_camera"):
    # 1) 直接找 4x4 / 3x4
    hit = find_first(h5, want_shape=(4,4), name_hints=POSE_HINTS)
    if not hit:
        hit = find_first(h5, want_shape=(3,4), name_hints=POSE_HINTS)
    if not hit: 
        return None, None
    name, M = hit
    if M.shape==(3,4):
        M = np.vstack([M, np.array([0,0,0,1.0])])
    # 判断是 c2w 还是 w2c：按名字猜，不靠谱就默认 world_from_camera
    nm=name.lower()
    kind="c2w"  # default
    if "world_from_camera" in nm or "worldfromcamera" in nm or "wfc" in nm:
        kind="c2w"
    elif "camera_to_world" in nm or "cameratoworld" in nm or "c2w" in nm:
        kind="c2w"
    elif "w2c" in nm or "world_to_camera" in nm:
        kind="w2c"
    elif "extrin" in nm or "pose" in nm:
        # 不知道就交给 prefer
        kind = "c2w" if prefer=="world_from_camera" else "w2c"
    return M.astype(np.float64), kind

def get_intrinsics(h5, W, H):
    fx=fy=cx=cy=None
    # K 优先
    hit = find_first(h5, want_shape=(3,3), name_hints=K_HINTS)
    if hit:
        _,K = hit
        fx,fy = float(K[0,0]), float(K[1,1])
        cx,cy = float(K[0,2]), float(K[1,2])
    # 其余键
    if fx is None or fy is None:
        # 焦距
        def read_first(keys):
            for k in keys:
                if k in h5: return np.array(h5[k])
                # 允许一层 group
                for g in h5.keys():
                    try:
                        if isinstance(h5[g], h5py.Group) and k in h5[g]:
                            return np.array(h5[g][k])
                    except: pass
            return None
        v=read_first(FXFY_HINTS)
        if v is not None:
            v=np.array(v).astype(float)
            if v.shape==(2,): fx,fy=float(v[0]),float(v[1])
            elif v.shape==(): fx=fy=float(v)
        # 主点
        v=read_first(PP_HINTS)
        if v is not None:
            v=np.array(v).astype(float)
            if v.shape==(2,): cx,cy=float(v[0]),float(v[1])
    # 兜底
    if cx is None or cy is None:
        cx,cy = W/2.0, H/2.0
    if fx is None or fy is None:
        # 没有就给个合理初值，Nerfstudio 后续会refine
        fx=fy = 0.9*max(W,H)
    return fx,fy,cx,cy

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scene-root", required=True)
    ap.add_argument("--rgb-dir", required=True, 
                    help="例如 images/scene_cam_00_geometry_preview")
    ap.add_argument("--color-h5-dir", required=True, 
                    help="例如 images/scene_cam_00_final_hdf5")
    ap.add_argument("--pattern", default="*.color.jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--link-images", action="store_true",
                    help="把训练图像硬链/复制到输出目录旁边")
    ap.add_argument("--invert", action="store_true",
                    help="若HDF5里是 w2c，则可统一反转成 c2w；自动猜错时手动开这个")
    args=ap.parse_args()

    rgb_root = op.join(args.scene_root, args.rgb_dir)
    h5_root  = op.join(args.scene_root, args.color_h5_dir)

    images = sorted(glob.glob(op.join(rgb_root, args.pattern)))
    if not images:
        raise SystemExit(f"[ERR] No RGBs at {rgb_root} with {args.pattern}")
    W,H = Image.open(images[0]).size

    # 先用第一个 h5 决定读取规则
    idx0 = parse_idx(images[0])
    h5_0 = op.join(h5_root, f"frame.{idx0:04d}.color.hdf5")
    if not op.isfile(h5_0):
        # 再尝试 .h5
        h5_0 = op.join(h5_root, f"frame.{idx0:04d}.color.h5")
        if not op.isfile(h5_0):
            raise SystemExit(f"[ERR] Missing {h5_0}")

    with h5py.File(h5_0,"r") as f:
        M0, kind0 = get_pose(f, prefer="world_from_camera")
        fx,fy,cx,cy = get_intrinsics(f, W, H)
        if M0 is None:
            raise SystemExit("[ERR] Cannot find 4x4 pose in the color.hdf5; run the 'keys dump' to告诉我有哪些键")

    # 输出目录
    out_dir = op.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # 可选：把图像硬链/复制到 out_dir
    def put_image(ip):
        if not args.link_images:
            # 相对路径，尽量写成相对 out_dir 的相对路径
            return op.relpath(ip, out_dir)
        name = op.basename(ip)
        opath = op.join(out_dir, name)
        if not op.exists(opath):
            try: os.link(ip, opath)
            except: shutil.copy2(ip, opath)
        return name

    frames=[]
    miss=0
    for ip in images:
        idx = parse_idx(ip)
        h5p = op.join(h5_root, f"frame.{idx:04d}.color.hdf5")
        if not op.isfile(h5p):
            h5p = op.join(h5_root, f"frame.{idx:04d}.color.h5")
        if not op.isfile(h5p):
            miss+=1; continue
        with h5py.File(h5p,"r") as f:
            M, kind = get_pose(f, prefer="world_from_camera")
            if M is None:
                miss+=1; continue
        if args.invert or kind=="w2c":
            M = np.linalg.inv(M)
        frames.append({
            "file_path": put_image(ip),
            "transform_matrix": M.tolist()
        })

    J={
        "w": W, "h": H,
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
        "frames": frames
    }
    with open(args.out,"w") as g:
        json.dump(J, g, indent=2)
    print(f"[OK] wrote {args.out} | frames={len(frames)} | miss_h5={miss}")
    print(f"    intrinsics (fx,fy,cx,cy)=({fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f})")
