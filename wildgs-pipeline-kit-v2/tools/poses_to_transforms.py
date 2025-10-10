#!/usr/bin/env python3
import os, os.path as op, json, argparse, csv, glob
import numpy as np

def read_image_list(img_dir):
    names = []
    for p in sorted(glob.glob(op.join(img_dir, "*.jpg")) + glob.glob(op.join(img_dir, "*.png"))):
        names.append(op.basename(p))
    return names

def parse_csv_poses(csv_path):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        keys = [k.strip() for k in r.fieldnames]
        has_intr = all(k in keys for k in ["fx","fy","cx","cy"])
        # 旋转/平移列名试图自动匹配 r00..r22,t0..t2 或 m00..m33
        has_r = all(f"r{i}{j}" in keys for i in range(3) for j in range(3))
        has_t = all(f"t{k}" in keys for k in range(3))
        has_m = all(f"m{i}{j}" in keys for i in range(4) for j in range(4))
        for row in r:
            name = row.get("image") or row.get("file") or row.get("filename")
            if not name: 
                continue
            if has_m:
                M = np.eye(4, dtype=np.float64)
                for i in range(4):
                    for j in range(4):
                        M[i,j] = float(row[f"m{i}{j}"])
            elif has_r and has_t:
                R = np.array([[float(row[f"r{i}{j}"]) for j in range(3)] for i in range(3)], dtype=np.float64)
                t = np.array([float(row[f"t{k}"]) for k in range(3)], dtype=np.float64).reshape(3,1)
                M = np.eye(4, dtype=np.float64); M[:3,:3]=R; M[:3,3]=t[:,0]
            else:
                raise RuntimeError("CSV缺少姿态列：需要 r00..r22,t0..t2 或 m00..m33")
            item = {"image": name, "M": M}
            if has_intr:
                item.update({k: float(row[k]) for k in ["fx","fy","cx","cy"]})
            rows.append(item)
    return rows

def read_txt_4x4(dir_path, images):
    out=[]
    for name in images:
        stem = op.splitext(name)[0]
        cand = [op.join(dir_path, stem+".txt"), op.join(dir_path, stem+".pose.txt")]
        posep = None
        for p in cand:
            if op.isfile(p): posep=p; break
        if not posep: 
            continue
        M = np.loadtxt(posep).reshape(4,4).astype(np.float64)
        out.append({"image": name, "M": M})
    return out

def invert_pose(M):
    R = M[:3,:3]; t = M[:3,3:4]
    Mi = np.eye(4, dtype=np.float64)
    Mi[:3,:3] = R.T
    Mi[:3,3:4] = -R.T @ t
    return Mi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图像目录（jpg/png）")
    ap.add_argument("--poses",  required=True, help="CSV 文件或 4x4 文本矩阵目录")
    ap.add_argument("--matrix", default="c2w", choices=["c2w","w2c"], help="输入矩阵的含义")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--out", required=True, help="输出 transforms.json 路径")
    args = ap.parse_args()

    os.makedirs(op.dirname(args.out), exist_ok=True)

    images = read_image_list(args.images)
    if not images:
        raise RuntimeError("images 目录下没找到 jpg/png")

    # 读取位姿
    items = []
    if op.isdir(args.poses):
        items = read_txt_4x4(args.poses, images)
    else:
        items = parse_csv_poses(args.poses)

    name2M = {it["image"]: it["M"] for it in items}
    # 汇总内参（优先 CSV 内的；否则用命令行给的；最后兜底用图像中心+等焦距）
    sample_img = op.join(args.images, images[0])
    from PIL import Image
    W,H = Image.open(sample_img).size

    fx = args.fx; fy = args.fy; cx = args.cx; cy = args.cy
    if fx is None or fy is None or cx is None or cy is None:
        # 尝试从 CSV 的第一行读取
        for it in items:
            if all(k in it for k in ["fx","fy","cx","cy"]):
                fx,fy,cx,cy = it["fx"],it["fy"],it["cx"],it["cy"]
                break
    if fx is None or fy is None: 
        fx = fy = 0.9 * W  # 合理兜底
    if cx is None or cy is None:
        cx, cy = W/2.0, H/2.0

    frames=[]
    miss=0
    for name in images:
        if name not in name2M:
            miss += 1
            continue
        M = name2M[name]
        M = M if args.matrix=="c2w" else invert_pose(M)
        # to python list
        Mlist = [[float(M[i,j]) for j in range(4)] for i in range(4)]
        frames.append({"file_path": name, "transform_matrix": Mlist})

    data = {
        "camera_model": "OPENCV",
        "fl_x": float(fx), "fl_y": float(fy),
        "cx": float(cx), "cy": float(cy),
        "w": int(W), "h": int(H),
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": frames
    }
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] transforms.json -> {args.out} | frames={len(frames)} | missing={miss}")

if __name__ == "__main__":
    main()
