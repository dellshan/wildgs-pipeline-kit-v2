#!/usr/bin/env python3
import os, os.path as op, argparse, json, math, shutil, glob

def yaw_from_c2w(m34):
    r00,r01,r02,tx = m34[0]
    r10,r11,r12,ty = m34[1]
    r20,r21,r22,tz = m34[2]
    return math.degrees(math.atan2(r10, r00)), (tx,ty,tz)

def load_frames_from_transforms(tpath, num):
    J = json.load(open(tpath))
    frames = J.get("frames", [])
    if not frames: return []
    out=[]
    for fr in frames[:num]:
        fp = fr["file_path"]
        # 常见 file_path 可能不带后缀，尝试匹配
        cand = [fp] if op.splitext(fp)[1] else [fp+x for x in (".png",".jpg",".jpeg",".JPG",".PNG")]
        src=None
        for c in cand:
            if op.isfile(c): src=c; break
        if src is None:
            # 再递归搜一下同名
            base = op.basename(fp)
            matches = glob.glob(f"**/{base}*", recursive=True)
            src = matches[0] if matches else None
        if src is None: 
            continue
        mat = fr.get("transform_matrix")
        if mat and len(mat)>=3 and len(mat[0])>=4:
            m34 = [row[:4] for row in mat[:3]]
            yaw,(tx,ty,tz) = yaw_from_c2w(m34)
        else:
            yaw,tx,ty,tz = 0.0,0.0,0.0,0.0
        out.append((src, tx,ty,tz, yaw))
    return out

def scan_images_fallback(sroot, num):
    pats = ["images/**", "**/images/**", "**/*.png", "**/*.jpg", "**/*.jpeg"]
    files=[]
    for p in pats:
        files += [x for x in glob.glob(op.join(sroot, p), recursive=True) 
                  if op.isfile(x) and op.splitext(x)[1].lower() in (".png",".jpg",".jpeg")]
        if len(files)>=num: break
    files = sorted(set(files))[:num]
    return [(f, 0.0,0.0,0.0, 0.0) for f in files]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="scene name under data/mipnerf360/raw/<scene>")
    ap.add_argument("--root", default="data/mipnerf360/raw")
    ap.add_argument("--out_root", default="outputs/mipnerf360")
    ap.add_argument("--num", type=int, default=80)
    args = ap.parse_args()

    sroot = op.join(args.root, args.scene)
    if not op.isdir(sroot):
        raise SystemExit(f"[ERR] scene folder not found: {sroot}")

    # 1) 优先找 transforms*.json（支持 train/val/test 合并）
    tpaths = []
    for name in ["transforms.json","transforms_train.json","transforms_val.json","transforms_test.json","dataset.json"]:
        p = op.join(sroot, name)
        if op.isfile(p): tpaths.append(p)

    frames=[]
    for tp in tpaths:
        frames += load_frames_from_transforms(tp, args.num - len(frames))
        if len(frames) >= args.num: break

    # 2) 退化：无 transforms 的场景直接扫 images
    if not frames:
        print(f"[WARN] no transforms*.json under {sroot}, fallback to scanning images/* (poses as 0)")
        frames = scan_images_fallback(sroot, args.num)

    if not frames:
        raise SystemExit(f"[ERR] no images found under {sroot}")

    out = op.join(args.out_root, args.scene)
    os.makedirs(op.join(out, "images"), exist_ok=True)
    rows=[]
    for src,tx,ty,tz,yaw in frames:
        dst = op.join(out, "images", op.basename(src))
        if op.realpath(src) != op.realpath(dst):
            os.makedirs(op.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        rows.append((op.basename(dst), 0, args.scene, "rgb", tx,ty,tz, yaw))

    with open(op.join(out, "frames_views.csv"), "w") as f:
        f.write("image,ts,scene,cam,tx,ty,tz,yaw_deg\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"[OK] mipnerf360 sample -> {out} | images= {len(rows)}")

if __name__ == "__main__":
    main()
