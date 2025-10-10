#!/usr/bin/env python3
import os, os.path as op, argparse, shutil, math
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

CAM_WHITELIST = {"CAM_FRONT"}

def quat_to_yaw_deg(q: Quaternion):
    # ego_pose 的旋转是 w,x,y,z，转换成朝向（绕Z轴）
    # 参考右手坐标，nusc 使用车体坐标系，取偏航角
    yaw = q.yaw_pitch_roll[0]
    return math.degrees(yaw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes/raw")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="outputs/nuscenes/mini")
    ap.add_argument("--per_scene", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(op.join(args.out, "images"), exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # 每个 scene 限制 N 帧（只取 CAM_FRONT）
    picked = []
    for scene in nusc.scene:
        sample_token = scene['first_sample_token']
        k = 0
        while sample_token and k < args.per_scene:
            sample = nusc.get('sample', sample_token)
            for cam, token in sample['data'].items():
                if cam not in CAM_WHITELIST:
                    continue
                sd = nusc.get('sample_data', token)
                src = op.join(nusc.dataroot, sd['filename'])
                dst_name = f"{scene['name']}_{sd['timestamp']}_{cam}.jpg"
                dst = op.join(args.out, "images", dst_name)
                os.makedirs(op.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

                # ego pose
                ep = nusc.get('ego_pose', sd['ego_pose_token'])
                t = ep['translation']  # x,y,z
                q = Quaternion(ep['rotation'])
                yaw = quat_to_yaw_deg(q)
                picked.append((dst_name, sd['timestamp'], scene['name'], cam, t[0], t[1], t[2], yaw))
                k += 1
                if k >= args.per_scene:
                    break
            sample_token = sample['next']

    # 写 frames_views.csv
    csv = op.join(args.out, "frames_views.csv")
    with open(csv, "w") as f:
        f.write("image,ts,scene,cam,tx,ty,tz,yaw_deg\n")
        for r in picked:
            f.write(",".join(map(str, r)) + "\n")
    print("[OK] nuScenes sample ->", args.out, "| images=", len(picked))

if __name__ == "__main__":
    main()