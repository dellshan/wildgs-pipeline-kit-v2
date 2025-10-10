# pipeline/estimate_poses_from_frames.py
import cv2, os, os.path as op, json, argparse, math
import numpy as np
from glob import glob

def yaw_deg_from_R(R):
    # 绕 Z 轴的 yaw（相机坐标，近似）
    return math.degrees(math.atan2(R[1,0], R[0,0]))

def K_from_size(w,h,f=None):
    if f is None: f = 0.9 * max(w,h)  # 一个保守假设
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1.0]], dtype=np.float64)

def load_gray(p): 
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None: raise FileNotFoundError(p)
    return g

def essential_pose(g1,g2,K):
    # ORB 特征
    orb = cv2.ORB_create(2000)
    k1,d1 = orb.detectAndCompute(g1, None)
    k2,d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None: return None,None,0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ms = bf.match(d1,d2)
    if len(ms) < 12: return None,None,0
    pts1 = np.float32([k1[m.queryIdx].pt for m in ms])
    pts2 = np.float32([k2[m.trainIdx].pt for m in ms])
    # E
    E,mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
    if E is None: return None,None,0
    _, R, t, mask2 = cv2.recoverPose(E, pts1, pts2, K)
    inl = int(mask2.sum()) if mask2 is not None else 0
    return R, t.reshape(3), inl

def avg_flow_px(g1,g2):
    # Farneback，用于估计像素尺度
    flow = cv2.calcOpticalFlowFarneback(g1,g2,None,0.5,3,21,3,5,1.2,0)
    mag = np.linalg.norm(flow, axis=-1)
    return float(np.median(mag)), float(np.mean(mag))

def traj_from_images(imgs, px2m=0.01):
    poses = {}
    if len(imgs)==0: return poses
    g0 = load_gray(imgs[0])
    h,w = g0.shape
    T = np.eye(4,dtype=np.float64)
    poses[op.basename(imgs[0])] = T.tolist()
    K = K_from_size(w,h)

    for i in range(len(imgs)-1):
        g1, g2 = g0, load_gray(imgs[i+1])
        R,t,inl = essential_pose(g1,g2,K)
        # 流作为尺度
        medpx,_ = avg_flow_px(g1,g2)
        scale = max(medpx*px2m, 1e-6)
        if R is None or t is None:
            # 退化：只用光流方向估计 yaw，其余置零
            dx_pix = 0.0; dy_pix = 0.0
            yaw = 0.0
        else:
            # 相机坐标的增量平移方向（单位向量）× scale
            dt = (t/np.linalg.norm(t)) * scale
            # 将增量并到全局：T_new = T * [R | t]
            dT = np.eye(4); dT[:3,:3] = R; dT[:3,3] = dt
            T = T @ dT

        poses[op.basename(imgs[i+1])] = T.tolist()
        g0 = g2
    return poses

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True, help='帧目录（包含 path_XXXXX.png）')
    ap.add_argument('--story', required=True, help='用于限定和排序的 story.json')
    ap.add_argument('--px2m', type=float, default=0.01, help='像素到米的换算（默认 100px≈1m）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    story = json.load(open(args.story,'r'))
    order = [op.join(args.frames, op.basename(fr['image'])) for fr in story.get('frames',[])]
    order = [p for p in order if op.isfile(p)]
    poses = traj_from_images(order, px2m=args.px2m)
    json.dump({'by_image': poses}, open(args.out,'w'), ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out} | frames={len(poses)}")

if __name__ == '__main__':
    main()
