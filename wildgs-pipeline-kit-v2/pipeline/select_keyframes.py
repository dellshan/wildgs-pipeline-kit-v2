#!/usr/bin/env python3
import os, sys, glob, math, shutil, argparse, numpy as np
ap=argparse.ArgumentParser()
ap.add_argument("traj"); ap.add_argument("rgb_dir"); ap.add_argument("out_dir")
ap.add_argument("--move-th",type=float,default=0.10); ap.add_argument("--deg-th",type=float,default=5.0)
ap.add_argument("--max-kf",type=int,default=30); args=ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
poses=[]
for ln in open(args.traj):
    if not ln.strip() or ln.startswith("#"): continue
    ps=ln.split()
    if len(ps)<8: continue
    ts,tx,ty,tz,qx,qy,qz,qw=ps[:8]
    poses.append((int(float(ts)), np.array([float(tx),float(ty),float(tz)]), np.array([float(qx),float(qy),float(qz),float(qw)])))
if not poses: sys.exit("no poses")
imgs=sorted(sum([glob.glob(os.path.join(args.rgb_dir,e)) for e in ("*.png","*.jpg","*.jpeg")],[]))
if not imgs: sys.exit("no images")
def ang(q1,q2): import numpy as n, math; return math.degrees(2*math.acos(abs(n.clip(n.dot(q1,q2),-1,1))))
idx=[0]; lastT,lastq=poses[0][1],poses[0][2]
for i,(_,T,q) in enumerate(poses[1:],1):
    if np.linalg.norm(T-lastT)>args.move_th or ang(q,lastq)>args.deg_th:
        idx.append(i); lastT,lastq=T,q
    if len(idx)>=args.max_kf: break
m=min(len(imgs),len(poses)); idx=[i for i in idx if i<m]
for k,src in enumerate([imgs[i] for i in idx]):
    import os,shutil; name=f"kf_{k:03d}{os.path.splitext(src)[1].lower()}"; shutil.copy(src, os.path.join(args.out_dir,name))
print(f"Selected {len(idx)} keyframes -> {args.out_dir}")
