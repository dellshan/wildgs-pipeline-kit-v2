import os, os.path as op, json, argparse, math, re
import pandas as pd

CAND_IMG = ['image','frame','file','name','frame_path','path']
CAND_X   = ['x','tx','pos_x','t_x','cam_x','x_m','tx_m']
CAND_Y   = ['y','ty','pos_y','t_y','cam_y','y_m','ty_m']
CAND_Z   = ['z','tz','pos_z','t_z','cam_z','z_m','tz_m']
CAND_YAW = ['yaw','yaw_deg','heading','theta','rot_y']

def pick_col(cols, cands):
    cols = [str(c) for c in cols]
    s = {c.lower(): c for c in cols}
    for k in cands:
        if k in s: return s[k]
    for pat in cands:
        for c in cols:
            if re.search(rf'\b{pat}\b', c, flags=re.I):
                return c
    return None

def as_pose4x4(x):
    if x is None: return None
    if isinstance(x, list):
        if len(x)==4 and all(isinstance(r, list) and len(r)==4 for r in x):
            return x
        if len(x)==16:
            return [list(x[i*4:(i+1)*4]) for i in range(4)]
    if isinstance(x, dict):
        for k in ['pose','matrix','Tcw','Twc','cam2world','world2cam']:
            if k in x:
                return as_pose4x4(x[k])
        if 'R' in x and 't' in x and isinstance(x['R'], list) and len(x['R'])==3 and isinstance(x['t'], (list,tuple)) and len(x['t'])==3:
            R, t = x['R'], x['t']
            return [[R[0][0],R[0][1],R[0][2],t[0]],
                    [R[1][0],R[1][1],R[1][2],t[1]],
                    [R[2][0],R[2][1],R[2][2],t[2]],
                    [0,0,0,1]]
    return None

def t_from_T(T): return (T[0][3], T[1][3], T[2][3])

def load_pose_map(fp):
    if not fp or not op.isfile(fp): return {}
    J = json.load(open(fp, 'r'))
    mp = {}
    def img_key(v): return op.basename(str(v)) if v else None
    if isinstance(J, dict):
        if 'frames' in J and isinstance(J['frames'], list):
            for fr in J['frames']:
                img = img_key(fr.get('image') or fr.get('file') or fr.get('name'))
                T = as_pose4x4(fr) or as_pose4x4(fr.get('pose')) or as_pose4x4(fr.get('cam')) \
                    or as_pose4x4(fr.get('Tcw')) or as_pose4x4(fr.get('Twc'))
                if img and T: mp[img] = T
        elif 'by_image' in J and isinstance(J['by_image'], dict):
            for img, v in J['by_image'].items():
                imgB = img_key(img)
                T = as_pose4x4(v)
                if imgB and T: mp[imgB] = T
        else:
            for k, v in J.items():
                imgB = img_key(k) if ('.png' in k or '.jpg' in k) else img_key(v.get('image') or v.get('file') or v.get('name')) if isinstance(v, dict) else None
                T = as_pose4x4(v) if isinstance(v, (list, dict)) else None
                if imgB and T: mp[imgB] = T
    elif isinstance(J, list):
        for it in J:
            if isinstance(it, dict):
                img = img_key(it.get('image') or it.get('file') or it.get('name'))
                T = as_pose4x4(it)
                if img and T: mp[img] = T
    return mp

ap = argparse.ArgumentParser()
ap.add_argument('--csv', help='frames_views.csv（可无 x/y/z，仅提供顺序和文件名）')
ap.add_argument('--poses', help='可选：位姿 JSON（如 output/poses_est_B.json）')
ap.add_argument('--story', required=True, help='用于确定序列顺序')
ap.add_argument('--out',   required=True)
ap.add_argument('--const_step', type=float, default=0.0, help='若没有位姿信息，用该步长兜底（米）')
args = ap.parse_args()

story = json.load(open(args.story,'r'))
seq   = [op.basename(f['image']) for f in story.get('frames', [])]

# 先尝试从 CSV 拿位姿
pos = {}  # img -> (x,y,z,yaw_opt)
if args.csv and op.isfile(args.csv):
    df = pd.read_csv(args.csv)
    imgc = pick_col(df.columns, CAND_IMG)
    xc   = pick_col(df.columns, CAND_X)
    yc   = pick_col(df.columns, CAND_Y)
    zc   = pick_col(df.columns, CAND_Z)
    yawc = pick_col(df.columns, CAND_YAW)
    if imgc:
        df['_img'] = df[imgc].astype(str).apply(op.basename)
        df = df.set_index('_img')
        # 如果有 x/y/z，就存起来；没有就只存占位
        for k in df.index.unique():
            x = float(df.loc[k, xc]) if xc and k in df.index else None
            y = float(df.loc[k, yc]) if yc and k in df.index else None
            z = float(df.loc[k, zc]) if zc and k in df.index else None
            yaw = None
            if yawc and yawc in df.columns:
                val = df.loc[k, yawc]
                try:
                    if pd.notna(val): yaw = float(val)
                except: pass
            pos[k] = (x, y, z, yaw)

# 若 CSV 没有 x/y/z，再尝试 poses.json
if (not any(v[:3] != (None, None, None) for v in pos.values())) and args.poses:
    P = load_pose_map(args.poses)
    for img, T in P.items():
        x, y, z = t_from_T(T)
        pos[img] = (x, y, z, None)

links=[]
miss=0

def make_text(dist, yaw):
    dtxt = f"{dist:.1f}" if dist >= 0.05 else "0.0"
    ytxt = f"{round(yaw):d}" if yaw is not None else "0"
    return f"forward {dtxt} m, yaw {ytxt}°"

for i in range(len(seq)-1):
    a, b = seq[i], seq[i+1]
    xa=ya=za=xb=yb=zb=None
    ya_opt=None
    if a in pos and pos[a][:3] != (None, None, None):
        xa, ya, za, _ = pos[a]
    if b in pos and pos[b][:3] != (None, None, None):
        xb, yb, zb, yb_opt = pos[b]
        ya_opt = yb_opt

    if None not in (xa,ya,za,xb,yb,zb):
        dx,dy,dz = xb-xa, yb-ya, zb-za
        dist = (dx*dx+dy*dy+dz*dz)**0.5
        yaw  = ya_opt if ya_opt is not None else math.degrees(math.atan2(dy, dx))
        links.append({
            "from": a, "to": b,
            "dx": dx, "dy": dy, "dz": dz,
            "dist_m": dist, "yaw_deg": yaw,
            "text": make_text(dist, yaw)
        })
    else:
        # 没位姿：按常量步长兜底或只给文案
        if args.const_step > 0:
            links.append({
                "from": a, "to": b,
                "dx": args.const_step, "dy": 0.0, "dz": 0.0,
                "dist_m": args.const_step, "yaw_deg": 0.0,
                "text": make_text(args.const_step, 0.0)
            })
        else:
            miss += 1
            links.append({"from": a, "to": b, "text": "move to next waypoint"})

json.dump({"links": links}, open(args.out,'w'), ensure_ascii=False, indent=2)
print(f"[OK] wrote {args.out} | pairs={len(links)} | miss={miss} | have_pose_pairs={sum(1 for lk in links if 'dx' in lk)}")
