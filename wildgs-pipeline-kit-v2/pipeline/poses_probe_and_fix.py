import json, os, os.path as op, sys, math

def as_pose4x4(x):
    # 接受: 嵌套 4x4, 或 dict 含 'matrix'/'pose'/'Tcw'/'Twc' 等，或平铺 16 长度
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
        # 有时位姿存为 {"R":[[...],[...],[...]], "t":[x,y,z]}
        if 'R' in x and 't' in x:
            R, t = x['R'], x['t']
            if isinstance(R, list) and len(R)==3 and all(len(r)==3 for r in R) and isinstance(t, (list,tuple)) and len(t)==3:
                return [ [R[0][0],R[0][1],R[0][2],t[0]],
                         [R[1][0],R[1][1],R[1][2],t[1]],
                         [R[2][0],R[2][1],R[2][2],t[2]],
                         [0,0,0,1] ]
    return None

def t_from_T(T):
    return (T[0][3], T[1][3], T[2][3])

def yaw_from_T(T):
    # 从旋转矩阵取 yaw（绕 z），近似计算
    r00, r01 = T[0][0], T[0][1]
    r10, r11 = T[1][0], T[1][1]
    yaw = math.degrees(math.atan2(r10, r00))
    # 也可以用 atan2(-r20, sqrt(r00^2+r10^2)) 视坐标系而定，这里选常见定义
    return yaw

def load_pose_map(fp):
    J = json.load(open(fp,'r'))
    mp = {}  # image_basename -> 4x4
    def img_key(x):
        return op.basename(str(x)) if x else None
    if isinstance(J, dict):
        if 'frames' in J and isinstance(J['frames'], list):
            for fr in J['frames']:
                img = img_key(fr.get('image') or fr.get('file') or fr.get('name'))
                T = as_pose4x4(fr) or as_pose4x4(fr.get('pose')) or as_pose4x4(fr.get('cam')) or as_pose4x4(fr.get('Tcw')) or as_pose4x4(fr.get('Twc'))
                if img and T: mp[img]=T
        elif 'by_image' in J and isinstance(J['by_image'], dict):
            for img, v in J['by_image'].items():
                imgB = img_key(img)
                T = as_pose4x4(v)
                if imgB and T: mp[imgB]=T
        else:
            # 平铺 dict：尝试每个 value
            for k,v in J.items():
                imgB = img_key(k) if ('.png' in k or '.jpg' in k) else img_key(v.get('image') or v.get('file') or v.get('name') if isinstance(v,dict) else None)
                T = as_pose4x4(v) if isinstance(v, (list,dict)) else None
                if imgB and T: mp[imgB]=T
    elif isinstance(J, list):
        for it in J:
            if isinstance(it, dict):
                img = img_key(it.get('image') or it.get('file') or it.get('name'))
                T = as_pose4x4(it)
                if img and T: mp[img]=T
    return mp

if __name__ == "__main__":
    story_fp = sys.argv[1]  # output/view_story_B_2d/story.with_names.json
    poses_fp = sys.argv[2]  # output/poses_B.json
    links_fp = sys.argv[3]  # output/view_story_B_2d/links.json

    story = json.load(open(story_fp,'r'))
    poses = load_pose_map(poses_fp)
    frames = [op.basename(fr.get('image')) for fr in story.get('frames',[])]

    print(f"[i] loaded poses: {len(poses)} | frames in story: {len(frames)}")
    # 预览前 12 段
    for i in range(min(12, len(frames)-1)):
        a, b = frames[i], frames[i+1]
        Ta, Tb = poses.get(a), poses.get(b)
        if not (Ta and Tb):
            print(f"[{i:02d}] {a}->{b}: MISSING POSE")
            continue
        xa,ya,za = t_from_T(Ta)
        xb,yb,zb = t_from_T(Tb)
        dx,dy,dz = xb-xa, yb-ya, zb-za
        dist = (dx*dx+dy*dy+dz*dz)**0.5
        yawb = yaw_from_T(Tb)
        print(f"[{i:02d}] {a}->{b}: d=({dx:.3f},{dy:.3f},{dz:.3f}) | |Δ|={dist:.3f} m | yaw_b={yawb:.1f}°")

    # 把 links.json 补齐
    if os.path.isfile(links_fp):
        L = json.load(open(links_fp,'r'))
    else:
        L = {"links":[]}
    out_links = []
    for i in range(len(frames)-1):
        a, b = frames[i], frames[i+1]
        Ta, Tb = poses.get(a), poses.get(b)
        rec = {"from": a, "to": b}
        if Ta and Tb:
            xa,ya,za = t_from_T(Ta)
            xb,yb,zb = t_from_T(Tb)
            dx,dy,dz = xb-xa, yb-ya, zb-za
            dist = (dx*dx+dy*dy+dz*dz)**0.5
            yawb = yaw_from_T(Tb)
            rec.update(dict(dx=dx,dy=dy,dz=dz,dist_m=dist,yaw_deg=yawb,
                            text=f"forward {dist:.1f} m, yaw {yawb:.0f}°"))
        else:
            rec["text"] = "move to next waypoint"
        out_links.append(rec)
    json.dump({"links": out_links}, open(links_fp,'w'), ensure_ascii=False, indent=2)
    print(f"[OK] links enriched -> {links_fp}")
