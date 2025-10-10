import os, os.path as op, json, argparse, math, csv

def as_pose4x4(x):
    if x is None: return None
    if isinstance(x, list):
        if len(x)==4 and all(isinstance(r, list) and len(r)==4 for r in x): return x
        if len(x)==16: return [list(x[i*4:(i+1)*4]) for i in range(4)]
    if isinstance(x, dict):
        for k in ['pose','matrix','Tcw','Twc','cam2world','world2cam']:
            if k in x: return as_pose4x4(x[k])
        if 'R' in x and 't' in x and len(x['R'])==3 and len(x['t'])==3:
            R, t = x['R'], x['t']
            return [[R[0][0],R[0][1],R[0][2],t[0]],
                    [R[1][0],R[1][1],R[1][2],t[1]],
                    [R[2][0],R[2][1],R[2][2],t[2]],
                    [0,0,0,1]]
    return None

def yaw_from_T(T):
    r00, r10 = T[0][0], T[1][0]
    return math.degrees(math.atan2(r10, r00))

ap = argparse.ArgumentParser()
ap.add_argument('--poses', required=True)
ap.add_argument('--out_csv', required=True)
args = ap.parse_args()

J = json.load(open(args.poses,'r'))
rows=[]
def img_key(v): return op.basename(str(v)) if v else None

if isinstance(J, dict) and 'frames' in J:
    for fr in J['frames']:
        img = img_key(fr.get('image') or fr.get('file') or fr.get('name'))
        T = as_pose4x4(fr) or as_pose4x4(fr.get('pose')) or as_pose4x4(fr.get('cam')) \
            or as_pose4x4(fr.get('Tcw')) or as_pose4x4(fr.get('Twc'))
        if not (img and T): continue
        x,y,z = T[0][3], T[1][3], T[2][3]
        yaw   = yaw_from_T(T)
        rows.append(dict(image=img, x=x, y=y, z=z, yaw_deg=yaw))
elif isinstance(J, dict) and 'by_image' in J:
    for img, v in J['by_image'].items():
        T = as_pose4x4(v)
        if not T: continue
        x,y,z = T[0][3], T[1][3], T[2][3]
        yaw   = yaw_from_T(T)
        rows.append(dict(image=img_key(img), x=x, y=y, z=z, yaw_deg=yaw))

with open(args.out_csv,'w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=['image','x','y','z','yaw_deg'])
    w.writeheader(); w.writerows(rows)
print(f"[OK] wrote {args.out_csv} | rows={len(rows)}")
