#!/usr/bin/env python3
import os, os.path as op, argparse, csv, glob
import numpy as np
from PIL import Image
from collections import defaultdict

NAV_SET = {  # navigation-relevant categories
    "person","chair","table","desk","door","box","trash bin","bag","ladder",
    "cabinet","shelf","sofa","cart","bucket","bicycle","plant","bench"
}

def read_features(p):
    D={}
    for i,r in enumerate(csv.DictReader(open(p))):
        D[r["image"]] = (float(r["coverage"]), float(r["sharpness"]))
    return D

def read_objects(p):
    # objects_views.csv rows: image,inst_id,mask,pred_label,score,topk_labels,topk_scores
    per = defaultdict(list)
    if not op.isfile(p): return per
    for r in csv.DictReader(open(p)):
        per[r["image"]].append((r["pred_label"], float(r["score"])))
    return per

def embed64(imgp):
    im = Image.open(imgp).convert("RGB").resize((64,64))
    v = np.asarray(im, dtype=np.float32).reshape(-1,3).mean(axis=1)  # 4096-d luminance-ish
    v = (v - v.mean()) / (v.std()+1e-6)
    return v

def cos(a,b): return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--objects_csv", required=True)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--w_cov", type=float, default=1.0)
    ap.add_argument("--w_shp", type=float, default=0.3)
    ap.add_argument("--w_sem", type=float, default=0.5)
    ap.add_argument("--lambda_div", type=float, default=0.6)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    F = read_features(args.features_csv)
    O = read_objects(args.objects_csv)

    # precompute semantics per image: max score over NAV_SET labels
    sem = {}
    for img, lst in O.items():
        s = 0.0
        for lab,sc in lst:
            if lab in NAV_SET:
                s = max(s, sc)
        sem[img] = s

    # collect candidates that exist on disk
    cands=[]
    for p in sorted(glob.glob(os.path.join(args.images,"*.jpg"))):
        name = os.path.basename(p)
        if name in F:
            cands.append(name)

    # precompute embeddings
    embs={}
    for name in cands:
        embs[name]=embed64(os.path.join(args.images,name))

    def qscore(name):
        cov, shp = F[name]
        se = sem.get(name, 0.0)
        return args.w_cov*cov + args.w_shp*shp + args.w_sem*se

    S=[]  # selected names
    while len(S) < min(args.K, len(cands)):
        best=None; best_gain=-1e9
        for name in cands:
            if name in S: continue
            q = qscore(name)
            div_term = 0.0
            if S:
                sim = max(cos(embs[name], embs[s]) for s in S)
                div_term = (1.0 - sim)
            gain = q + args.lambda_div*div_term
            if gain > best_gain:
                best_gain, best = gain, name
        S.append(best)

    # write out
    rows=[("rank","image","score","coverage","sharpness","semantics")]
    for i,name in enumerate(S,1):
        cov,shp=F[name]; se=sem.get(name,0.0)
        rows.append((i,name, f"{qscore(name):.4f}", f"{cov:.4f}", f"{shp:.4f}", f"{se:.4f}"))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv,"w",newline="") as f: csv.writer(f).writerows(rows)
    print("[OK] wrote", args.out_csv, "| K=", len(S))
if __name__=="__main__":
    main()
