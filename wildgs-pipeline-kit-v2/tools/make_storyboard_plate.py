#!/usr/bin/env python3
import os, os.path as op, argparse, csv, glob
import numpy as np
from PIL import Image, ImageOps, ImageFont, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"]=200

def read_features(p):
    xs, ys = [], []
    for r in csv.DictReader(open(p)):
        xs.append(float(r["coverage"]))
        ys.append(float(r["sharpness"]))
    return np.array(xs), np.array(ys)

def read_selected(p):
    sel=[]
    for i,r in enumerate(csv.DictReader(open(p))):
        sel.append(r["image"])
    return sel

def save_hist(x, title, outp):
    fig = plt.figure(figsize=(3.2,2.2))
    plt.hist(x, bins=20)
    plt.title(title, fontsize=9)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)

def grid12(images, outp, cols=4, pad=6, cell_h=220):
    # 等高自适应，铺 3x4
    rows = int(np.ceil(len(images)/cols))
    # 先统一高度
    ims=[]
    cell_w=0
    for p in images:
        im = Image.open(p).convert("RGB")
        h=cell_h
        w=int(im.width*h/im.height)
        im=im.resize((w,h), Image.LANCZOS)
        ims.append(im)
        cell_w=max(cell_w,w)
    W = cols*cell_w + (cols+1)*pad
    H = rows*cell_h + (rows+1)*pad + 24  # + 标题条
    canvas = Image.new("RGB",(W,H),(255,255,255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad,4), "AutoView top-12", fill=(0,0,0))
    y=pad+20
    k=0
    for r in range(rows):
        x=pad
        for c in range(cols):
            if k>=len(ims): break
            im=ims[k]
            # 置中每格
            offx = x + (cell_w-im.width)//2
            canvas.paste(im, (offx, y))
            x += cell_w + pad
            k+=1
        y += cell_h + pad
    canvas.save(outp)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scene_root", required=True)  # outputs/hypersim/ai_001_001
    ap.add_argument("--out_png", required=True)
    args=ap.parse_args()
    R=args.scene_root
    feat_csv = op.join(R, "features.csv")
    sel_csv  = op.join(R, "autoview_top12.csv")

    cov, shp = read_features(feat_csv)
    os.makedirs(op.join(R,"figs"), exist_ok=True)
    h_cov = op.join(R,"figs","hist_cov.png")
    h_shp = op.join(R,"figs","hist_shp.png")
    save_hist(cov, "Mask union coverage", h_cov)
    save_hist(shp, "Laplacian sharpness (norm.)", h_shp)

    # 优先 vis_masks12 -> vis_selected12 -> images
    def pick_dir(name):
        d = op.join(R, name)
        return d if op.isdir(d) and glob.glob(op.join(d,"*.jpg")) else None
    d_vis = pick_dir("vis_masks12") or pick_dir("vis_selected12") or op.join(R,"images")
    selected = read_selected(sel_csv)
    sel_paths = [op.join(d_vis, s) for s in selected if op.isfile(op.join(d_vis, s))]
    if len(sel_paths)<len(selected):
        # 回退扩展名差异
        fixed=[]
        for s in selected:
            hit=None
            for ext in [".jpg",".png",".JPG",".PNG",".jpeg",".JPEG"]:
                p=op.join(d_vis, op.splitext(s)[0]+ext)
                if op.isfile(p): hit=p; break
            if hit: fixed.append(hit)
        sel_paths=fixed

    grid = op.join(R,"figs","top12_grid.png")
    grid12(sel_paths, grid)

    # 拼接三图为面板
    a=Image.open(h_cov).convert("RGB")
    b=Image.open(h_shp).convert("RGB")
    g=Image.open(grid).convert("RGB")
    pad=20
    W = a.width + b.width + g.width + 4*pad
    H = max(a.height, b.height, g.height) + 2*pad
    canvas = Image.new("RGB",(W,H),(255,255,255))
    x=pad
    for im in [a,b,g]:
        y = (H-im.height)//2
        canvas.paste(im, (x,y))
        x += im.width + pad
    canvas.save(args.out_png)
    print("[OK] wrote plate:", args.out_png)

if __name__=="__main__":
    main()
