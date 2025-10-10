import os, json, math, warnings, argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_out(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def imread_gray(fp):
    im = Image.open(fp).convert("L")
    return np.array(im, dtype=np.uint8)

def plot_line(xs, ys, title, xlabel, ylabel, out_png):
    plt.figure(figsize=(8,3))
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_hist(vals, bins, title, xlabel, out_png):
    plt.figure(figsize=(5,3))
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def compute_sharpness(frames_dir, images_sorted):
    sharp = []
    for fn in tqdm(images_sorted, desc="sharpness", ncols=80):
        fp = os.path.join(frames_dir, fn)
        g = imread_gray(fp)
        lv = cv2.Laplacian(g, cv2.CV_64F).var()
        sharp.append(lv)
    return np.array(sharp, float)

def load_union_mask(mask_root, mask_files, size_hw):
    H,W = size_hw
    union = np.zeros((H,W), np.uint8)
    for mf in mask_files:
        mp = os.path.join(mask_root, mf)
        if not os.path.isfile(mp):
            continue
        m = imread_gray(mp)
        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)
        union = np.maximum(union, m)
    return union

def compute_coverage(frames_dir, images_sorted, df, mask_root):
    cover = []
    areas = []
    # Build mapping: image -> list of mask filenames
    df_map = df.groupby("image")["mask"].apply(lambda s: [str(x) for x in s]).to_dict()
    for fn in tqdm(images_sorted, desc="coverage", ncols=80):
        fp = os.path.join(frames_dir, fn)
        if not os.path.isfile(fp):
            cover.append(0.0); areas.append(0); continue
        im = Image.open(fp).convert("RGB")
        W,H = im.size
        mask_files = df_map.get(fn, [])
        if not mask_files:
            cover.append(0.0); areas.append(W*H); continue
        union = load_union_mask(mask_root, mask_files, (H,W))
        c = float((union>0).sum())/(W*H)
        cover.append(c); areas.append(W*H)
    return np.array(cover, float), np.array(areas, float)

def evenly_pick(lst, k):
    if k<=0: return []
    if k>=len(lst): return list(lst)
    idx = np.linspace(0, len(lst)-1, num=k, dtype=int)
    return [lst[i] for i in idx]

def make_storyboard(frames_dir, mask_root, df_map, pick, out_png, cols=4, alpha=160):
    if not pick: return
    im0 = Image.open(os.path.join(frames_dir, pick[0])).convert("RGB")
    W,H = im0.size
    rows = math.ceil(len(pick)/cols)
    canvas = Image.new("RGB", (cols*W, rows*H), (255,255,255))
    for idx,fn in enumerate(pick):
        base = Image.open(os.path.join(frames_dir, fn)).convert("RGBA")
        masks = df_map.get(fn, [])
        if masks:
            union = load_union_mask(mask_root, masks, (base.height, base.width))
            color = Image.new("RGBA", (W,H), (255,0,0,0))
            a = Image.fromarray((union>0).astype(np.uint8)*alpha)
            color.putalpha(a)
            base = Image.alpha_composite(base, color)
        r = idx//cols; c = idx%cols
        canvas.paste(base.convert("RGB"), (c*W, r*H))
    canvas.save(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    cfg = json.load(open(args.config, "r"))

    frames_dir   = cfg["frames_dir"]
    instances_csv= cfg["instances_csv"]
    mask_root    = cfg.get("mask_root", "")
    out_dir      = cfg.get("out_dir", "figs")
    tables_dir   = cfg.get("tables_dir", "tables")
    topk_list    = cfg.get("topk_list", [10,20,50,100])
    storyboard_n = int(cfg.get("storyboard_topn", 12))

    ensure_out(out_dir, tables_dir)

    df = pd.read_csv(instances_csv)
    for c in ["image","mask","conf"]:
        if c not in df.columns:
            raise RuntimeError(f"CSV 缺少列: {c}")

    images_sorted = sorted(df["image"].unique().tolist())

    # (1) confidence histogram
    conf_vals = df["conf"].astype(float).values
    plot_hist(conf_vals, bins=np.linspace(0,1,21),
              title="Confidence Histogram",
              xlabel="confidence",
              out_png=os.path.join(out_dir, "conf_hist.png"))
    pd.DataFrame({"conf": conf_vals}).to_csv(os.path.join(tables_dir,"conf_values.csv"), index=False)

    # (2) instances per frame
    inst_pf = df.groupby("image").size().reset_index(name="instances")
    inst_pf = inst_pf.sort_values("image")
    inst_pf.to_csv(os.path.join(tables_dir,"instances_per_frame.csv"), index=False)
    plot_line(np.arange(len(inst_pf)), inst_pf["instances"].values,
              title="Instances per Frame",
              xlabel="frame index (sorted by filename)", ylabel="#instances",
              out_png=os.path.join(out_dir, "instances_per_frame.png"))

    # (3) sharpness curve
    try:
        sharp_vals = compute_sharpness(frames_dir, images_sorted)
        pd.DataFrame({"image":images_sorted, "sharpness":sharp_vals}).to_csv(
            os.path.join(tables_dir,"sharpness.csv"), index=False)
        plot_line(np.arange(len(images_sorted)), sharp_vals,
                  title="Sharpness (Laplacian Var)",
                  xlabel="frame index (sorted by filename)", ylabel="variance",
                  out_png=os.path.join(out_dir, "sharpness_curve.png"))
    except Exception as e:
        warnings.warn(f"Sharpness 计算失败（略过）: {e}")
        sharp_vals = None

    # (4) coverage curve (if mask_root exists)
    cover_vals = None
    if mask_root and os.path.isdir(mask_root):
        cover_vals, areas = compute_coverage(frames_dir, images_sorted, df, mask_root)
        pd.DataFrame({"image":images_sorted, "coverage":cover_vals}).to_csv(
            os.path.join(tables_dir,"coverage.csv"), index=False)
        plot_line(np.arange(len(images_sorted)), cover_vals,
                  title="Coverage (union of masks per frame)",
                  xlabel="frame index (sorted by filename)", ylabel="coverage ratio",
                  out_png=os.path.join(out_dir, "coverage_curve.png"))
    else:
        warnings.warn("找不到 mask_root，覆盖度与故事板将跳过。")

    # (5) baselines table (means of selected frames)
    rows = []
    def mean_of(vals, idxs):
        if vals is None: return None
        sel = [vals[i] for i in idxs]
        return float(np.mean(sel)) if len(sel)>0 else None

    N = len(images_sorted)
    # uniform
    for K in topk_list:
        sel = [images_sorted.index(fn) for fn in evenly_pick(images_sorted, K)]
        rows.append({
            "baseline":"Uniform","K":K,
            "mean_coverage": mean_of(cover_vals, sel),
            "mean_sharpness": mean_of(sharp_vals, sel)
        })
    # coverage-only
    if cover_vals is not None:
        rank_cov = np.argsort(-cover_vals)
        for K in topk_list:
            sel = rank_cov[:K].tolist()
            rows.append({
                "baseline":"CoverageOnly","K":K,
                "mean_coverage": mean_of(cover_vals, sel),
                "mean_sharpness": mean_of(sharp_vals, sel)
            })
    # sharpness-only
    if sharp_vals is not None:
        rank_sh = np.argsort(-sharp_vals)
        for K in topk_list:
            sel = rank_sh[:K].tolist()
            rows.append({
                "baseline":"SharpnessOnly","K":K,
                "mean_coverage": mean_of(cover_vals, sel),
                "mean_sharpness": mean_of(sharp_vals, sel)
            })
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(tables_dir,"baselines.csv"), index=False)

    # (6) storyboard of top-N by coverage
    if cover_vals is not None and len(images_sorted)>0:
        rank_cov = np.argsort(-cover_vals)
        pick = [images_sorted[i] for i in rank_cov[:storyboard_n]]
        df_map = df.groupby("image")["mask"].apply(lambda s: [str(x) for x in s]).to_dict()
        make_storyboard(frames_dir, mask_root, df_map, pick,
                        out_png=os.path.join(out_dir, "storyboard_topN.png"),
                        cols=4, alpha=160)

    print("[OK] figs ->", out_dir, "| tables ->", tables_dir)

if __name__ == "__main__":
    main()
