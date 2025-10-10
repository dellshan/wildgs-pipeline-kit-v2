#!/usr/bin/env python3
import os, json, math, argparse, collections, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams

# Try a CJK-capable font if available
for name in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "Arial Unicode MS"]:
    if name in {f.name for f in fm.fontManager.ttflist}:
        rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        break

def load_json(p): return json.load(open(p, "r", encoding="utf-8"))
def ensure_dir(d): os.makedirs(d, exist_ok=True)
def savefig(fig, out_base):
    for ext in (".png", ".pdf"):
        fig.savefig(out_base + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------- utilities ----------
def yaw_delta_series(links):
    yaws = [float(l.get("yaw_deg", 0.0)) for l in links]
    if len(yaws) < 2: return np.array([])
    dy = np.diff(yaws)
    # wrap to [-180, 180]
    dy = (dy + 180.0) % 360.0 - 180.0
    return np.abs(dy)

def risk_per_frame(obst):
    out = []
    for fr in obst.get("frames", []):
        total = sum(float(o.get("score", 0.0)) for o in fr.get("obstacles", []))
        out.append((fr.get("image"), total))
    return out  # list of (image, risk)

def object_freq(story):
    cnt = collections.Counter()
    for fr in story.get("frames", []):
        w = 1.0 + float(fr.get("coverage", 0.0))
        for o in fr.get("objects", []):
            n = (o.get("name") or o.get("label") or o.get("category") or "object").strip().lower()
            if n: cnt[n] += w
    return cnt

def object_risk(obst):
    # aggregate risk per object category
    agg = collections.Counter()
    for fr in obst.get("frames", []):
        for o in fr.get("obstacles", []):
            n = str(o.get("name","object")).strip().lower()
            agg[n] += float(o.get("score",0.0))
    return agg

def cooccurrence(story, topk=20):
    # frame-level co-occurrence of object names (unique set per frame)
    freq = object_freq(story)
    vocab = [k for k,_ in freq.most_common(topk)]
    idx = {v:i for i,v in enumerate(vocab)}
    M = np.zeros((len(vocab), len(vocab)), dtype=float)
    pairs = collections.Counter()
    for fr in story.get("frames", []):
        S = sorted({(o.get("name") or o.get("label") or o.get("category") or "object").strip().lower()
                    for o in fr.get("objects", []) if (o.get("name") or o.get("label") or o.get("category"))})
        for a,b in itertools.combinations(S, 2):
            if a in idx and b in idx:
                i, j = idx[a], idx[b]
                M[i,j] += 1; M[j,i] += 1
                pairs[(a,b)] += 1
    return vocab, M, pairs

# ---------- new figures ----------
def fig_turn_hist(links, out_dir, bins=12):
    dy = yaw_delta_series(links)
    fig, ax = plt.subplots()
    ax.hist(dy, bins=bins)
    ax.set_xlabel("|Δyaw| (deg) between consecutive links")
    ax.set_ylabel("Count")
    ax.set_title("Turn-angle histogram (path smoothness)")
    savefig(fig, os.path.join(out_dir, "fig_turn_hist"))

def fig_risk_timeline(obst, story, out_dir):
    rp = risk_per_frame(obst)
    if not rp: return
    names, vals = zip(*rp)
    x = np.arange(len(vals))
    cum = np.cumsum(vals)
    fig, ax = plt.subplots()
    ax.plot(x, vals, marker="o")
    ax2 = ax.twinx()
    ax2.plot(x, cum, linestyle="--")
    ax.set_xlabel("View index")
    ax.set_ylabel("Risk per frame")
    ax2.set_ylabel("Cumulative risk")
    ax.set_title("Risk timeline")
    savefig(fig, os.path.join(out_dir, "fig_risk_timeline"))

def fig_coocc_heatmap(story, out_dir, topk=15):
    vocab, M, _ = cooccurrence(story, topk=topk)
    if len(vocab) == 0: return
    fig, ax = plt.subplots()
    im = ax.imshow(M, origin="lower")
    ax.set_xticks(range(len(vocab))); ax.set_xticklabels(vocab, rotation=90)
    ax.set_yticks(range(len(vocab))); ax.set_yticklabels(vocab)
    ax.set_title(f"Object co-occurrence heatmap (top-{topk})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, os.path.join(out_dir, "fig_coocc_heatmap"))

def fig_freq_vs_risk_scatter(story, obst, out_dir, topk=30):
    freq = object_freq(story)
    risk = object_risk(obst)
    keys = [k for k,_ in freq.most_common(topk)]
    x = np.array([freq[k] for k in keys], float)
    y = np.array([risk.get(k, 0.0) for k in keys], float)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for xi, yi, label in zip(x, y, keys):
        if yi > 0.02 * (y.max() if y.max() > 0 else 1):  # label salient
            ax.annotate(label, (xi, yi), xytext=(3,3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Weighted frequency")
    ax.set_ylabel("Aggregated risk score")
    ax.set_title("Object frequency vs. risk")
    savefig(fig, os.path.join(out_dir, "fig_freq_vs_risk_scatter"))

def fig_cov_vs_risk_scatter(story, obst, out_dir):
    # per-frame coverage vs. risk
    cov = [float(fr.get("coverage",0.0)) for fr in story.get("frames",[])]
    risk_map = dict(risk_per_frame(obst))
    r = [risk_map.get(fr.get("image"), 0.0) for fr in story.get("frames",[])]
    if not cov: return
    fig, ax = plt.subplots()
    ax.scatter(cov, r)
    ax.set_xlabel("Coverage per view")
    ax.set_ylabel("Risk per view")
    ax.set_title("Coverage vs. risk (per view)")
    savefig(fig, os.path.join(out_dir, "fig_cov_vs_risk_scatter"))

# ---------- tables (CSV + LaTeX) ----------

def to_latex_table(rows, headers, out_tex, caption="", label="tab:summary"):
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\\n\\centering\\n")
        f.write("\\begin{tabular}{" + "l" * len(headers) + "}\\n\\hline\\n")
        f.write(" & ".join(headers) + " \\\\ \\n\\hline\\n")
        for r in rows:
            f.write(" & ".join(map(str, r)) + " \\\\ \\n")
        f.write("\\hline\\n\\end{tabular}\\n")
        f.write("\\caption{%s}\\n\\label{%s}\\n\\end{table}\\n" % (caption, label))
