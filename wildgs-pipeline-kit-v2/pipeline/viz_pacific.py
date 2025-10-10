#!/usr/bin/env python3
"""
PacificVis figure pack generator.

Reads:
  --story      JSON with frames (e.g., output/view_story_B_2d/story.with_names.norm.json)
  --links      JSON with {"links":[{dist_m, yaw_deg, ...}, ...]}
  --obstacles  JSON with {"frames":[{"image":..., "obstacles":[{"name","score","area_frac"}, ...]}]}
Writes to --out_dir (default: output/pacificvis_figs):
  - fig_distance_per_link.{png,pdf}
  - fig_cumulative_distance.{png,pdf}
  - fig_yaw_hist.{png,pdf}
  - fig_yaw_rose.{png,pdf}
  - fig_coverage_per_view.{png,pdf}
  - fig_top_objects.{png,pdf}
  - fig_top_obstacles.{png,pdf}
  - summary_stats.csv
  - top_objects.csv
  - top_obstacles.csv

Constraints satisfied:
  1) matplotlib only (no seaborn)
  2) one chart per figure
  3) no explicit colors/styles set
"""

import os, json, math, argparse, collections
import numpy as np

# Headless first — must be set before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams

# ---------- Font setup (CJK-friendly if available) ----------
def configure_fonts():
    """
    Try to find a CJK-capable font so Chinese/Japanese/Korean glyphs render.
    Falls back to DejaVu Sans if not available.
    """
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Arial Unicode MS",
    ]
    # add a local bundled font if user placed one here
    local_font_files = [
        "./fonts/NotoSansCJKsc-Regular.otf",
        "./fonts/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in local_font_files:
        if os.path.isfile(fp):
            try:
                fm.fontManager.addfont(fp)
                # The common family name after addfont for Noto CJK
                candidates.insert(0, "Noto Sans CJK SC")
                break
            except Exception:
                pass

    avail = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in avail:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            return
    # fallback
    rcParams["font.sans-serif"] = ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


# ---------- IO helpers ----------
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ---------- Plotting helpers (no seaborn; single-figure; default colors) ----------
FIGSIZE = (4.0, 3.0)

def savefig(fig, out_base):
    for ext in (".png", ".pdf"):
        fig.savefig(out_base + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- Figures ----------
def fig_distance_per_link(links, out_dir):
    dists = [float(l.get("dist_m", 0.0)) for l in links]
    x = np.arange(len(dists))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, dists, marker="o")
    ax.set_xlabel("Link index")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Per-link distance")
    savefig(fig, os.path.join(out_dir, "fig_distance_per_link"))

def fig_cumulative_distance(links, out_dir):
    dists = [float(l.get("dist_m", 0.0)) for l in links]
    cum = np.cumsum(dists)
    x = np.arange(len(cum))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, cum, marker="o")
    ax.set_xlabel("Link index")
    ax.set_ylabel("Cumulative distance (m)")
    ax.set_title("Cumulative path length")
    savefig(fig, os.path.join(out_dir, "fig_cumulative_distance"))

def fig_yaw_hist(links, out_dir, bins=12):
    yaw = [float(l.get("yaw_deg", 0.0)) for l in links]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(yaw, bins=bins)
    ax.set_xlabel("Yaw (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Yaw distribution")
    savefig(fig, os.path.join(out_dir, "fig_yaw_hist"))

def fig_yaw_rose(links, out_dir, bins=12):
    # polar histogram (degrees -> radians)
    yaw = np.array([float(l.get("yaw_deg", 0.0)) for l in links])
    yaw = np.deg2rad((yaw + 360.0) % 360.0)
    edges = np.linspace(0.0, 2*np.pi, bins+1)
    counts, _ = np.histogram(yaw, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="polar")
    # A more navigation-friendly polar setup
    ax.set_theta_zero_location("N")   # 0° at north (up)
    ax.set_theta_direction(-1)        # clockwise
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°","90°","180°","270°"])
    ax.bar(centers, counts, width=(2*np.pi/bins), align="center")
    ax.set_title("Yaw rose (polar histogram)")
    savefig(fig, os.path.join(out_dir, "fig_yaw_rose"))

def fig_coverage_per_view(story, out_dir):
    cov = [float(fr.get("coverage", 0.0)) for fr in story.get("frames", [])]
    x = np.arange(len(cov))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x, cov)
    ax.set_xlabel("View index")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage per view")
    savefig(fig, os.path.join(out_dir, "fig_coverage_per_view"))

def fig_top_objects(story, out_dir, topk=15):
    # frequency weighted lightly by coverage
    freq = collections.Counter()
    frames = story.get("frames", [])
    for fr in frames:
        w = 1.0 + float(fr.get("coverage", 0.0))
        for o in fr.get("objects", []):
            n = (o.get("name") or o.get("label") or o.get("category") or "object")
            n = str(n).strip().lower()
            if not n:
                continue
            freq[n] += w
    top = freq.most_common(topk)
    if not top:
        return
    names = [k for k,_ in top]
    vals  = [v for _,v in top]
    idx = np.arange(len(top))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(idx, vals)
    ax.set_yticks(idx)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted frequency")
    ax.set_title(f"Top-{topk} objects")
    savefig(fig, os.path.join(out_dir, "fig_top_objects"))

def fig_top_obstacles(obst, out_dir, topk=10):
    items = []
    for fr in obst.get("frames", []):
        img = fr.get("image")
        for o in fr.get("obstacles", []):
            items.append((f"{o.get('name','object')}", float(o.get("score", 0.0)), img))
    items.sort(key=lambda x: -x[1])
    items = items[:topk]
    if not items:
        return
    names = [f"{n} ({img})" for n,_,img in items]
    scores = [s for _,s,_ in items]
    idx = np.arange(len(items))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(idx, scores)
    ax.set_yticks(idx)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Risk score")
    ax.set_title(f"Top-{topk} obstacles")
    savefig(fig, os.path.join(out_dir, "fig_top_obstacles"))


# ---------- Tables / summaries ----------
def write_summary_csv(links, out_dir):
    import csv
    dists = [float(l.get("dist_m", 0.0)) for l in links]
    summary = {
        "num_links": len(dists),
        "mean_dist": float(np.mean(dists)) if dists else 0.0,
        "total_dist": float(np.sum(dists)) if dists else 0.0,
    }
    with open(os.path.join(out_dir, "summary_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader(); w.writerow(summary)

def write_tables(story, obstacles, out_dir):
    import csv
    # top objects
    freq = collections.Counter()
    for fr in story.get("frames", []):
        w = 1.0 + float(fr.get("coverage", 0.0))
        for o in fr.get("objects", []):
            n = (o.get("name") or o.get("label") or o.get("category") or "object")
            n = str(n).strip().lower()
            if n:
                freq[n] += w
    with open(os.path.join(out_dir, "top_objects.csv"), "w", newline="") as f:
        wcsv = csv.writer(f); wcsv.writerow(["object","weighted_freq"])
        for k,v in freq.most_common(30):
            wcsv.writerow([k, f"{v:.3f}"])

    # top obstacles
    obs = []
    for fr in obstacles.get("frames", []):
        for o in fr.get("obstacles", []):
            obs.append([
                fr.get("image"),
                o.get("name","object"),
                float(o.get("score",0.0)),
                float(o.get("area_frac",0.0)),
            ])
    obs.sort(key=lambda x: (-x[2], -x[3]))
    with open(os.path.join(out_dir, "top_obstacles.csv"), "w", newline="") as f:
        wcsv = csv.writer(f); wcsv.writerow(["image","name","score","area_frac"])
        for row in obs[:50]:
            wcsv.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}"])


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--story", required=True, help=".../output/view_story_B_2d/story.with_names.norm.json")
    parser.add_argument("--links", required=True, help=".../output/view_story_B_2d/links.json")
    parser.add_argument("--obstacles", required=True, help=".../output/view_story_B_2d/obstacles_recomputed.json")
    parser.add_argument("--out_dir", default="output/pacificvis_figs", help="where to write figures")
    args = parser.parse_args()

    configure_fonts()
    ensure_dir(args.out_dir)

    story = load_json(args.story)
    links = load_json(args.links).get("links", [])
    obstacles = load_json(args.obstacles)

    # figures
    fig_distance_per_link(links, args.out_dir)
    fig_cumulative_distance(links, args.out_dir)
    fig_yaw_hist(links, args.out_dir)
    fig_yaw_rose(links, args.out_dir)
    fig_coverage_per_view(story, args.out_dir)
    fig_top_objects(story, args.out_dir)
    fig_top_obstacles(obstacles, args.out_dir)

    # tables / summary
    write_summary_csv(links, args.out_dir)
    write_tables(story, obstacles, args.out_dir)

    print("[OK] PacificVis figure pack ->", args.out_dir)


if __name__ == "__main__":
    main()
