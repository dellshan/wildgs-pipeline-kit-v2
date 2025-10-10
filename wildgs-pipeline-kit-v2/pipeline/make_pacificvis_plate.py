#!/usr/bin/env python3
import os, glob, os.path as op
from PIL import Image, ImageDraw, ImageFont

def load(p): 
    return Image.open(p).convert("RGB")

def pick(fig_dir, name):
    g = sorted(glob.glob(op.join(fig_dir, f"{name}.png")))
    return g[0] if g else None

def main(fig_dir="output/pacificvis_figs", story_dir="output/view_story_B_2d", out="output/pacificvis_plate"):
    os.makedirs(op.dirname(out), exist_ok=True)
    # sources
    storyboard = op.join(story_dir, "storyboard_annot.png")
    charts = [
        pick(fig_dir, "fig_yaw_rose"),
        pick(fig_dir, "fig_cumulative_distance"),
        pick(fig_dir, "fig_distance_per_link"),
        pick(fig_dir, "fig_coverage_per_view"),
        pick(fig_dir, "fig_top_objects"),
        pick(fig_dir, "fig_top_obstacles"),
    ]
    charts = [c for c in charts if c and os.path.isfile(c)]
    imgs = [load(storyboard).resize((1280,720))] + [load(p).resize((1280,720)) for p in charts]

    # grid 3 cols
    W,H = 1280,720
    cols = 3
    rows = (len(imgs)+cols-1)//cols
    canvas = Image.new("RGB", (cols*W, rows*H), (255,255,255))
    for i,im in enumerate(imgs):
        r,c = divmod(i, cols)
        canvas.paste(im, (c*W, r*H))

    png = out + ".png"
    pdf = out + ".pdf"
    canvas.save(png); canvas.save(pdf)
    print("[OK] PacificVis plate ->", png, "|", pdf)

if __name__ == "__main__":
    main()
