import os
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def load_font(path: str | None, size: int = 18):
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def text_wh(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    # Prefer Pillow >=10 API
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        # Fallbacks for older Pillow
        try:
            return font.getsize(text)
        except Exception:
            l, t, r, b = font.getbbox(text)
            return r - l, b - t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ROOT", "."), help="repo root")
    ap.add_argument("--frames", default="frames_views.csv")
    ap.add_argument("--objects", default="objects_views.csv")
    ap.add_argument("--view-id", type=int, default=1)
    ap.add_argument("--out", default="vis_seq_view1_lowconf")
    ap.add_argument("--font", default=None, help="path to .ttf (optional)")
    ap.add_argument("--thickness", type=int, default=3)
    ap.add_argument("--start", type=int, default=None, help="start frame index (0-based in sequence order)")
    ap.add_argument("--end", type=int, default=None, help="end frame index (exclusive)")
    ap.add_argument("--masks-dir", default=None, help="optional dir of per-frame masks named 0000.png,0001.png...")
    args = ap.parse_args()

    ROOT = Path(args.root).resolve()
    FV = pd.read_csv(ROOT / args.frames)
    OV = pd.read_csv(ROOT / args.objects)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(args.font, 18)

    # Frame order for this view
    idxs = sorted(FV[FV.view_id == args.view_id].idx.unique())
    if args.start is not None or args.end is not None:
        s = 0 if args.start is None else args.start
        e = len(idxs) if args.end is None else args.end
        idxs = idxs[s:e]

    n_written = 0
    for k, idx in enumerate(idxs):
        fr = FV[(FV.idx == idx) & (FV.view_id == args.view_id)]
        if fr.empty:
            continue
        r = fr.iloc[0]
        img = Image.open(r.frame_path).convert("RGB")
        d = ImageDraw.Draw(img)

        # optional mask overlay
        if args.masks_dir:
            mp = Path(args.masks_dir) / f"{int(idx):04d}.png"
            if mp.is_file():
                m = Image.open(mp).convert("L")
                overlay = Image.new("RGBA", img.size, (0, 255, 0, 80))
                img = Image.composite(overlay, img.convert("RGBA"), m).convert("RGB")
                d = ImageDraw.Draw(img)

        boxes = OV[(OV.idx == idx) & (OV.view_id == args.view_id)]
        for _, o in boxes.iterrows():
            x, y, w, h = map(int, [o.bbox_x, o.bbox_y, o.bbox_w, o.bbox_h])
            d.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=args.thickness)

            name = None
            if "class_name" in o and pd.notna(o["class_name"]):
                name = str(o["class_name"])
            elif "class_id" in o and pd.notna(o["class_id"]):
                name = f"cls{int(o['class_id'])}"
            else:
                name = "cls?"

            conf = float(o["conf"]) if "conf" in o and pd.notna(o["conf"]) else 1.0
            label = f"{name} {conf:.2f}"

            tw, th = text_wh(d, label, font)
            y0 = max(0, y - th - 2)
            d.rectangle([x, y0, x + tw + 6, y0 + th + 2], fill=(0, 255, 0))
            d.text((x + 3, y0 + 1), label, fill=(0, 0, 0), font=font)

        out_path = out_dir / f"frame_{k:04d}.png"
        img.save(out_path)
        n_written += 1

    print(f"wrote {n_written} frames -> {out_dir}")

if __name__ == "__main__":
    main()
