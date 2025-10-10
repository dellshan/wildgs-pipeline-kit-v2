#!/usr/bin/env python3
import os, os.path as op, argparse, csv, json, glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

DEF_LABELS_EN = [
    "person","chair","table","desk","door","window","cabinet","shelf","sofa",
    "bed","monitor","keyboard","mouse","laptop","tv","picture","whiteboard",
    "bag","box","trash bin","ladder","bottle","cup","sink","toilet","bathtub",
    "stove","microwave","refrigerator","plant","vase","kettle","backpack",
    "bench","bicycle","lamp","fan","printer","book","cart","bucket","towel",
]

def read_labels(txt_path, lang):
    if txt_path and op.isfile(txt_path):
        labs = [ln.strip() for ln in open(txt_path) if ln.strip()]
        labs = [l for l in labs if l and not l.startswith("#")]
        return labs
    # default English list
    return DEF_LABELS_EN

def tight_bbox(mask_arr):
    ys, xs = np.where(mask_arr > 127)
    if xs.size == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def load_instances(inst_csv, masks_dir):
    rows = []
    if op.isfile(inst_csv):
        for i, r in enumerate(csv.DictReader(open(inst_csv))):
            mpath = op.join(masks_dir, r["mask"])
            if op.isfile(mpath):
                rows.append({
                    "image": r["image"],
                    "inst_id": int(r["inst_id"]),
                    "mask": r["mask"],
                })
        return rows
    # fallback: enumerate all *_instXXXX.png
    for p in sorted(glob.glob(op.join(masks_dir, "*_inst*.png"))):
        stem = op.basename(p).split("_inst")[0]
        inst = int(op.splitext(op.basename(p))[0].split("_inst")[-1])
        rows.append({"image": stem, "inst_id": inst, "mask": op.basename(p)})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks",  required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--labels_txt", default="")
    ap.add_argument("--lang", default="en", choices=["en"])
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    labels = read_labels(args.labels_txt, args.lang)
    text_prompts = [f"a {l}" for l in labels]

    inst_csv = op.join(op.dirname(args.masks), "instances.csv")
    items = load_instances(inst_csv, args.masks)
    if not items:
        print("[WARN] no instances found; nothing to do.")
        # still write header
        with open(args.out_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["image","inst_id","mask","pred_label","score","topk_labels","topk_scores"]
            )
        return

    out_rows = [("image","inst_id","mask","pred_label","score","topk_labels","topk_scores")]
    for it in tqdm(items, desc="CLIP name"):
        # locate image file
        stem = it["image"]
        imgp = None
        for ext in (".jpg",".JPG",".png",".PNG",".jpeg",".JPEG"):
            cand = op.join(args.images, stem+ext)
            if op.isfile(cand):
                imgp = cand; break
        if imgp is None:
            # recursive fallback
            matches = glob.glob(op.join(args.images, "**", stem+".*"), recursive=True)
            imgp = matches[0] if matches else None
        if imgp is None or not op.isfile(imgp):
            continue

        # crop by mask bbox
        mpath = op.join(args.masks, it["mask"])
        try:
            mask = Image.open(mpath).convert("L")
            m = np.array(mask)
            bb = tight_bbox(m)
            if bb is None:
                # whole image fallback
                crop = Image.open(imgp).convert("RGB")
            else:
                x0,y0,x1,y1 = bb
                crop = Image.open(imgp).convert("RGB").crop((x0,y0,x1+1,y1+1))
        except Exception:
            crop = Image.open(imgp).convert("RGB")

        with torch.inference_mode():
            inputs = processor(text=text_prompts, images=crop, return_tensors="pt", padding=True).to(device)
            out = model(**inputs)
            # logits_per_image: [1, num_text]
            logits = out.logits_per_image[0].float().cpu()
            probs = logits.softmax(dim=-1).numpy()
            topk = min(args.topk, len(labels))
            idx = np.argsort(-probs)[:topk]
            top_labels = [labels[i] for i in idx]
            top_scores = [float(probs[i]) for i in idx]
            pred_label, pred_score = top_labels[0], top_scores[0]

        out_rows.append((
            stem, it["inst_id"], it["mask"], pred_label, f"{pred_score:.4f}",
            "|".join(top_labels), "|".join(f"{s:.4f}" for s in top_scores)
        ))

    os.makedirs(op.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)
    print(f"[OK] wrote {args.out_csv} | objects= {len(out_rows)-1}")

if __name__ == "__main__":
    main()
