#!/usr/bin/env python3
import os, os.path as op, argparse, csv, math
import torch, clip
from PIL import Image
import numpy as np
from tqdm import tqdm

DEF_LABELS = [
  "person","bicycle","car","motorcycle","bus","truck",
  "dog","cat","bird",
  "chair","couch","bed","dining table","toilet",
  "tv","laptop","keyboard","mouse","cell phone","book","bottle","cup",
  "potted plant","sink","refrigerator","microwave","oven","backpack","umbrella",
  "traffic cone","fire hydrant","bench","vase","scissors","remote","clock"
]

def load_masked_patch(img_path, mask_path, x1,y1,x2,y2):
    im = Image.open(img_path).convert("RGB")
    mw = Image.open(mask_path).convert("L")
    im = im.crop((x1,y1,x2+1,y2+1))
    mw = mw.crop((x1,y1,x2+1,y2+1))
    # apply mask: keep fg, black bg
    arr = np.array(im)
    m = (np.array(mw) > 0)[:, :, None]
    arr = arr * m
    return Image.fromarray(arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--instances_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="label words; defaults to a common list")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    labels = args.labels or DEF_LABELS
    device = args.device
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    with torch.no_grad():
        text = clip.tokenize([f"a photo of a {w}" for w in labels]).to(device)
        text_feats = model.encode_text(text)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    rows_out = [("image","inst_id","pred","score","x1","y1","x2","y2","mask_path")]
    with open(args.instances_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in tqdm(reader, desc="CLIP name"):
            img = op.join(args.images_root, r["image"])
            mask_path = op.join(op.dirname(args.instances_csv), r["mask_path"])
            x1,y1,x2,y2 = map(int, (r["x1"],r["y1"],r["x2"],r["y2"]))
            patch = load_masked_patch(img, mask_path, x1,y1,x2,y2)
            patch = preprocess(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                im_feat = model.encode_image(patch)
                im_feat = im_feat / im_feat.norm(dim=-1, keepdim=True)
                sims = (100.0 * im_feat @ text_feats.T).softmax(dim=-1).squeeze(0)
                score, idx = sims.max(dim=-1)
                pred = labels[int(idx)]
                rows_out.append((r["image"], r["inst_id"], pred, float(score), x1,y1,x2,y2, r["mask_path"]))

    os.makedirs(op.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(rows_out[0])
        for r in rows_out[1:]:
            w.writerow(r)
    print("[OK] wrote", args.out_csv, "| objects=", len(rows_out)-1)

if __name__ == "__main__":
    main()
