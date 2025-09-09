#!/usr/bin/env python3
import os, glob, json, csv, argparse, collections
from PIL import Image
import torch, open_clip

CANDIDATES = {
    "environment": ["indoor room","office","lab","hallway","classroom","meeting room",
                    "outdoor street","plaza","park","subway station","industrial site"],
    "people": ["no people","one person","two people","a small group","a crowd"],
    "activity": ["standing","walking","sitting","talking","using a computer",
                 "presenting","looking at a screen","reading","writing","carrying a bag"],
    "objects": ["chair","table","desk","screen","monitor","whiteboard","door","window",
                "computer","laptop","backpack","bottle","umbrella","tripod","camera"],
    "style": ["photorealistic","cinematic","vibrant","muted","natural lighting",
              "fluorescent lighting","warm lighting","cool lighting"],
    "time": ["morning","afternoon","evening","night","indoor lighting"]
}

def rank_text(model, tok, device, image_feat, texts):
    with torch.no_grad():
        t = tok(texts).to(device)
        tf = model.encode_text(t)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        sim = (image_feat @ tf.T).squeeze(0)  # [N]
        top_idx = torch.argsort(sim, descending=True)
    return [(texts[i], float(sim[i])) for i in top_idx]

def sentence_from_tags(tags, lang="en"):
    env = tags.get("environment")
    ppl = tags.get("people")
    act = tags.get("activity")
    objs = [o for o in tags.get("objects", [])][:3]
    style = tags.get("style")
    time = tags.get("time")

    if lang == "zh":
        parts = []
        if env: parts.append(f"场景：{env}")
        if time: parts.append(f"时间/光线：{time}")
        if ppl: parts.append(f"人群：{ppl}")
        if act: parts.append(f"活动：{act}")
        if objs: parts.append("物体：" + "、".join(objs))
        if style: parts.append(f"风格：{style}")
        return "；".join(parts)
    else:
        segs = []
        lead = []
        if env: lead.append(env)
        if time: lead.append(time)
        if style: lead.append(style)
        if lead: segs.append(", ".join(lead))
        if ppl: segs.append(ppl)
        if act: segs.append(act)
        if objs: segs.append("objects: " + ", ".join(objs))
        return "; ".join(segs) if segs else "scene"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_json", default="output/clip_descriptions.json")
    ap.add_argument("--out_csv",  default="output/clip_descriptions.csv")
    ap.add_argument("--srt", default=None, help="optional .srt path (ordered by filename)")
    ap.add_argument("--lang", default="en", choices=["en","zh"])
    ap.add_argument("--model", default="ViT-H-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b79k")
    ap.add_argument("--topk_objects", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tok = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.*")))
    if not paths:
        raise SystemExit(f"no images in {args.frames_dir}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    all_rows = []
    agg_counts = collections.defaultdict(collections.Counter)

    for p in paths:
        im = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            imf = model.encode_image(im)
            imf = imf / imf.norm(dim=-1, keepdim=True)

        # per-category best
        tags = {}
        for cat, opts in CANDIDATES.items():
            ranked = rank_text(model, tok, device, imf, opts)
            if cat == "objects":
                tags[cat] = [t for t,_ in ranked[:args.topk_objects]]
                for t,_ in ranked[:args.topk_objects]:
                    agg_counts[cat][t]+=1
            else:
                best = ranked[0][0]
                tags[cat] = best
                agg_counts[cat][best]+=1

        caption = sentence_from_tags(tags, lang=args.lang)
        row = {
            "image": os.path.basename(p),
            "caption": caption,
            **{f"{k}": (", ".join(v) if isinstance(v, list) else v) for k,v in tags.items()},
        }
        all_rows.append(row)

    # write json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    # write csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows: w.writerow(r)

    # optional SRT (1..N, naive 2s per frame)
    if args.srt:
        with open(args.srt, "w", encoding="utf-8") as f:
            for i, r in enumerate(all_rows, 1):
                t0 = 2*(i-1); t1 = 2*i
                hhmmss = lambda t: f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d},000"
                f.write(f"{i}\n{hhmmss(t0)} --> {hhmmss(t1)}\n{r['caption']}\n\n")

        # simple majority summary (修复 objects 被当作字符串逐字符切分的问题)
    summary = {}
    for cat, ctr in agg_counts.items():
        common = ", ".join([f"{k}({v})" for k,v in ctr.most_common(3)])
        summary[cat] = common

    # 为 scene_summary 传入 objects 的“前三名列表”，其余类别传入单个最佳标签
    objects_top3 = [k for k, _ in agg_counts["objects"].most_common(3)]
    summary_tags = {
        "environment": (agg_counts["environment"].most_common(1)[0][0] if agg_counts["environment"] else None),
        "people":      (agg_counts["people"].most_common(1)[0][0]      if agg_counts["people"]      else None),
        "activity":    (agg_counts["activity"].most_common(1)[0][0]    if agg_counts["activity"]    else None),
        "style":       (agg_counts["style"].most_common(1)[0][0]       if agg_counts["style"]       else None),
        "time":        (agg_counts["time"].most_common(1)[0][0]        if agg_counts["time"]        else None),
        "objects":     objects_top3,
    }
    scene_summary = sentence_from_tags(summary_tags, lang=args.lang)

    print(f"Wrote {args.out_json} and {args.out_csv}")
    print("Summary:", scene_summary)
