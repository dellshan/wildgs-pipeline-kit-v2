#!/usr/bin/env python3
import os, glob, json, argparse, torch, open_clip, pandas as pd
from PIL import Image
ap=argparse.ArgumentParser()
ap.add_argument("--frames_dir",required=True); ap.add_argument("--out",default="clip_results.json")
ap.add_argument("--csv",default=None); ap.add_argument("--model",default="ViT-H-14"); ap.add_argument("--pretrained",default="laion2b_s32b_b79k")
args=ap.parse_args()
device="cuda" if torch.cuda.is_available() else "cpu"
model,_,pre=open_clip.create_model_and_transforms(args.model,pretrained=args.pretrained); model=model.to(device).eval()
tok=open_clip.get_tokenizer(args.model)
labels={"environment":["indoor","outdoor city","suburban","countryside","industrial"],
        "time_of_day":["morning","midday","afternoon","golden hour","night"],
        "weather":["clear","cloudy","overcast","rainy","foggy"],
        "style":["photorealistic","cinematic","minimalist","vibrant","muted"]}
@torch.no_grad()
def best(imgf, opts):
    t=tok(opts).to(device); tf=model.encode_text(t); tf=tf/tf.norm(dim=-1,keepdim=True)
    s=(imgf@tf.T).squeeze(0); return opts[int(s.argmax().item())]
out=[]; imgs=sorted(sum([glob.glob(os.path.join(args.frames_dir,e)) for e in ("*.png","*.jpg","*.jpeg")],[]))
for p in imgs:
    im=pre(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        f=model.encode_image(im); f=f/f.norm(dim=-1,keepdim=True)
    tags={k:best(f,v) for k,v in labels.items()}
    out.append({"image":os.path.basename(p),"tags":tags})
with open(args.out,"w") as f: json.dump(out,f,indent=2); print("Wrote",args.out)
if args.csv:
    rows=[dict(image=r["image"],**r["tags"]) for r in out]; pd.DataFrame(rows).to_csv(args.csv,index=False); print("Wrote",args.csv)
