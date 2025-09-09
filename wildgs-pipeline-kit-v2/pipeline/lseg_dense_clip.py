# pipeline/lseg_dense_clip.py
import os, sys, json, math, argparse, glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

ZH2EN = {
    "桌子":"table","杂物":"clutter","显示器":"monitor","三脚架":"tripod",
    "电脑":"computer","瓶子":"bottle","白板":"whiteboard","相机":"camera",
    "笔记本电脑":"laptop","地面":"floor","墙":"wall","窗户":"window","门":"door",
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="关键帧目录")
    ap.add_argument("--prompts", nargs="+", default=["table","monitor","tripod","computer","camera","bottle","laptop","whiteboard","clutter"])
    ap.add_argument("--lang", default="en", choices=["en","zh"])
    ap.add_argument("--out_dir", default="output/lseg_dense")
    ap.add_argument("--backbone", default="clip", choices=["clip","declip"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.25, help="掩码阈值（相对 softmax）")
    ap.add_argument("--topk_keep", type=int, default=1, help="每像素保留前K类，默认取argmax")
    ap.add_argument("--describe", action="store_true", help="生成一段中文描述")
    return ap.parse_args()

def translate_prompts_if_needed(prompts, lang):
    if lang == "en": 
        return prompts
    out = []
    for p in prompts:
        if p in ZH2EN: out.append(ZH2EN[p])
        else: out.append(p)  # 未知词直接原样（建议自己补齐字典或改用英文）
    return out

def load_clip(device):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval().to(device)
    return model, proc

@torch.no_grad()
def dense_clip_logits_for_image(model, proc, image_pil, text_list, device):
    # 1) 文本特征（投影后）
    inputs_text = proc(text=text_list, return_tensors="pt", padding=True, truncation=True).to(device)
    text_feat = model.get_text_features(**inputs_text)                 # [C, D]
    text_feat = F.normalize(text_feat, dim=-1)

    # 2) 视觉 patch 特征（未池化的 token）
    inputs_img = proc(images=image_pil, return_tensors="pt").to(device)
    vout = model.vision_model(pixel_values=inputs_img["pixel_values"], output_hidden_states=True)
    tokens = vout.last_hidden_state    # [1, 1+N, D_v]
    patch = tokens[:,1:,:]             # 去掉 CLS
    # 维度恢复为 [1, H_p, W_p, D_v]
    H_p = W_p = int(math.sqrt(patch.shape[1]))
    patch = patch.reshape(1, H_p, W_p, -1).contiguous()                # [1,H_p,W_p,D_v]

    # 3) 视觉投影到共享空间（用 CLIP 的 visual_projection）
    # 视觉投影是线性层：D_v -> D
    W = model.visual_projection.weight    # [D, D_v]
    b = model.visual_projection.bias      # [D]
    # 手工仿射：[..., D_v] -> [..., D]
    vproj = torch.einsum("bhwd,md->bhwm", patch, W) + b                # [1,H_p,W_p,D]
    vproj = F.normalize(vproj, dim=-1)

    # 4) 点乘得到每类的 patch 级 logits
    # text_feat: [C,D] -> [1,1,1,C,D] 方便广播
    tf = text_feat.unsqueeze(0).unsqueeze(0).unsqueeze(0)              # [1,1,1,C,D]
    logits = torch.einsum("bhwd,bhwcd->bhwc", vproj, tf)               # [1,H_p,W_p,C]
    return logits.squeeze(0)  # [H_p, W_p, C]

def upsample_to_image(logits_hw_c, target_hw):
    # 双线性上采样到原图分辨率
    x = logits_hw_c.permute(2,0,1).unsqueeze(0)  # [1,C,H_p,W_p]
    x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
    return x.squeeze(0).permute(1,2,0)          # [H,W,C]

def masks_from_logits(logits_hwc, threshold=0.25, topk_keep=1):
    # softmax over classes
    probs = F.softmax(logits_hwc.float(), dim=-1).cpu().numpy()  # [H,W,C]
    H,W,C = probs.shape
    if topk_keep <= 1:
        lab = probs.argmax(-1)                     # [H,W]
        conf= probs.max(-1)
        masks = []
        for c in range(C):
            m = (lab==c) & (conf>=threshold)
            masks.append(m.astype(np.uint8)*255)
        return masks, conf
    else:
        # 每像素保留 topk
        topk = np.argpartition(-probs, kth=topk_keep-1, axis=-1)[..., :topk_keep]
        confk= np.take_along_axis(probs, topk, axis=-1).max(-1)
        masks=[]
        for c in range(C):
            m = (topk==c).any(-1) & (confk>=threshold)
            masks.append(m.astype(np.uint8)*255)
        return masks, confk

def relation_on_table(mask_table, mask_obj):
    # 简单 2D “on” 判定：obj 与 table 接触，且 obj 的质心在 table 的上缘附近
    import cv2
    if mask_table.sum()==0 or mask_obj.sum()==0: return False
    t = (mask_table>0).astype(np.uint8); o=(mask_obj>0).astype(np.uint8)
    # 膨胀 table 作为接触带
    ker = np.ones((7,7),np.uint8); t_border=cv2.dilate(t,ker,iterations=1)
    touch = (t_border & o).sum()>50
    # 质心与上缘
    ys,xs = np.where(o>0)
    if len(ys)==0: return False
    cy = ys.mean(); top_edge = np.where(t>0)[0].min() if (t>0).any() else cy
    above = cy < (top_edge + 0.15*mask_table.shape[0])  # 距离上缘不远
    return bool(touch and above)

def summarize_scene_cn(classes, areas, rels):
    # classes: [str], areas: dict{name:pixel_count}, rels: list[str]
    total = sum(areas.values())+1e-6
    top = sorted(classes, key=lambda k: -areas.get(k,0))[:5]
    comp = "、".join([f"{c}" for c in top if areas.get(c,0)>0])
    reltxt = "；".join(rels) if rels else ""
    sent_main = f"场景包含主要物体：{comp}。"
    if reltxt:
        return sent_main + f" 观察到空间关系：{reltxt}。"
    return sent_main

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    frames = sorted([p for p in glob.glob(os.path.join(args.frames_dir,"*")) if p.lower().endswith((".jpg",".png",".jpeg"))])
    if len(frames)==0:
        print("[ERR] no frames found"); sys.exit(1)

    prompts = translate_prompts_if_needed(args.prompts, args.lang)
    print(f"[LSeg-like] prompts: {prompts}")

    if args.backbone=="clip":
        model, proc = load_clip(args.device)
    else:
        raise RuntimeError("backbone=declip 尚未接入：请安装官方 DeCLIP 并在此处替换 dense_clip_logits_for_image() 以返回 [H_p,W_p,C] logits。")

    all_stats=[]
    for img_path in frames:
        img = Image.open(img_path).convert("RGB")
        W,H = img.size
        with torch.no_grad():
            logits_p = dense_clip_logits_for_image(model, proc, img, prompts, args.device)  # [H_p,W_p,C]
            logits = upsample_to_image(logits_p, (H,W))  # [H,W,C]
        masks, conf = masks_from_logits(logits, threshold=args.threshold, topk_keep=args.topk_keep)

        # 保存掩码
        base = os.path.splitext(os.path.basename(img_path))[0]
        per_area={}
        for i,(m,name) in enumerate(zip(masks, prompts)):
            outp = os.path.join(args.out_dir, f"{base}_{name}.png")
            Image.fromarray(m).save(outp)
            per_area[name]=int((m>0).sum())

        # 关系检测：桌子上的“杂物”
        rels=[]
        if "table" in prompts:
            tidx = prompts.index("table")
            if "clutter" in prompts:
                cidx = prompts.index("clutter")
                if relation_on_table(masks[tidx], masks[cidx]):
                    rels.append("桌子上存在若干杂物")
            # 可再对若干小物件类求并集判断 on_table
            smalls=[n for n in ["bottle","laptop","camera"] if n in prompts]
            if smalls:
                import cv2
                union = np.zeros_like(masks[tidx], dtype=np.uint8)
                for n in smalls:
                    union |= (masks[prompts.index(n)]>0).astype(np.uint8)*255
                if relation_on_table(masks[tidx], union):
                    rels.append("桌面上分布若干小物件")

        # 描述
        desc = summarize_scene_cn(prompts, per_area, rels) if args.describe else ""
        all_stats.append({"image":img_path,"areas":per_area,"relations":rels,"desc":desc})

        # 文本保存
        with open(os.path.join(args.out_dir,f"{base}_desc.txt"),"w",encoding="utf-8") as f:
            f.write(desc if desc else " ")

    with open(os.path.join(args.out_dir,"summary.json"),"w",encoding="utf-8") as f:
        json.dump(all_stats,f,ensure_ascii=False,indent=2)
    print(f"[OK] wrote masks + summary to {args.out_dir}")

if __name__=="__main__":
    main()
