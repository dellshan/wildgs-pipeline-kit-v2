#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, argparse, glob, warnings, re
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# -------- CLIPSeg (always available) --------
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# -------- optional: SAM-2 (for instance masks) --------
_HAS_SAM2 = True
try:
    # Preferred (no YAML): from_pretrained(repo)
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    # Fallback (needs YAML+ckpt)
    from sam2.build_sam import build_sam2
except Exception as e:
    warnings.warn(f"[WARN] SAM-2 import failed: {e}")
    _HAS_SAM2 = False


# ---------- ZH→EN labels (extended) ----------
ZH2EN: Dict[str, str] = {
    # furniture / scene
    "桌子":"table","书桌":"desk","椅子":"chair","沙发":"sofa","柜子":"cabinet","架子":"shelf",
    "显示器":"monitor","屏幕":"screen","电脑":"computer","笔记本电脑":"laptop","键盘":"keyboard","鼠标":"mouse",
    "投影仪":"projector","麦克风":"microphone","相机":"camera","三脚架":"tripod","机械臂":"robot arm","传感器":"sensor",
    "打印机":"printer","插线板":"power strip","充电器":"charger","适配器":"adapter","电缆":"cable",
    # small items
    "瓶子":"bottle","杯子":"cup","水杯":"cup","马克杯":"mug","书":"book","笔记本":"notebook","纸张":"paper",
    "盒子":"box","包":"bag","背包":"backpack","手机":"phone","遥控器":"remote control","笔":"pen","马克笔":"marker",
    "胶带":"tape","剪刀":"scissors","工具箱":"toolbox",
    # people / scenes
    "人":"person","两个人":"two people","一小群人":"a small group",
    "实验室":"laboratory","办公室":"office","机房":"server room","会议室":"meeting room","车间":"workshop","仓库":"warehouse",
    # NEW objects you mentioned
    "篮球":"basketball","梯子":"ladder","垃圾桶":"trash can","垃圾箱":"trash bin","垃圾篓":"wastebasket",
    "塑料袋":"plastic bag","香蕉":"banana","纸箱":"cardboard box","纸箱子":"cardboard box","箱子":"box"
}

# Derive English sets
SMALL_ITEMS_ZH = {
    "杯子","水杯","马克杯","瓶子","书","笔记本","纸张","手机","遥控器","笔","马克笔",
    "胶带","剪刀","工具箱","盒子","包","背包","电缆","充电器","适配器","相机","三脚架","键盘","鼠标"
}
SMALL_ITEMS_EN = set(ZH2EN[z] for z in SMALL_ITEMS_ZH if z in ZH2EN)
TABLE_ZH = {"桌子","书桌"}
TABLE_EN = set(ZH2EN[z] for z in TABLE_ZH if z in ZH2EN)

# A few helpful English synonyms for CLIPSeg probing
EN_SYNONYMS: Dict[str, List[str]] = {
    "trash can": ["trash can", "garbage can", "trash bin", "waste bin", "wastebasket"],
    "plastic bag": ["plastic bag", "shopping bag"],
    "cardboard box": ["cardboard box", "carton", "box"],
    "desk": ["desk", "table"],  # help desk/table ambiguity
    "table": ["table", "desk"],
}


# ---------- helpers ----------
def timecode(t: float) -> str:
    ms = int(round(t * 1000.0)); s, ms = divmod(ms, 1000)
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def parse_csv_list(s: str) -> List[str]:
    if not s: return []
    return [x.strip() for x in s.split(",") if x.strip()]

def uniq_keeporder(xs: List[str]) -> List[str]:
    seen=set(); out=[]
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a * b).sum().item()
    uni   = a.sum().item() + b.sum().item() - inter + 1e-6
    return inter / uni

def overlap_ratio(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    inter = (mask_a * mask_b).sum().item()
    area_a = mask_a.sum().item() + 1e-6
    return inter / area_a

def is_english_text(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s))

def to_en(label: str) -> str:
    """Convert zh to en if known; otherwise return original."""
    return ZH2EN.get(label, label)

def _an(word: str) -> str:
    if not word: return "a"
    c = word.lstrip().lower()[:1]
    return "an" if c in "aeiou" else "a"

def join_en(xs: List[str], n: int = 3) -> str:
    xs = xs[:n]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]

def format_caption(lang: str, scene_zh_or_en: Optional[str],
                   table_items_en: List[str], visible_items_en: List[str],
                   has_people: bool) -> str:
    if lang == "en":
        parts: List[str] = []
        if scene_zh_or_en:
            scene_en = ZH2EN.get(scene_zh_or_en, scene_zh_or_en)
            parts.append(f"This looks like {_an(scene_en)} {scene_en}.")
        if has_people:
            parts.append("People are present.")
        if table_items_en:
            parts.append(f"On the desk: {join_en(table_items_en)}.")
        elif visible_items_en:
            parts.append(f"Visible: {join_en(visible_items_en)}.")
        return " ".join(parts) if parts else "This is an indoor scene."
    else:
        parts: List[str] = []
        if scene_zh_or_en:
            # If given EN, it’s fine to embed directly
            parts.append(f"这是一个{scene_zh_or_en}")
        if has_people:
            parts.append("有人活动")
        if table_items_en:
            parts.append(f"桌面上有{'、'.join(table_items_en[:3])}")
        elif visible_items_en:
            parts.append(f"可见{'、'.join(visible_items_en[:3])}")
        return "，".join(parts) if parts else "这是一个室内场景"


# ---------- CLIPSeg unified ----------
def load_clipseg(device: str):
    proc  = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
    return proc, model

@torch.no_grad()
def clipseg_prob(proc, model, img_pil: Image.Image, text: str, device: str) -> torch.Tensor:
    # Remove padding/truncation to avoid warnings from ViTImageProcessor
    inputs = proc(text=[text], images=[img_pil], return_tensors="pt")
    inputs = {k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}
    out = model(**inputs)
    logits = out.logits
    # normalize to [1,1,H,W]
    if logits.dim() == 4:
        if logits.size(1) > 1: logits = logits[:, 0:1]
    elif logits.dim() == 3:
        logits = logits.unsqueeze(1)
    elif logits.dim() == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected CLIPSeg logits dim={logits.dim()} shape={tuple(logits.shape)}")
    H, W = img_pil.size[1], img_pil.size[0]
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)  # [1,1,H,W]
    prob = torch.sigmoid(logits[0,0])
    return prob

@torch.no_grad()
def clipseg_prob_multi(proc, model, img_pil: Image.Image, queries: List[str], device: str) -> torch.Tensor:
    """Union (pixelwise max) of multiple textual queries."""
    probs = []
    for q in queries:
        probs.append(clipseg_prob(proc, model, img_pil, q, device))
    if not probs:
        H, W = img_pil.size[1], img_pil.size[0]
        return torch.zeros((H, W), dtype=torch.float32, device="cpu")
    return torch.stack(probs, dim=0).max(dim=0).values

@torch.no_grad()
def clipseg_mask(proc, model, img_pil: Image.Image, text: str, device: str, th: float=0.5) -> torch.Tensor:
    return (clipseg_prob(proc, model, img_pil, text, device) >= th).float()


# ---------- heatmap -> peaks ----------
def find_peaks_from_prob(prob: torch.Tensor, topk: int=5, min_val: float=0.35, min_dist: int=16) -> List[Tuple[int,int,float]]:
    H, W = prob.shape
    k = max(1, int(min_dist))
    pooled = F.max_pool2d(prob[None,None], kernel_size=2*k+1, stride=1, padding=k)[0,0]
    peaks = (prob >= min_val) & (prob == pooled)
    ys, xs = torch.where(peaks)
    if ys.numel() == 0:
        return []
    scores = prob[ys, xs]
    order = torch.argsort(scores, descending=True)
    xs = xs[order]; ys = ys[order]; scores = scores[order]
    pts = [(int(xs[i].item()), int(ys[i].item()), float(scores[i].item())) for i in range(min(topk, xs.numel()))]
    return pts


# ---------- SAM-2 predictor ----------
def load_sam2_predictor(
    device: str,
    sam2_repo: Optional[str] = None,
    sam2_model_yaml: Optional[str] = None,
    sam2_ckpt: Optional[str] = None,
):
    """
    Preferred: pass --sam2_repo like 'facebook/sam2.1-hiera-large' (no YAML needed).
    Fallback: pass both --sam2_model (yaml name) and --sam2_ckpt (local .pt).
    """
    if not _HAS_SAM2:
        raise RuntimeError("SAM-2 package not available.")
    if sam2_repo:
        return SAM2ImagePredictor.from_pretrained(sam2_repo)
    if not (sam2_model_yaml and sam2_ckpt):
        raise RuntimeError("Provide --sam2_repo (preferred), or both --sam2_model and --sam2_ckpt.")
    model = build_sam2(model_cfg=sam2_model_yaml, checkpoint=sam2_ckpt, device=device)
    return SAM2ImagePredictor(model)


@torch.no_grad()
def sam2_masks_from_points(predictor, img_np: np.ndarray, points: List[Tuple[int,int,float]]) -> List[torch.Tensor]:
    predictor.set_image(img_np)
    masks_all: List[torch.Tensor] = []
    for (x, y, _s) in points:
        m, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        if len(scores) == 0:
            continue
        i = int(np.argmax(scores))
        mm = torch.from_numpy(m[i].astype(np.uint8))  # [H,W] {0,1}
        masks_all.append(mm.float())
    return masks_all

def merge_masks_by_iou(masks: List[torch.Tensor], iou_th: float=0.6) -> List[torch.Tensor]:
    if not masks: return []
    areas = [m.sum().item() for m in masks]
    order = np.argsort(areas)[::-1].tolist()
    kept: List[torch.Tensor] = []
    for idx in order:
        m = masks[idx]
        merged = False
        for j, km in enumerate(kept):
            if iou(m, km) >= iou_th:
                kept[j] = torch.maximum(km, m)
                merged = True
                break
        if not merged:
            kept.append(m)
    return kept


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_srt", required=True)
    ap.add_argument("--fps", type=float, default=7.0)

    # common
    ap.add_argument("--mask_th", type=float, default=0.5)
    ap.add_argument("--on_table_ratio", type=float, default=0.20)
    ap.add_argument("--max_items", type=int, default=3)
    ap.add_argument("--suppress", type=str, default="")
    ap.add_argument("--mask_backend", choices=["clipseg","sam2"], default="clipseg")

    # output language (default English)
    ap.add_argument("--out_lang", choices=["en","zh"], default="en")

    # NEW: area filter to suppress tiny false positives
    ap.add_argument("--min_item_area", type=int, default=600)

    # SAM-2 (recommended: repo form)
    ap.add_argument("--sam2_repo", type=str, default="", help="e.g., facebook/sam2.1-hiera-large (preferred)")
    # fallback (local ckpt + YAML)
    ap.add_argument("--sam2_ckpt", type=str, default="")
    ap.add_argument("--sam2_model", type=str, default="", help="YAML name, e.g., sam2_hiera_l.yaml")

    # hotspot & targets
    ap.add_argument("--targets", type=str, default="cup,bottle,book,pen,bag,toolbox,tripod")
    ap.add_argument("--peak_topk", type=int, default=5)
    ap.add_argument("--peak_min", type=float, default=0.35)
    ap.add_argument("--peak_min_dist", type=int, default=16)

    # SAM-2 mask post
    ap.add_argument("--mask_iou_th", type=float, default=0.6)

    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data["frames"]

    # build frame name -> path
    name2path: Dict[str,str] = {}
    for e in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG"):
        for p in glob.glob(os.path.join(args.frames_dir, e)):
            name2path[os.path.basename(p)] = p

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc_clip, model_clip = load_clipseg(device)

    # SAM-2 predictor (if needed)
    predictor = None
    if args.mask_backend == "sam2":
        sam2_repo = args.sam2_repo.strip() or None
        sam2_model_yaml = args.sam2_model.strip() or None
        sam2_ckpt = args.sam2_ckpt.strip() or None
        predictor = load_sam2_predictor(
            device=device,
            sam2_repo=sam2_repo,
            sam2_model_yaml=sam2_model_yaml,
            sam2_ckpt=sam2_ckpt,
        )
        print(f"[INFO] SAM-2 predictor ready ({'repo:'+sam2_repo if sam2_repo else sam2_model_yaml})")

    suppress_set = set(parse_csv_list(args.suppress))

    # Parse targets (zh/en), normalize to EN
    targets_raw = parse_csv_list(args.targets)
    targets_en = uniq_keeporder([to_en(t) for t in targets_raw])

    new_frames = []
    srt_lines = []

    for i, rec in enumerate(frames):
        fname = rec.get("frame")
        fpath = name2path.get(fname)
        if not fpath:
            new_frames.append(rec)
            continue

        img_pil = Image.open(fpath).convert("RGB")
        img_np  = np.array(img_pil)  # H,W,3 (RGB)

        # ---- pull present labels (robust to zh/en fields) ----
        present = rec.get("topk", [])
        present_labels_all: List[str] = []
        for x in present:
            for k in ("label", "label_en", "label_zh"):
                v = x.get(k)
                if v: present_labels_all.append(v)
        present_labels_en = uniq_keeporder([to_en(v) for v in present_labels_all])

        # scene info (get both zh & en best-effort)
        scene_top = rec.get("scene_top", [])
        scene_zh = None
        scene_en = None
        if scene_top:
            scene_en = scene_top[0].get("label_en") or scene_top[0].get("label")
            scene_zh = scene_top[0].get("label_zh")
            if not scene_en and scene_zh:
                scene_en = to_en(scene_zh)

        # ----- (1) table/desk mask -----
        if args.mask_backend == "clipseg":
            table_mask: Optional[torch.Tensor] = None
            # probe with synonyms union for stability
            for base in sorted(TABLE_EN):
                queries = EN_SYNONYMS.get(base, [base])
                prob_union = clipseg_prob_multi(proc_clip, model_clip, img_pil, queries, device)
                m = (prob_union >= args.mask_th).float()
                table_mask = m if table_mask is None else torch.maximum(table_mask, m)
        else:
            # sam2: CLIPSeg prob (with synonyms) -> peaks -> SAM2 instances -> largest area as table
            prob_table_list = []
            for base in sorted(TABLE_EN):
                queries = EN_SYNONYMS.get(base, [base])
                prob_table_list.append(clipseg_prob_multi(proc_clip, model_clip, img_pil, queries, device))
            if not prob_table_list:
                table_mask = None
            else:
                prob_table = torch.stack(prob_table_list, dim=0).max(dim=0).values
                pts = find_peaks_from_prob(prob_table, topk=max(3, args.peak_topk),
                                           min_val=args.peak_min, min_dist=args.peak_min_dist)
                masks_tb = sam2_masks_from_points(predictor, img_np, pts) if pts else []
                masks_tb = merge_masks_by_iou(masks_tb, iou_th=args.mask_iou_th)
                if not masks_tb:
                    table_mask = None
                else:
                    areas = [m.sum().item() for m in masks_tb]
                    table_mask = masks_tb[int(np.argmax(areas))].float()

        # ----- (2) which small items to instance -----
        # candidates from present small items (EN) OR explicit targets
        cand_en = [lab for lab in present_labels_en if (lab in SMALL_ITEMS_EN or lab in targets_en)]
        for lab in targets_en:
            if lab not in cand_en:
                cand_en.append(lab)
        # suppress
        cand_en = [c for c in cand_en if c not in suppress_set]
        # skip items that do NOT have an English prompt (prevents zh-only)
        cand_en = [c for c in cand_en if is_english_text(c)]
        cand_en = uniq_keeporder(cand_en)

        table_items_en, visible_items_en = [], []

        if args.mask_backend == "clipseg":
            # STRICT: require an actual binary mask with area ≥ min_item_area
            for en_lab in cand_en:
                queries = EN_SYNONYMS.get(en_lab, [en_lab])
                prob_item = clipseg_prob_multi(proc_clip, model_clip, img_pil, queries, device)
                m_item = (prob_item >= args.mask_th).float()
                if m_item.sum().item() < args.min_item_area:
                    continue  # too small/no evidence → skip
                if table_mask is not None:
                    r = overlap_ratio(m_item, table_mask)
                    if r >= args.on_table_ratio: table_items_en.append(en_lab)
                    else:                         visible_items_en.append(en_lab)
                else:
                    visible_items_en.append(en_lab)
        else:
            # SAM2: hotspots (prob union) -> SAM2 masks -> require area; DO NOT add if no peaks/masks
            for en_lab in cand_en:
                queries = EN_SYNONYMS.get(en_lab, [en_lab])
                prob = clipseg_prob_multi(proc_clip, model_clip, img_pil, queries, device)
                pts = find_peaks_from_prob(prob, topk=args.peak_topk, min_val=args.peak_min, min_dist=args.peak_min_dist)
                if not pts:
                    continue  # no hotspot → don't report
                masks_it = sam2_masks_from_points(predictor, img_np, pts)
                masks_it = merge_masks_by_iou(masks_it, iou_th=args.mask_iou_th)
                if not masks_it:
                    continue
                m_item = max(masks_it, key=lambda t: t.sum().item()).float()
                if m_item.sum().item() < args.min_item_area:
                    continue
                if table_mask is not None:
                    r = overlap_ratio(m_item, table_mask)
                    if r >= args.on_table_ratio: table_items_en.append(en_lab)
                    else:                         visible_items_en.append(en_lab)
                else:
                    visible_items_en.append(en_lab)

        # enforce uniqueness & max_items limit for output
        table_items_en = uniq_keeporder(table_items_en)[:args.max_items]
        visible_items_en = uniq_keeporder(visible_items_en)[:args.max_items]

        # ----- (3) caption (EN/ZH) -----
        has_people = (
            any(x in present_labels_all for x in ["人","两个人","一小群人"]) or
            any(x in present_labels_en for x in ["person","two people","a small group"])
        )
        # prefer zh scene if available; format_caption handles zh->en fallback
        scene_for_cap = scene_zh or scene_en
        new_caption = format_caption(args.out_lang, scene_for_cap, table_items_en, visible_items_en, has_people)

        rec["caption_refined"] = new_caption
        new_frames.append(rec)

        t0, t1 = i/args.fps, (i+1)/args.fps
        srt_lines.append(f"{i+1}\n{timecode(t0)} --> {timecode(t1)}\n{new_caption}\n")

    out = {"frames": new_frames}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(args.out_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f"[OK] refine done -> {args.out_json}, {args.out_srt}")


if __name__ == "__main__":
    main()
