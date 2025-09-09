#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, json, csv, glob, re
from PIL import Image
import torch
import torchvision.transforms as T

# ====== robust LSeg import (supports multiple layouts) ======
_HAS_LSEG = False
try:
    # case 1: repo root as CWD and pipeline is a package
    from pipeline.lseg.lseg_net import LSegNet  # noqa: F401
    _HAS_LSEG = True
    _LSEG_IMPORT_PATH = "pipeline.lseg.lseg_net"
except Exception:
    try:
        # case 2: running from pipeline/ directory
        from lseg.lseg_net import LSegNet  # noqa: F401
        _HAS_LSEG = True
        _LSEG_IMPORT_PATH = "lseg.lseg_net"
    except Exception:
        # case 3: add repo root to sys.path and retry both
        this_dir = os.path.dirname(os.path.abspath(__file__))               # .../pipeline
        repo_root = os.path.abspath(os.path.join(this_dir, ".."))           # repo root
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from pipeline.lseg.lseg_net import LSegNet  # noqa: F401
            _HAS_LSEG = True
            _LSEG_IMPORT_PATH = "pipeline.lseg.lseg_net"
        except Exception:
            try:
                from lseg.lseg_net import LSegNet  # noqa: F401
                _HAS_LSEG = True
                _LSEG_IMPORT_PATH = "lseg.lseg_net"
            except Exception:
                _HAS_LSEG = False
                _LSEG_IMPORT_PATH = None

# -------------------- utils --------------------
def timecode(t):
    ms = int(round(t * 1000.0)); s, msec = divmod(ms, 1000)
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"

def parse_ignore(s):
    if not s: return set()
    return set([x.strip() for x in re.split(r"[,\u3001;；\s]+", s) if x.strip()])

def list_frames(frames_dir):
    files = []
    for e in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG"):
        files += glob.glob(os.path.join(frames_dir, e))
    files = sorted(files)
    return files

def default_prompts(lang):
    # English prompts drive the model; zh labels only for display
    en = [
        # structure / room
        "wall","floor","ceiling","window","door","whiteboard",
        # furniture
        "table","desk","chair","sofa","cabinet","shelf",
        # electronics
        "monitor","screen","computer","laptop","keyboard","mouse",
        "projector","microphone","camera","tripod","robot arm","sensor",
        "printer","power strip","charger","adapter","cable",
        # small items / clutter
        "bottle","cup","mug","book","notebook","paper","box","bag","backpack",
        "phone","remote control","pen","marker","tape","scissors","toolbox",
        # people & lights / coarse scene hints
        "person","two people","a small group","lab","office","indoor lighting","natural lighting",
        # helpful phrases
        "desk clutter","messy desk","random small items"
    ]
    zh = [
        "墙","地面","天花板","窗","门","白板",
        "桌子","书桌","椅子","沙发","柜子","架子",
        "显示器","屏幕","电脑","笔记本电脑","键盘","鼠标",
        "投影仪","麦克风","相机","三脚架","机械臂","传感器",
        "打印机","插线板","充电器","适配器","电缆",
        "瓶子","杯子","马克杯","书","笔记本","纸张","盒子","包","背包",
        "手机","遥控器","笔","马克笔","胶带","剪刀","工具箱",
        "人","两个人","一小群人","实验室","办公室","室内光","自然光",
        "桌面杂物","凌乱的桌面","一些小物品"
    ]
    return (zh, en) if lang == "zh" else (en, en)

# -------------------- CLIPSeg backend --------------------
def load_clipseg(device):
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    model.eval()
    return proc, model

@torch.no_grad()
def run_clipseg_on_image(img, texts, proc, model, device, bs=16,
                         pooling="topk", topk_ratio=0.10):
    all_scores = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        imgs = [img] * len(chunk)
        inputs = proc(text=chunk, images=imgs,
                      padding=True, truncation=True,
                      return_tensors="pt")
        inputs = {k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}
        out = model(**inputs)                 # logits [B,1,H',W']
        logits = out.logits.squeeze(1)
        h, w = img.size[1], img.size[0]
        logits = torch.nn.functional.interpolate(
            logits.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(1)
        probs = torch.sigmoid(logits).flatten(1)  # [B, H*W]
        if pooling == "mean":
            scores = probs.mean(dim=1)
        else:
            k = max(1, int(probs.shape[1] * float(topk_ratio)))
            topk_vals, _ = torch.topk(probs, k, dim=1)
            scores = topk_vals.mean(dim=1)
        all_scores.extend(scores.detach().float().cpu().tolist())
    return all_scores

# -------------------- LSeg backend --------------------
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def _letterbox_pad(x, s):
    _, h, w = x.shape
    scale = min(s / h, s / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    x = T.functional.resize(x, [nh, nw], antialias=True)
    out = torch.zeros((3, s, s), dtype=x.dtype, device=x.device)
    out += torch.tensor(_CLIP_MEAN, dtype=x.dtype, device=x.device).view(3,1,1)
    top = (s - nh) // 2
    left = (s - nw) // 2
    out[:, top:top+nh, left:left+nw] = x
    return out

def _make_preproc(target=384):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),  # scales to [0,1]
        T.Lambda(lambda x: _letterbox_pad(x, target)),
        T.Normalize(_CLIP_MEAN, _CLIP_STD),
    ])

def _clean_state_dict(sd):
    if "state_dict" in sd: sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for p in ("module.", "net.", "model."):
            if nk.startswith(p): nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd

def load_lseg(labels, ckpt, backbone, device, in_size=384):
    if not _HAS_LSEG:
        raise RuntimeError(
            "无法导入 LSegNet。请确认以下任一条件成立：\n"
            "1) 存在文件 pipeline/__init__.py 与 pipeline/lseg/{__init__,lseg_net, ...}.py；\n"
            "2) 或把 repo 根目录加入 PYTHONPATH；\n"
            "3) 或使用本脚本内置的路径探测（已尝试失败）。"
        )
    # 动态导入真正的类对象（为兼容不同导入路径）
    if _LSEG_IMPORT_PATH == "pipeline.lseg.lseg_net":
        from pipeline.lseg.lseg_net import LSegNet as _LSegNet
    else:
        from lseg.lseg_net import LSegNet as _LSegNet

    # 首选：路径+标签构造（某些 fork 支持）
    try:
        model = _LSegNet(
            labels=labels,
            path=ckpt,
            backbone=backbone,
            features=256,
            activation='lrelu'
        )
        model = model.to(device).eval()
    except TypeError:
        # 退回：手动 load_state_dict
        model = _LSegNet(
            backbone=backbone,
            features=256,
            activation='lrelu',
            num_classes=len(labels) if "num_classes" in _LSegNet.__init__.__code__.co_varnames else None
        )
        sd = torch.load(ckpt, map_location="cpu")
        sd = _clean_state_dict(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(unexpected) > 0:
            print(f"[WARN] LSeg ckpt 有未用到权重: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        if len(missing) > 0:
            print(f"[WARN] LSeg 模型缺少权重: {missing[:5]}{'...' if len(missing)>5 else ''}")
        model = model.to(device).eval()

    preproc = _make_preproc(in_size)
    print(f"[INFO] LSeg imported from: {_LSEG_IMPORT_PATH}")
    return preproc, model

@torch.no_grad()
def run_lseg_on_image(pil_img, labels, preproc, model, device,
                      pooling="mean", topk_ratio=0.1):
    x = preproc(pil_img).unsqueeze(0).to(device)  # [1,3,S,S]
    out = model(x)
    if isinstance(out, (list, tuple)):
        logits = out[0]
    elif isinstance(out, dict):
        logits = out.get("out", out.get("logits", None))
        if logits is None:
            for v in out.values():
                if torch.is_tensor(v): logits = v; break
            if logits is None:
                raise RuntimeError("LSeg forward returned dict without logits.")
    else:
        logits = out
    if logits.ndim != 4:
        raise RuntimeError(f"Unexpected LSeg output shape: {tuple(logits.shape)}")
    probs = torch.softmax(logits, dim=1)[0]   # [C,H,W]
    C, H, W = probs.shape
    flat = probs.view(C, -1)
    if pooling == "topk":
        k = max(1, int(flat.shape[1] * topk_ratio))
        vals, _ = torch.topk(flat, k, dim=1)
        scores = vals.mean(dim=1)
    else:
        scores = flat.mean(dim=1)
    return [float(s) for s in scores], (H, W)

# -------------------- scene classification (global CLIP) --------------------
SCENE_CAND_EN = [
    "office","laboratory","classroom","meeting room","library","corridor",
    "kitchen","restaurant","living room","workshop","warehouse","server room","lobby","reception"
]
SCENE_EN2ZH = {
    "office":"办公室","laboratory":"实验室","classroom":"教室","meeting room":"会议室","library":"图书馆",
    "corridor":"走廊","kitchen":"厨房","restaurant":"餐厅","living room":"客厅","workshop":"车间",
    "warehouse":"仓库","server room":"机房","lobby":"大堂","reception":"前台"
}

def load_global_clip(device):
    from transformers import CLIPProcessor, CLIPModel
    dtype = torch.float16 if (torch.cuda.is_available() and "cuda" in str(device)) else torch.float32
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        attn_implementation="eager",
        torch_dtype=dtype
    ).to(device).eval()
    return proc, model

@torch.no_grad()
def classify_scene(img, scene_texts, proc, model, device):
    ti = proc(text=[f"a photo of a {t}" for t in scene_texts], return_tensors="pt", padding=True).to(device)
    ii = proc(images=img, return_tensors="pt").to(device)
    text_feat = model.get_text_features(**ti)
    img_feat  = model.get_image_features(**ii)
    text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
    img_feat  = torch.nn.functional.normalize(img_feat, dim=-1)
    sims = (img_feat @ text_feat.T).squeeze(0)
    topv, topi = torch.topk(sims, k=min(2, sims.numel()))
    return [(scene_texts[i], float(v)) for v, i in zip(topv.tolist(), topi.tolist())]

# -------------------- caption compose --------------------
SMALL_ITEMS_ZH = {"杯子","马克杯","瓶子","书","笔记本","纸张","手机","遥控器","笔","马克笔","胶带","剪刀","工具箱","盒子","包","背包","电缆","充电器","适配器","相机","三脚架","键盘","鼠标"}
TABLE_ZH = {"桌子","书桌"}
PEOPLE_ZH = {"人","两个人","一小群人"}

def compose_captions(lang, present_labels_zh, scene_top):
    scene_en = scene_top[0][0] if scene_top else None
    scene_zh = SCENE_EN2ZH.get(scene_en, None) if scene_en else None
    items = [x for x in present_labels_zh if x in SMALL_ITEMS_ZH]
    people = [x for x in present_labels_zh if x in PEOPLE_ZH]
    has_table = any(x in TABLE_ZH for x in present_labels_zh)
    def list_zh(xs, topn=3): return "、".join(xs[:topn])
    parts = []
    if scene_zh: parts.append(f"这是一个{scene_zh}")
    if people: parts.append(f"有人活动")
    if has_table and items: parts.append(f"桌面上有{list_zh(items,3)}")
    elif items: parts.append(f"可见{list_zh(items,3)}")
    cap_zh = "，".join(parts) if parts else "这是一个室内场景"
    scene_en_phrase = f"in a {scene_en}" if scene_en else "indoors"
    if has_table and items:
        obj_en = " and ".join(items[:2]) if len(items) >= 2 else (items[0] if items else "some items")
        cap_en = f"A photo of {obj_en} on a table {scene_en_phrase}."
    elif items:
        obj_en = " and ".join(items[:2]) if len(items) >= 2 else items[0]
        cap_en = f"A photo of {obj_en} {scene_en_phrase}."
    else:
        cap_en = f"A photo {scene_en_phrase}."
    return cap_zh, cap_en

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--backend", default="clipseg", choices=["clipseg","lseg"])
    ap.add_argument("--lang", default="zh", choices=["zh","en"])
    ap.add_argument("--out_json", default="output/ovseg.json")
    ap.add_argument("--out_csv",  default="output/ovseg.csv")
    ap.add_argument("--srt",      default="output/ovseg.srt")
    ap.add_argument("--fps", type=float, default=7.0)

    ap.add_argument("--present_thres", type=float, default=0.20)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--prompt_bs", type=int, default=12)  # only for clipseg
    ap.add_argument("--ignore_labels", type=str, default="墙,地面,办公室,实验室,室内光,自然光")
    ap.add_argument("--score_pooling", choices=["mean","topk"], default="topk")
    ap.add_argument("--score_topk_ratio", type=float, default=0.10)

    ap.add_argument("--enable_scene", action="store_true")
    ap.add_argument("--scene_list", type=str, default="office,laboratory,classroom,meeting room,library,corridor,kitchen,restaurant,living room,workshop,warehouse,server room,lobby,reception")

    # LSeg options
    ap.add_argument("--lseg_ckpt", type=str, default="models/demo_e200.ckpt")
    ap.add_argument("--lseg_backbone", type=str, default="clip_vitl16_384")
    ap.add_argument("--lseg_size", type=int, default=384)
    args = ap.parse_args()

    files = list_frames(args.frames_dir)
    if not files:
        print(f"[ERR] No images in {args.frames_dir}", file=sys.stderr); sys.exit(1)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    zh_labels, en_prompts = default_prompts(args.lang)
    out_labels = zh_labels if args.lang == "zh" else en_prompts
    ignore = parse_ignore(args.ignore_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # scene backend
    scene_texts = [x.strip() for x in args.scene_list.split(",") if x.strip()]
    if args.enable_scene:
        scene_proc, scene_model = load_global_clip(device)
    else:
        scene_proc = scene_model = None

    scene_flag = f"scene_cls=ON({len(scene_texts)})" if args.enable_scene else "scene_cls=OFF"

    if args.backend == "clipseg":
        proc, model = load_clipseg(device)
        print(f"[RUN] backend=clipseg frames={len(files)} prompts={len(en_prompts)} device={device} "
              f"bs={args.prompt_bs} pooling={args.score_pooling}@{args.score_topk_ratio} {scene_flag}")
        infer_fn = lambda img: run_clipseg_on_image(
            img, en_prompts, proc, model, device,
            bs=args.prompt_bs, pooling=args.score_pooling, topk_ratio=args.score_topk_ratio
        )
    else:
        preproc, lseg_model = load_lseg(
            labels=en_prompts,
            ckpt=args.lseg_ckpt,
            backbone=args.lseg_backbone,
            device=device,
            in_size=args.lseg_size
        )
        print(f"[RUN] backend=lseg frames={len(files)} prompts={len(en_prompts)} device={device} "
              f"input={args.lseg_size} pooling={args.score_pooling}@{args.score_topk_ratio} {scene_flag}")
        infer_fn = lambda img: run_lseg_on_image(
            img, en_prompts, preproc, lseg_model, device,
            pooling=args.score_pooling, topk_ratio=args.score_topk_ratio
        )[0]

    per_frame, csv_rows = [], []
    for i, fp in enumerate(files):
        img = Image.open(fp).convert("RGB")
        scores = infer_fn(img)
        for (l, s) in zip(out_labels, scores):
            csv_rows.append({"frame": os.path.basename(fp), "label": l, "score": f"{s:.4f}"})
        pairs_all = list(zip(out_labels, scores))
        pairs_all.sort(key=lambda x: x[1], reverse=True)
        pairs = [(l, s) for (l, s) in pairs_all if l not in ignore]
        present = [(l, s) for (l, s) in pairs if s >= args.present_thres][:args.topk]
        present_labels = [l for (l, _) in present]
        scene_top = classify_scene(img, scene_texts, scene_proc, scene_model, device) if scene_model is not None else []
        cap_zh, cap_en = compose_captions(args.lang, present_labels, scene_top)
        print(f"[{i+1}/{len(files)}] {os.path.basename(fp)} :: {cap_zh}")
        per_frame.append({
            "frame": os.path.basename(fp),
            "topk": [{"label": l, "score": s} for (l, s) in present],
            "scene_top": [{"label_en": en, "label_zh": SCENE_EN2ZH.get(en, en), "score": sc} for (en, sc) in scene_top],
            "caption": cap_zh,
            "caption_en": cap_en
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"frames": per_frame}, f, ensure_ascii=False, indent=2)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "label", "score"])
        w.writeheader(); w.writerows(csv_rows)
    with open(args.srt, "w", encoding="utf-8") as f:
        for i, rec in enumerate(per_frame):
            t0 = i / args.fps; t1 = (i + 1) / args.fps
            f.write(f"{i+1}\n{timecode(t0)} --> {timecode(t1)}\n{rec['caption']}\n\n")
    print(f"[OK] 写入：{args.out_json}, {args.out_csv}, {args.srt}")

if __name__ == "__main__":
    main()
