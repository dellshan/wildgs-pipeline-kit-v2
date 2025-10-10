import os, os.path as op, json, random, shutil, argparse, glob
from collections import defaultdict

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def is_img(p): return op.splitext(p)[1].lower() in IMG_EXT

def find_all_images(root):
    return [p for p in glob.glob(op.join(root, "**/*"), recursive=True) if op.isfile(p) and is_img(p)]

def guess_label_txt(img_path, label_roots):
    base = op.splitext(op.basename(img_path))[0] + ".txt"
    for lr in label_roots:
        cand = op.join(lr, base)
        if op.isfile(cand): return cand
    local = op.join(op.dirname(img_path), base)
    return local if op.isfile(local) else None

def scan_yolo_txts(imgs, label_roots):
    pairs = []
    classes = set()
    for im in imgs:
        txt = guess_label_txt(im, label_roots)
        if txt and op.getsize(txt) > 0:
            pairs.append((im, txt))
            try:
                for ln in open(txt,"r",encoding="utf-8").read().splitlines():
                    ln=ln.strip()
                    if not ln: continue
                    c = int(ln.split()[0])
                    classes.add(c)
            except Exception:
                pass
    return pairs, sorted(classes)

def write_split(pairs, out_root, names):
    random.seed(42)
    imgs = [im for im,_ in pairs]
    random.shuffle(imgs)
    n = len(imgs)
    n_val = max(1, int(0.1*n))
    val_set = set(imgs[:n_val])
    os.makedirs(op.join(out_root,"images/train"), exist_ok=True)
    os.makedirs(op.join(out_root,"images/val"), exist_ok=True)
    os.makedirs(op.join(out_root,"labels/train"), exist_ok=True)
    os.makedirs(op.join(out_root,"labels/val"), exist_ok=True)

    txt_by_img = {im: txt for im,txt in pairs}

    for im in imgs:
        split = "val" if im in val_set else "train"
        dst_im = op.join(out_root, "images", split, op.basename(im))
        shutil.copy2(im, dst_im)
        if im in txt_by_img:
            dst_txt = op.join(out_root, "labels", split, op.splitext(op.basename(im))[0]+".txt")
            shutil.copy2(txt_by_img[im], dst_txt)

    yaml = f"""# auto-generated
path: datasets/indoor_obstacle
train: images/train
val: images/val
nc: {len(names)}
names: {json.dumps(names)}
"""
    cfg_dir = "configs/yolo"
    os.makedirs(cfg_dir, exist_ok=True)
    with open(op.join(cfg_dir, "data_indoor.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)
    print("[OK] wrote configs/yolo/data_indoor.yaml")

def find_coco_json(root):
    cands = []
    for p in glob.glob(op.join(root,"**/*.json"), recursive=True):
        try:
            J = json.load(open(p,"r",encoding="utf-8"))
            if isinstance(J, dict) and "images" in J and "annotations" in J:
                cands.append(p)
        except Exception:
            pass
    return cands[0] if cands else None

def coco_to_yolo(coco_json, out_tmp, image_root=None):
    os.makedirs(out_tmp, exist_ok=True)
    J = json.load(open(coco_json,"r",encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in J.get("categories", [])}
    names = [cats[k] for k in sorted(cats.keys())]
    name2idx = {n:i for i,n in enumerate(names)}
    imgs = {it["id"]:it for it in J["images"]}
    anns_by_img = defaultdict(list)
    for a in J["annotations"]:
        if "bbox" in a and a.get("iscrowd",0)==0:
            anns_by_img[a["image_id"]].append(a)

    pairs=[]
    for img_id, it in imgs.items():
        file_name = it["file_name"]
        im_path = file_name if op.isabs(file_name) else (op.join(image_root, file_name) if image_root else file_name)
        if not op.isfile(im_path):
            guess = glob.glob(op.join(op.dirname(coco_json),"**", file_name), recursive=True)
            if guess: im_path = guess[0]
        if not op.isfile(im_path): 
            continue
        W = it.get("width") or 0
        H = it.get("height") or 0
        lines=[]
        for a in anns_by_img.get(img_id, []):
            x,y,w,h = a["bbox"]
            cx = (x + w/2)/W if W else 0
            cy = (y + h/2)/H if H else 0
            nw = w/W if W else 0
            nh = h/H if H else 0
            cls_name = cats.get(a["category_id"], "object")
            cls_id = name2idx.get(cls_name, 0)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if lines:
            txt = op.join(out_tmp, op.splitext(op.basename(file_name))[0]+".txt")
            with open(txt,"w") as f: f.write("\n".join(lines))
            pairs.append((im_path, txt))
    return pairs, names

def find_voc_xmls(root):
    return glob.glob(op.join(root, "**/*.xml"), recursive=True)

def voc_to_yolo(xmls, out_tmp):
    import xml.etree.ElementTree as ET
    os.makedirs(out_tmp, exist_ok=True)
    names_order=[]
    name2idx={}
    pairs=[]
    for xp in xmls:
        try:
            tree = ET.parse(xp); root = tree.getroot()
            fn = root.findtext("filename")
            W = int(root.findtext("size/width"))
            H = int(root.findtext("size/height"))
            img_dir = op.dirname(xp)
            im_path = op.join(img_dir, fn)
            if not op.isfile(im_path):
                cand = glob.glob(op.join(op.dirname(xp), "**", fn), recursive=True)
                if cand: im_path = cand[0]
            if not op.isfile(im_path): 
                continue
            lines=[]
            for obj in root.findall("object"):
                cname = obj.findtext("name") or "object"
                if cname not in name2idx:
                    name2idx[cname] = len(name2idx)
                    names_order.append(cname)
                cls = name2idx[cname]
                bb = obj.find("bndbox")
                xmin = float(bb.findtext("xmin")); ymin=float(bb.findtext("ymin"))
                xmax = float(bb.findtext("xmax")); ymax=float(bb.findtext("ymax"))
                x = xmin; y = ymin; w = xmax - xmin; h = ymax - ymin
                cx = (x + w/2)/W; cy=(y + h/2)/H; nw=w/W; nh=h/H
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            if lines:
                txt = op.join(out_tmp, op.splitext(op.basename(fn))[0] + ".txt")
                with open(txt,"w") as f: f.write("\n".join(lines))
                pairs.append((im_path, txt))
        except Exception:
            pass
    return pairs, names_order or list(name2idx.keys())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="datasets/_raw/indoor_kaggle",
                    help="where you unzipped Kaggle dataset")
    ap.add_argument("--out_root", default="datasets/indoor_obstacle",
                    help="YOLO-ready output root")
    ap.add_argument("--split_ratio", type=float, default=0.9,
                    help="train ratio (rest for val)")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    imgs = find_all_images(args.raw_root)
    if not imgs:
        raise SystemExit(f"[ERR] no images found under {args.raw_root}")

    label_roots = []
    for sub in ("labels","Annotations","annotations","label","ann"):
        p = op.join(args.raw_root, sub)
        if op.isdir(p): label_roots.append(p)
    pairs, cls_ids = scan_yolo_txts(imgs, label_roots)
    names = []
    if pairs:
        print(f"[INFO] Found YOLO txt labels for {len(pairs)} images")
        classes_txt = glob.glob(op.join(args.raw_root, "**/classes.txt"), recursive=True)
        if classes_txt:
            names = [ln.strip() for ln in open(classes_txt[0],"r",encoding="utf-8").read().splitlines() if ln.strip()]
        else:
            K = max(cls_ids)+1 if cls_ids else 1
            names = [f"class_{i}" for i in range(K)]
        write_split(pairs, args.out_root, names)
        return

    coco_json = find_coco_json(args.raw_root)
    if coco_json:
        print(f"[INFO] Found COCO json: {coco_json}")
        tmp = op.join(args.out_root, "_tmp_labels")
        pairs, names = coco_to_yolo(coco_json, tmp, image_root=args.raw_root)
        if not pairs:
            raise SystemExit("[ERR] COCO file found but produced no pairs")
        write_split(pairs, args.out_root, names)
        shutil.rmtree(tmp, ignore_errors=True)
        return

    xmls = find_voc_xmls(args.raw_root)
    if xmls:
        print(f"[INFO] Found {len(xmls)} VOC XML files")
        tmp = op.join(args.out_root, "_tmp_voc")
        pairs, names = voc_to_yolo(xmls, tmp)
        if not pairs:
            raise SystemExit("[ERR] VOC XMLs found but produced no pairs")
        write_split(pairs, args.out_root, names)
        shutil.rmtree(tmp, ignore_errors=True)
        return

    raise SystemExit("[ERR] Could not find labels in YOLO/COCO/VOC format under raw_root")

if __name__ == "__main__":
    main()
