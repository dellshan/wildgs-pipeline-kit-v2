#!/usr/bin/env python3
import os, os.path as op, json, argparse, re
from math import fabs

try:
    from PIL import Image
except Exception:
    Image = None  # Pillow optional

# ---------- blocklists (not objects) ----------
PHRASE_BLOCK = {
    "looks like", "look like", "kind of", "sort of",
    "present", "visible", "maybe", "probably", "possibly",
    "there is", "there are", "appears", "seems",
}
TOKEN_BLOCK = {
    "looks", "look", "like", "present", "visible",
    "maybe", "probably", "possibly",
    "photo", "image", "scene", "object", "thing", "stuff",
}

# ---------- canonical set for lab scenes ----------
CANONICAL = {
    "person", "desk", "ladder", "door", "box",
    "trash bin",   # unify trash/garbage/dustbin/bin
    "bag",         # unify paper/plastic/backpack
    "cabinet",
    "basketball",
    "banana",
    "orange",
    "wood",        # unify plank/board/timber
    "watch",
    # optional tech/furniture seen in your scenes
    "bottle", "chair", "computer", "monitor", "robot", "cart",
}

# ---------- phrase-level aliases (run BEFORE token aliases) ----------
PHRASE_ALIASES = {
    # receptacles
    "trash can": "trash bin",
    "garbage can": "trash bin",
    "garbage bin": "trash bin",
    "dust bin": "trash bin",
    "dustbin": "trash bin",
    "waste bin": "trash bin",
    "wastebasket": "trash bin",

    # bags
    "paper bag": "bag",
    "plastic bag": "bag",

    # boxes / wood
    "cardboard box": "box",
    "shoe box": "box",
    "wooden board": "wood",
    "wood board": "wood",

    # watches
    "wrist watch": "watch",
    "wristwatch": "watch",
}

# ---------- token-level aliases (AFTER phrase aliases) ----------
ALIASES = {
    # people
    "people": "person", "persons": "person",
    "men": "person", "man": "person",
    "women": "person", "woman": "person",
    "kids": "person", "child": "person", "children": "person",
    "ladies": "person", "lady": "person",

    # furniture / fixtures
    "desks": "desk", "table": "desk", "tables": "desk",
    "workbench": "desk", "workbenches": "desk",
    "ladders": "ladder", "step-ladder": "ladder", "stepladder": "ladder",
    "doors": "door",
    "boxes": "box", "carton": "box", "cartons": "box",
    "cabinet": "cabinet", "cabinets": "cabinet",
    "cupboard": "cabinet", "cupboards": "cabinet",
    "locker": "cabinet", "lockers": "cabinet",

    # receptacles / bags
    "bin": "trash bin", "bins": "trash bin", "ashcan": "trash bin",
    "trash": "trash bin", "trashcan": "trash bin",
    "bag": "bag", "bags": "bag", "backpack": "bag", "backpacks": "bag",

    # small objects / food / sports
    "bottles": "bottle",
    "chairs": "chair",
    "watches": "watch",
    "bananas": "banana",
    "oranges": "orange",
    "basketballs": "basketball",

    # materials / parts
    "plank": "wood", "planks": "wood",
    "board": "wood", "boards": "wood",
    "timber": "wood", "woods": "wood", "plywood": "wood",

    # optional tech
    "computers": "computer", "pcs": "computer",
    "monitors": "monitor",
    "robots": "robot", "carts": "cart",
}

def enforce_canonical(name: str) -> str:
    """Only allow names from CANONICAL (after aliasing)."""
    if not name:
        return ""
    if name in CANONICAL:
        return name
    name2 = name.replace("-", " ")
    return name2 if name2 in CANONICAL else ""

def norm_name(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()

    # keep known phrases first
    for p, repl in PHRASE_ALIASES.items():
        s = re.sub(r"\b" + re.escape(p) + r"\b", repl, s)

    # kill bad phrases (e.g., "looks like")
    for ph in PHRASE_BLOCK:
        if ph in s:
            return ""

    # keep letters/space/hyphen; collapse spaces
    s = re.sub(r"[^a-z\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    # remove filler tokens
    toks = [t for t in s.split() if t not in TOKEN_BLOCK]
    if not toks:
        return ""
    s = " ".join(toks)

    # token-level aliasing (plural→singular etc.)
    toks = [ALIASES.get(t, t) for t in s.split()]
    s = " ".join(toks).strip()
    s = re.sub(r"\s+", " ", s)

    # very short -> drop
    if len(s) <= 1:
        return ""

    # finally enforce canonical set
    s = enforce_canonical(s)
    return s

# ---------- area helpers ----------
def polygon_area(poly):
    """poly: [x1,y1,x2,y2,...] or [[x,y], ...]"""
    if not poly:
        return 0.0
    if isinstance(poly[0], (list, tuple)):
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
    else:
        if len(poly) % 2 != 0:
            return 0.0
        xs = list(map(float, poly[0::2]))
        ys = list(map(float, poly[1::2]))
    n = len(xs)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return fabs(s) / 2.0

def bbox_area(bb):
    """Accept [x,y,w,h] or [x1,y1,x2,y2]."""
    if not bb or len(bb) < 4:
        return 0.0
    try:
        x, y, w, h = map(float, bb[:4])
        if w >= 0 and h >= 0:
            return w * h
    except Exception:
        pass
    try:
        x1, y1, x2, y2 = map(float, bb[:4])
        if x2 > x1 and y2 > y1:
            return (x2 - x1) * (y2 - y1)
    except Exception:
        pass
    return 0.0

def mask_area(mask_path):
    if not mask_path or not Image:
        return 0.0
    try:
        im = Image.open(mask_path).convert("L")
        # count >0 as foreground pixels
        return float(sum(1 for v in im.getdata() if v > 0))
    except Exception:
        return 0.0

def compute_area_px(obj, mask_root=None):
    # 1) mask
    m = obj.get("mask")
    if m:
        p = m if op.isabs(m) else op.join(mask_root, m) if mask_root else m
        a = mask_area(p)
        if a > 0:
            return a
    # 2) polygon
    if obj.get("polygon"):
        a = polygon_area(obj["polygon"])
        if a > 0:
            return a
    # 3) bbox
    if obj.get("bbox"):
        a = bbox_area(obj["bbox"])
        if a > 0:
            return a
    # 4) existing numeric area fields
    for k in ("area_px", "area", "pix", "area_px_est"):
        try:
            a = float(obj.get(k) or 0)
            if a > 0:
                return a
        except Exception:
            pass
    return 0.0

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--mask_root", default="",
                    help="Root for binary masks if relative paths are used")
    ap.add_argument("--drop_empty_names", action="store_true",
                    help="If cleaned name is empty, drop the instance")
    args = ap.parse_args()

    J = json.load(open(args.in_json, "r", encoding="utf-8"))

    fixed, removed, area_fixed = 0, 0, 0
    for fr in J.get("frames", []):
        objs = fr.get("objects", []) or []
        new_objs = []
        for o in objs:
            raw = o.get("name") or o.get("label") or o.get("category") or ""
            new = norm_name(raw)
            if not new:
                if args.drop_empty_names:
                    removed += 1
                    continue
                else:
                    for k in ("name", "label", "category"):
                        if k in o:
                            o[k] = ""
            else:
                if new != raw:
                    fixed += 1
                o["name"] = new

            a = compute_area_px(o, mask_root=args.mask_root)
            if a > 0:
                prev = float(o.get("area_px") or 0)
                if prev != a:
                    o["area_px"] = a
                    area_fixed += 1

            new_objs.append(o)
        fr["objects"] = new_objs

    json.dump(J, open(args.out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out_json}; names_fixed={fixed}, removed={removed}, area_fixed={area_fixed}")

if __name__ == "__main__":
    main()
