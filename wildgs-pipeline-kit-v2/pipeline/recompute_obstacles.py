# pipeline/recompute_obstacles.py
import os, os.path as op, json, argparse, glob, collections
from PIL import Image, ImageOps

# Tunable risk weights
RISK_WEIGHTS = {
    'cable': 1.6, 'wire': 1.6, 'ladder': 1.8, 'box': 1.3, 'chair': 1.2,
    'trash': 1.2, 'trash can': 1.1, 'stool': 1.2, 'drawer': 1.1,
    'person': 0.8, 'table': 0.9, 'cart': 1.2, 'robot': 1.1, 'tripod': 1.2,
    'bottle': 1.1, 'cone': 1.2, 'crate': 1.2
}

GROUND_LIKE = (
    'box','ladder','stool','trash','bin','cable','wire','bottle','cone',
    'crate','cart','robot','tool','stand','tripod','pipe','hose','plank',
    'board','pallet','case'
)

def mask_area(mask_path):
    try:
        m = Image.open(mask_path).convert('L')
        # treat >0 as foreground
        m = m.point(lambda x: 255 if x > 0 else 0)
        return m.histogram()[255]
    except Exception:
        return 0

def find_mask(mask_root, image_path, inst=None, hinted=None):
    """Try to locate a mask file for this object if story doesn't give a good path."""
    if hinted and op.isfile(hinted):
        return hinted
    if hinted and op.isfile(op.join(mask_root, hinted)):
        return op.join(mask_root, hinted)
    stem = op.splitext(op.basename(image_path))[0] if image_path else None
    if not stem:
        return None
    patt = op.join(mask_root, f"{stem}*{'*'+str(inst) if inst is not None else ''}*.png")
    cands = glob.glob(patt)
    if not cands:
        cands = glob.glob(op.join(mask_root, f"{stem}*.png"))
    return max(cands, key=lambda p: op.getsize(p)) if cands else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--story', required=True)
    ap.add_argument('--mask_root', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    story = json.load(open(args.story,'r'))

    # Global frequency (lightly weighted by frame coverage) used as fallback
    freq = collections.Counter()
    total_seen = 0.0
    for fr in story.get('frames', []):
        w = 1.0 + float(fr.get('coverage', 0.0))
        for o in fr.get('objects', []):
            n = (o.get('name') or o.get('label') or o.get('category') or '').strip().lower()
            if not n: continue
            freq[n] += w
            total_seen += w

    out = {'frames': []}
    masks_found = 0
    masks_referenced = 0

    for fr in story.get('frames', []):
        img  = fr.get('image')
        W = int(fr.get('width', 1280)); H = int(fr.get('height', 720))
        items=[]
        for o in fr.get('objects', []):
            name = (o.get('name') or o.get('label') or o.get('category') or '').strip().lower()
            if not name: 
                continue

            # 1) try masks
            mrel = o.get('mask') or ''
            inst = o.get('inst') or o.get('instance') or o.get('id') or o.get('instance_id')
            hinted = mrel if op.isabs(mrel) else (op.join(args.mask_root, mrel) if mrel else None)
            if mrel: masks_referenced += 1
            mabs = find_mask(args.mask_root, img, inst=inst, hinted=hinted)

            area = 0
            if mabs and op.isfile(mabs):
                area = mask_area(mabs)
                masks_found += 1
            else:
                # 2) fallback numeric fields if present
                area = int(o.get('area') or o.get('pix') or o.get('area_px') or 0)

            # 3) fraction
            frac = (area / float(max(1, W*H))) if area > 0 else 0.0

            # 4) if still zero, use a robust fallback so bars aren't blank
            if frac == 0.0:
                f = freq.get(name, 0.0) / max(1e-6, total_seen)     # 0..~1 normalized frequency
                on_floor = any(k in name for k in GROUND_LIKE)
                # Small base + frequency + on-floor boost; clipped to reasonable range
                frac = min(0.05 + 0.35 * f + (0.10 if on_floor else 0.0), 0.9)

            # 5) risk score
            w = RISK_WEIGHTS.get(name, 1.0)
            score = w * frac

            items.append(dict(name=name, area_frac=frac, score=score))

        items.sort(key=lambda x: -x['score'])
        out['frames'].append(dict(image=img, obstacles=items[:8]))

    json.dump(out, open(op.join(args.out_dir, 'obstacles_recomputed.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"[OK] obstacles -> {op.join(args.out_dir, 'obstacles_recomputed.json')} | "
          f"mask_refs={masks_referenced} found={masks_found}")

if __name__ == '__main__':
    main()
