#!/usr/bin/env python3
# pipeline/patch_story_names_from_semframes.py
import re, os, os.path as op, json, argparse
from collections import Counter

# ---------- helpers ----------
def idx(name: str):
    m = re.findall(r'\d+', op.basename(str(name)))
    return int(m[-1]) if m else None

_ASCII_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9 _\-]*$')

def is_english_token(s: str) -> bool:
    if not s: return False
    s = str(s).strip()
    return bool(_ASCII_RE.fullmatch(s))

def clean_english(s: str) -> str:
    """Keep only ASCII letters/digits/space/_/-; strip; return '' if empty."""
    if s is None: return ''
    s = re.sub(r'[^A-Za-z0-9 _\-]', '', str(s)).strip()
    return s

def top_words_from_caption(cap: str, topk=6):
    if not cap: return []
    stop = {
        'the','a','an','of','in','on','with','and','or','to','at','this','that',
        'these','those','for','from','into','by','as','is','are','be'
    }
    ws = [w.lower() for w in re.findall(r'[A-Za-z]+', cap or '')]
    ws = [w for w in ws if len(w) >= 3 and w not in stop]
    # caption words are already English; keep them as-is
    return [w for w, _ in Counter(ws).most_common(topk)]

def collect_sem_by_index(sem_json):
    """
    Return idx -> {'labels': [English strings], 'caption': str, 'scene': [English strings]}
    """
    out = {}
    J = sem_json
    def add_frame(k, labels, cap, scene):
        # filter to English-only & cleaned
        lbls = []
        for lab in labels or []:
            lab2 = clean_english(lab)
            if is_english_token(lab2):
                lbls.append(lab2)
        scns = []
        for s in scene or []:
            s2 = clean_english(s)
            if is_english_token(s2):
                scns.append(s2)
        out[k] = {'labels': lbls, 'caption': cap or '', 'scene': scns}

    if isinstance(J, dict) and isinstance(J.get('frames'), list):
        for fr in J['frames']:
            k = idx(fr.get('frame') or fr.get('image') or fr.get('name'))
            if k is None: continue
            labels = []
            for it in fr.get('topk', []):
                # Prefer explicit English, otherwise try generic label and clean to English.
                lab = it.get('label_en') or it.get('label') or it.get('cat') or it.get('name')
                if lab: labels.append(lab)
            cap = fr.get('caption_refined') or fr.get('caption_en') or fr.get('caption')
            scene = []
            for it in fr.get('scene_top', []):
                s = it.get('label_en') or it.get('label') or it.get('label_zh') or it.get('name')
                if s: scene.append(s)
            add_frame(k, labels, cap, scene)
    else:
        # Fallback structure: best-effort
        for it in (J if isinstance(J, list) else []):
            k = idx(it.get('frame') or it.get('image') or it.get('name'))
            if k is None: continue
            labels = [str(x) for x in it.get('labels', [])]
            cap = it.get('caption_refined') or it.get('caption_en') or it.get('caption')
            scene = []
            add_frame(k, labels, cap, scene)
    return out

def canonical_area(o) -> float:
    a = o.get('area')
    if a is None: a = o.get('pix')
    if a is None: a = o.get('area_px')
    try:
        return float(a or 0.0)
    except Exception:
        return 0.0

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--story', required=True)
    ap.add_argument('--sem_frames', required=True)
    ap.add_argument('--max_names_per_frame', type=int, default=12,
                    help='Max candidate names per frame to assign to the largest instances')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    story = json.load(open(args.story, 'r', encoding='utf-8'))
    sem   = json.load(open(args.sem_frames, 'r', encoding='utf-8'))
    sem_by_idx = collect_sem_by_index(sem)

    total_frames = 0
    filled = 0
    retained_english_tokens = 0

    for fr in story.get('frames', []):
        total_frames += 1
        k = idx(fr.get('image'))
        cand = []
        if k in sem_by_idx:
            rec = sem_by_idx[k]
            # Candidates: scene + topk labels + caption words (all English-only already)
            cand += rec.get('scene', [])
            cand += rec.get('labels', [])
            cand += top_words_from_caption(rec.get('caption'))

        # de-duplicate (case-insensitive), keep order
        seen = set()
        names = []
        for w in cand:
            w2 = w.strip()
            if not is_english_token(w2):  # enforce English-only
                continue
            wkey = w2.lower()
            # drop generic junk if present
            if wkey in ('photo', 'office', 'room'):
                continue
            if wkey not in seen:
                seen.add(wkey)
                names.append(w2)
        retained_english_tokens += len(names)
        if not names:
            continue

        # Sort objects by area (desc). Also normalize area_px on the way in.
        objs = fr.get('objects', []) or []
        for o in objs:
            a = canonical_area(o)
            o['__area'] = a
            # write canonical area_px if missing
            if 'area_px' not in o and a > 0:
                o['area_px'] = a
        objs.sort(key=lambda x: -x.get('__area', 0.0))

        # Assign names to the largest N unnamed instances
        N = min(len(objs), min(args.max_names_per_frame, len(names)))
        j = 0
        for i in range(len(objs)):
            if j >= N: break
            o = objs[i]
            if o.get('name') or o.get('label') or o.get('category') or o.get('class'):
                continue
            o['name'] = names[j % len(names)]
            filled += 1
            j += 1

        # Keep current (area-sorted) order; if you need to preserve original order, remove this line.
        fr['objects'] = objs

    json.dump(story, open(args.out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out}; frames={total_frames} | english_tokens_retained={retained_english_tokens} | filled names for {filled} objects")

if __name__ == '__main__':
    main()
