#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich story outputs with:
 (i) inter-view link descriptions (Δtranslation, Δyaw/pitch) from poses
(ii) mixed narration scaffolds (connections + per-frame summaries)
(iii) heuristic obstacle analysis (children/elderly risks)
(iv) risk overlays on frames

Inputs:
  --story      path to story.json from build_view_story3d.py
  --frames     directory of frames (png/jpg)
  --mask_root  mask root to compute bbox if story lacks bbox (optional)
  --poses      optional poses json (robust loader supports multiple formats)
  --out_dir    output folder (you may reuse story’s out_dir)

Outputs (under out_dir):
  links.json, links.md
  obstacles.json, obstacles.md
  narrative_prompt.md         (LLM-ready scaffold)
  ann_obstacles/*.png         (risk overlays)
"""

import os, json, math, argparse, warnings, os.path as op
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# ---------- utilities ----------

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p):
    os.makedirs(op.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_text(s, p):
    os.makedirs(op.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)

def norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))

def wrap_deg(a):
    # wrap to [-180, 180)
    a = (a + 180.0) % 360.0 - 180.0
    return float(a)

# ---------- pose loading (robust to formats) ----------

def _as44(m):
    M = np.asarray(m, dtype=float)
    if M.shape == (4,4): return M
    raise ValueError("not 4x4")

def _as33(m):
    R = np.asarray(m, dtype=float)
    if R.shape == (3,3): return R
    raise ValueError("not 3x3")

def _quat_to_R(qwqxqyqz):
    qw,qx,qy,qz = map(float, qwqxqyqz)
    # unit quaternion
    n = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz)+1e-8
    qw,qx,qy,qz = qw/n, qx/n, qy/n, qz/n
    # Hamilton -> rotation
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)],
    ], float)
    return R

def rot_to_yaw_pitch(Rwc):
    # camera forward z_cam = [0,0,1] in world -> d = Rwc @ ez
    ez = np.array([0,0,1.0], float)
    d  = (Rwc @ ez.reshape(3,1)).ravel()
    # yaw around y-up or z-up? Assume y-up world: yaw from xz-plane
    # If your world is z-up, adapt accordingly.
    yaw   = math.degrees(math.atan2(d[0], d[2]))          # [-180,180)
    pitch = math.degrees(math.atan2(-d[1], math.hypot(d[0], d[2])))
    return float(yaw), float(pitch)

def pose_from_entry(ent: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (Rwc, twc) for camera-to-world.
    Supports common keys:
      - 4x4 under keys ['Twc','c2w','pose','T','matrix']
      - world-to-camera 4x4 under ['Tcw','w2c'] (will invert)
      - R(3x3)+t(3), optionally flag 'convention': 'c2w'|'w2c'
      - quaternion + t: {'q':[w,x,y,z],'t':[x,y,z], convention?}
    """
    # 4x4 c2w
    for k in ['Twc','c2w','pose','T','matrix']:
        if k in ent:
            M = _as44(ent[k])
            Rwc, twc = M[:3,:3], M[:3,3]
            return Rwc, twc
    # 4x4 w2c
    for k in ['Tcw','w2c']:
        if k in ent:
            M = _as44(ent[k])
            Rcw, tcw = M[:3,:3], M[:3,3]
            Rwc = Rcw.T
            twc = -Rcw.T @ tcw
            return Rwc, twc
    # R + t
    if 'R' in ent and 't' in ent:
        R = _as33(ent['R']); t = np.asarray(ent['t'], float).reshape(3)
        conv = str(ent.get('convention','c2w')).lower()
        if conv in ['c2w','twc','camera_to_world','cam2world']:
            return R, t
        else:
            # assume w2c
            Rwc = R.T; twc = -R.T @ t
            return Rwc, twc
    # q + t
    if 'q' in ent and 't' in ent:
        R = _quat_to_R(ent['q'])
        t = np.asarray(ent['t'], float).reshape(3)
        conv = str(ent.get('convention','c2w')).lower()
        if conv in ['c2w','twc','camera_to_world','cam2world']:
            return R, t
        else:
            Rwc = R.T; twc = -R.T @ t
            return Rwc, twc
    raise ValueError("Unrecognized pose entry format")

def load_poses(poses_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns map: image_name -> {'Rwc':(3,3), 'twc':(3,), 'yaw':deg, 'pitch':deg}
    Supports:
      - { "path_00000.png": {...}, ... }
      - { "frames":[ {"image":..., ...pose...}, ...] }
      - list of entries with 'image' key
    """
    P = load_json(poses_path)
    out = {}
    # top-level dict mapping
    if isinstance(P, dict) and all(isinstance(v, (dict,list)) for v in P.values()):
        for k,v in P.items():
            try:
                Rwc, twc = pose_from_entry(v)
                yaw, pitch = rot_to_yaw_pitch(Rwc)
                out[k] = {'Rwc':Rwc, 'twc':twc, 'yaw':yaw, 'pitch':pitch}
            except Exception:
                continue
        if out:
            return out
    # frames list
    if isinstance(P, dict) and 'frames' in P:
        L = P['frames']
    elif isinstance(P, list):
        L = P
    else:
        L = []
    for e in L:
        name = e.get('image') or e.get('file') or e.get('name')
        if not name: continue
        try:
            Rwc, twc = pose_from_entry(e)
            yaw, pitch = rot_to_yaw_pitch(Rwc)
            out[name] = {'Rwc':Rwc, 'twc':twc, 'yaw':yaw, 'pitch':pitch}
        except Exception:
            continue
    return out

# ---------- bbox & overlays ----------

def bbox_from_mask(mask_path, size_wh=None):
    m = np.array(Image.open(mask_path).convert("L"))
    if size_wh is not None and (m.shape[1], m.shape[0]) != tuple(size_wh):
        m = cv2.resize(m, size_wh, interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

def draw_box(img: Image.Image, bbox, color=(255,0,0), text=None):
    draw = ImageDraw.Draw(img)
    x,y,w,h = bbox
    draw.rectangle([x,y,x+w,y+h], outline=color, width=3)
    if text:
        # simple text box
        tx, ty = x+4, max(0, y-16)
        draw.rectangle([tx-2, ty-2, tx+6*len(text), ty+14], fill=(0,0,0))
        draw.text((tx, ty), text, fill=(255,255,255))

# ---------- obstacle heuristics ----------

RISK_MAP = {
    # high risk (red)
    "stair": "high", "stairs":"high", "ladder":"high", "hole":"high",
    "cable":"high", "wire":"high", "cord":"high",
    "knife":"high", "glass":"high", "broken":"high",
    "spill":"high", "water":"high", "liquid":"high",
    "balcony":"high", "railing_missing":"high",

    # medium risk (yellow)
    "table":"med", "chair":"med", "sofa":"med", "couch":"med",
    "bench":"med", "box":"med", "plant":"med", "stool":"med",
    "toy":"med", "shoe":"med", "trash":"med", "bucket":"med",
    "threshold":"med", "mat":"med", "rug":"med", "step":"med",

    # low (green)
    "wall":"low", "door":"low", "window":"low", "desk":"low",
}

def risk_of(cat: str) -> str:
    c = (cat or "").lower()
    for k,v in RISK_MAP.items():
        if k in c:
            return v
    return "low"

def make_obstacle_notes(frame: Dict[str,Any]) -> List[Dict[str,Any]]:
    # frame['objects'] expected from story.json by build_view_story3d
    out = []
    objs = frame.get("objects") or []
    for o in objs:
        cat = o.get("category") or o.get("cat") or ""
        area = float(o.get("area", 0.0))
        bbox = o.get("bbox")
        risk = risk_of(cat)
        note = {
            "category": cat,
            "risk": risk,
            "area": area,
            "bbox": bbox,
            "conf": float(o.get("conf", 1.0)),
            "mask": o.get("mask"),
        }
        out.append(note)
    # simple saliency by (risk priority, area)
    pri = {"high":2, "med":1, "low":0}
    out.sort(key=lambda x: (pri[x["risk"]], x["area"]), reverse=True)
    return out

# ---------- links between selected frames ----------

def make_links(frames: List[Dict[str,Any]], poses: Dict[str,Any]) -> List[Dict[str,Any]]:
    links = []
    for i in range(len(frames)-1):
        a = frames[i]
        b = frames[i+1]
        na, nb = a["image"], b["image"]
        pa, pb = poses.get(na), poses.get(nb)
        link = {"from":na, "to":nb}
        if pa and pb and ("twc" in pa) and ("twc" in pb):
            da = np.asarray(pa["twc"], float).reshape(3)
            db = np.asarray(pb["twc"], float).reshape(3)
            dist = norm(db - da)
            link["translation_m"] = dist
            # yaw/pitch deltas if present
            if "yaw" in pa and "yaw" in pb:
                link["delta_yaw_deg"]   = wrap_deg(pb["yaw"] - pa["yaw"])
                link["delta_pitch_deg"] = wrap_deg(pb["pitch"] - pa["pitch"])
            # text
            parts = [f"translate {dist:.2f} m"]
            if "delta_yaw_deg" in link:
                parts.append(f"yaw {link['delta_yaw_deg']:+.0f}°")
            if "delta_pitch_deg" in link:
                parts.append(f"pitch {link['delta_pitch_deg']:+.0f}°")
            link["text"] = " ; ".join(parts)
        else:
            link["text"] = "move to next waypoint"
        links.append(link)
    return links

# ---------- main ----------

def main(args):
    story = load_json(args.story)
    frames = story.get("frames") or []
    if not frames:
        raise SystemExit("story.json has no frames[]")

    # optional poses
    poses = {}
    if args.poses and op.isfile(args.poses):
        try:
            poses = load_poses(args.poses)
            if not poses:
                warnings.warn("poses loaded but empty/unsupported; links will be generic.")
        except Exception as e:
            warnings.warn(f"failed to load poses: {e}")

    out_dir = args.out_dir or op.dirname(args.story)
    os.makedirs(out_dir, exist_ok=True)
    ann_dir = op.join(out_dir, "ann_obstacles")
    os.makedirs(ann_dir, exist_ok=True)

    # (i) links
    links = make_links(frames, poses)
    save_json({"links":links}, op.join(out_dir, "links.json"))
    # human readable
    t_lines = []
    for i,l in enumerate(links):
        t_lines.append(f"[{i:02d}] {l['from']} → {l['to']}: {l.get('text','')}")
    save_text("\n".join(t_lines) + "\n", op.join(out_dir, "links.md"))

    # (iii) obstacle analysis + overlays
    obstacles_all = []
    for f in frames:
        img = f["image"]
        fp  = op.join(args.frames, img)
        if not op.isfile(fp):
            continue
        im = Image.open(fp).convert("RGB")
        W,H = im.size
        notes = make_obstacle_notes(f)
        # fill bbox if missing by reading mask
        for n in notes:
            if not n.get("bbox") and n.get("mask") and args.mask_root:
                mp = op.join(args.mask_root, op.basename(str(n["mask"])))
                bb = bbox_from_mask(mp, (W,H)) if op.isfile(mp) else None
                n["bbox"] = bb
        # draw
        for n in notes[:12]:  # cap to avoid clutter
            bb = n.get("bbox")
            if not bb: continue
            col = {"high":(255,0,0), "med":(255,165,0), "low":(0,180,0)}[n["risk"]]
            label = f"{n.get('category','obj')} ({n['risk']})"
            draw_box(im, bb, color=col, text=label)
        im.save(op.join(ann_dir, op.basename(img)))

        obstacles_all.append({
            "image": img,
            "top_obstacles": notes[:10],
        })
    save_json({"frames":obstacles_all}, op.join(out_dir, "obstacles.json"))

    # human-readable obstacle md
    lines = ["# Obstacles (heuristic)\n"]
    for fr in obstacles_all:
        lines.append(f"## {fr['image']}")
        if not fr["top_obstacles"]:
            lines.append("- (none)")
            continue
        for o in fr["top_obstacles"]:
            lines.append(f"- **{o.get('category','obj')}** — risk: **{o['risk']}**, conf: {o.get('conf',1.0):.2f}, area: {o.get('area',0):.0f}")
        lines.append("")
    save_text("\n".join(lines), op.join(out_dir, "obstacles.md"))

    # (ii) mixed narration scaffold for LLM
    # combine link text + per-frame bullet notes into a single prompt
    narr = ["You are an expert indoor navigation assistant. "
            "Given a sequence of views, write (a) a concise scene overview covering major features and navigationally-relevant objects; "
            "(b) an obstacle analysis for pedestrian passage; "
            "Be precise but succinct. Use neutral tone.\n"]
    for i,f in enumerate(frames):
        narr.append(f"### View {i:02d} — {f['image']}")
        # per-frame quick bullets
        cats = [ (o.get("category","obj"), float(o.get("area",0))) for o in (f.get("objects") or []) ]
        cats.sort(key=lambda x: x[1], reverse=True)
        cats = [c for c,_ in cats[:8]]
        if cats:
            narr.append("- prominent objects: " + ", ".join(cats))
        narr.append(f"- coverage: {float(f.get('coverage',0.0)):.3f}")
        if i < len(links):
            narr.append(f"- transition to next: {links[i].get('text','move to next')}")
        narr.append("")
    narr.append("### Deliverables\n(a) Scene overview (4–6 sentences).\n(b) Obstacle analysis (bullet list, most critical first).")
    save_text("\n".join(narr), op.join(out_dir, "narrative_prompt.md"))

    print("[OK] links  ->", op.join(out_dir, "links.{json,md}"))
    print("[OK] obst   ->", op.join(out_dir, "obstacles.{json,md}"), "| ann ->", ann_dir)
    print("[OK] prompt ->", op.join(out_dir, "narrative_prompt.md"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--mask_root", default="")
    ap.add_argument("--poses", default="")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()
    main(args)
