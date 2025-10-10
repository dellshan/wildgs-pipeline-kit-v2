import os, os.path as op, json, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument('--story', required=True)
ap.add_argument('--links', required=True)
ap.add_argument('--obstacles', required=True)  # 可用 obstacles_recomputed.json
ap.add_argument('--out', required=True)
args = ap.parse_args()

story = json.load(open(args.story,'r'))
links = json.load(open(args.links,'r'))
obst  = json.load(open(args.obstacles,'r'))

# (iii-a) 统计全局出现频次 + 覆盖度加权
freq = collections.Counter()
for fr in story.get('frames', []):
    cov = float(fr.get('coverage',0.0))
    for o in fr.get('objects', []):
        n = (o.get('name') or o.get('label') or o.get('category') or 'object').lower()
        freq[n] += 1 + cov

top_objs = [n for n,_ in freq.most_common(8)]

# (iii-b) 障碍物排行（按 score/area_frac）
danger = []
for fr in obst.get('frames', []):
    for o in fr.get('obstacles', []):
        danger.append((fr['image'], o.get('name','object'), float(o.get('score',0.0)), float(o.get('area_frac',0.0))))
danger.sort(key=lambda x: (-x[2], -x[3]))
danger = danger[:10]

lines=[]
lines += ["# (iii-a) Scene Overview (rule-based, draft)"]
lines += [f"- Dominant objects: {', '.join(top_objs) or 'N/A'}"]
lines += [f"- Frames: {len(story.get('frames',[]))}; Transitions: {len(links.get('links',[]))}"]
lines += ["- Navigation: follow the annotated arrows; distances are per-link in meters."]

lines += ["\n# (iii-b) Obstacles (rule-based, draft) — top risks"]
for img, name, sc, frac in danger:
    lines += [f"- {name} @ {img}: score={sc:.3f}, area={frac*100:.1f}%"]

open(args.out,'w',encoding='utf-8').write("\n".join(lines))
print(f"[OK] wrote {args.out}")
