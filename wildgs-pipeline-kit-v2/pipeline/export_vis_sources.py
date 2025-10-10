#!/usr/bin/env python3
import os, os.path as op, json, argparse, csv, zipfile, shutil, collections

def jload(p):
    with open(p,'r',encoding='utf-8') as f: return json.load(f)

def ensure(d): os.makedirs(d, exist_ok=True)

def write_csv(path, rows, header):
    ensure(op.dirname(path))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def dump_per_link(links, outdir):
    rows=[]
    for i,l in enumerate(links):
        rows.append([i,l.get('from'),l.get('to'),
                     l.get('dx',0),l.get('dy',0),l.get('dz',0),
                     l.get('dist_m',0),l.get('yaw_deg',0),l.get('text','')])
    write_csv(op.join(outdir,'per_link.csv'),
              rows, ['idx','from','to','dx','dy','dz','dist_m','yaw_deg','text'])
    write_csv(op.join(outdir,'per_link_distance.csv'),
              [[i,l.get('dist_m',0)] for i,l in enumerate(links)],
              ['idx','dist_m'])
    write_csv(op.join(outdir,'per_link_yaw_deg.csv'),
              [[i,l.get('yaw_deg',0)] for i,l in enumerate(links)],
              ['idx','yaw_deg'])

def dump_coverage(story, outdir):
    rows=[]
    for i,fr in enumerate(story.get('frames',[])):
        rows.append([i, fr.get('image'), float(fr.get('coverage',0.0))])
    write_csv(op.join(outdir,'coverage_per_view.csv'), rows, ['view_idx','image','coverage'])

def dump_objects(story, outdir, topk=100):
    rows=[]
    for fr in story.get('frames',[]):
        img = fr.get('image')
        for o in fr.get('objects',[]):
            name = (o.get('name') or o.get('label') or o.get('category') or '').strip()
            area = o.get('area') or o.get('pix') or o.get('area_px') or 0
            rows.append([img, name, area])
    write_csv(op.join(outdir,'objects_all.csv'), rows, ['image','name','area_like'])

    cnt = collections.Counter()
    cov_by_img = {fr.get('image'): float(fr.get('coverage',0.0)) for fr in story.get('frames',[])}
    for img,name,_ in rows:
        if not name: continue
        cnt[name.lower()] += 1 + cov_by_img.get(img,0.0)
    top = cnt.most_common(topk)
    write_csv(op.join(outdir,'top_objects.csv'),
              [[n,f'{v:.3f}'] for n,v in top],
              ['object','weighted_freq'])

def dump_obstacles(obst, outdir, topk=100):
    items=[]
    for fr in obst.get('frames',[]):
        img=fr.get('image')
        for o in fr.get('obstacles',[]):
            items.append([img, o.get('name','object'),
                          float(o.get('score',0.0)), float(o.get('area_frac',0.0))])
    write_csv(op.join(outdir,'obstacles_all.csv'),
              items, ['image','name','score','area_frac'])
    items.sort(key=lambda x:(-x[2], -x[3]))
    write_csv(op.join(outdir,'top_obstacles.csv'),
              items[:topk], ['image','name','score','area_frac'])

def copy_inputs_and_code(story, links, obst, extra, outdir):
    raw_dir = op.join(outdir,'raw_inputs'); ensure(raw_dir)
    for p in [story, links, obst] + (extra or []):
        if p and op.isfile(p): shutil.copy2(p, raw_dir)
    code_dir = op.join(outdir,'code'); ensure(code_dir)
    for p in [
        'pipeline/viz_pacific.py',
        'pipeline/viz_pacific_plus.py',
        'pipeline/make_links_from_views_csv.py',
        'pipeline/poses_json_to_views_csv.py',
        'pipeline/recompute_obstacles.py',
        'pipeline/summarize_scene_rule_based.py',
        'pipeline/normalize_names.py',
    ]:
        if op.isfile(p): shutil.copy2(p, code_dir)

def make_zip(outdir):
    zpath = outdir.rstrip('/').rstrip('\\') + '.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root,_,files in os.walk(outdir):
            for fn in files:
                p = op.join(root, fn)
                zf.write(p, op.relpath(p, op.dirname(outdir)))
    print('[OK] zipped ->', zpath)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--story', required=True)
    ap.add_argument('--links', required=True)
    ap.add_argument('--obstacles', required=True)
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--outdir', default='output/vis_sources')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    story = jload(args.story)
    links = jload(args.links).get('links',[])
    obstacles = jload(args.obstacles)

    dump_per_link(links, args.outdir)
    dump_coverage(story, args.outdir)
    dump_objects(story, args.outdir)
    dump_obstacles(obstacles, args.outdir)
    copy_inputs_and_code(args.story, args.links, args.obstacles, args.extra, args.outdir)
    make_zip(args.outdir)
    print('[OK] VIS sources ready at', args.outdir)

if __name__ == '__main__':
    main()
