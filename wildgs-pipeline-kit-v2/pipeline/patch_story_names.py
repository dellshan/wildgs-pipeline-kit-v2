import json, os.path as op, argparse
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--story', required=True)
    ap.add_argument('--semantics', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()

    story=json.load(open(args.story,'r'))
    sem  =json.load(open(args.semantics,'r'))

    # 构造 (image, mask_basename) -> label
    m = {}
    def add(img, mask, label):
        if not mask or not label: return
        b=op.basename(str(mask))
        m.setdefault(img, {})[b]=str(label)

    if isinstance(sem, dict):
        if 'frames' in sem:
            for fr in sem['frames']:
                img = fr.get('image') or fr.get('file') or fr.get('name')
                for o in fr.get('objects', fr.get('instances', [])):
                    lab = o.get('name') or o.get('label') or o.get('category') or o.get('class')
                    mk  = o.get('mask') or o.get('mask_path') or o.get('mask_file') or o.get('path')
                    add(img, mk, lab)
        elif 'by_image' in sem:
            for img, lst in sem['by_image'].items():
                for o in lst:
                    lab = o.get('name') or o.get('label') or o.get('category') or o.get('class')
                    mk  = o.get('mask') or o.get('mask_path') or o.get('mask_file') or o.get('path')
                    add(img, mk, lab)
        elif 'mask_to_name' in sem:  # 兜底：不区分 image
            for mk, lab in sem['mask_to_name'].items():
                add('*', mk, lab)

    filled=0
    for fr in story.get('frames', []):
        img = fr.get('image')
        for o in fr.get('objects', []):
            if o.get('name'): continue
            b = op.basename(str(o.get('mask','')))
            lab = (m.get(img, {}) or {}).get(b) or (m.get('*', {}) or {}).get(b)
            if lab:
                o['name'] = lab
                filled += 1

    json.dump(story, open(args.out,'w'), ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out}; filled names for {filled} objects")
if __name__=='__main__':
    main()
