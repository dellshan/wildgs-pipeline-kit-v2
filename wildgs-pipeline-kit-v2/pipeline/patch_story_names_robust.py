import json, os, os.path as op, argparse, re
from collections import defaultdict

def to_int(x):
    try: 
        return int(x)
    except Exception:
        try:
            # 可能是 "12" / "12.0" / numpy int 的 str
            return int(float(x))
        except Exception:
            return None

def basename_or_none(p):
    if not p: return None
    s=str(p)
    return op.basename(s)

def extract_sem_tuples(sem):
    """
    返回三种索引：
    A: (image_basename, inst_id_int) -> name
    B: (image_basename, mask_basename) -> name
    C: mask_basename -> name   （兜底）
    """
    A, B, C = {}, {}, {}
    def putA(img, inst, name):
        if img and inst is not None and name:
            A[(img, inst)] = str(name)
    def putB(img, msk, name):
        if img and msk and name:
            B[(img, msk)] = str(name)
    def putC(msk, name):
        if msk and name:
            C[msk] = str(name)

    def norm_image_key(v):
        # 从绝对/相对路径里取 basename
        if not v: return None
        b = op.basename(str(v))
        return b

    def visit_obj(img_ctx, o):
        if not isinstance(o, dict): return
        imgB = norm_image_key(o.get('image') or o.get('file') or o.get('frame') or img_ctx)
        # 候选名称字段
        name = o.get('name') or o.get('label') or o.get('category') or o.get('class')
        # 候选实例 id 字段
        inst = o.get('inst') or o.get('instance') or o.get('id') or o.get('instance_id') or o.get('track_id')
        inst = to_int(inst)
        # 候选 mask 字段
        mask = o.get('mask') or o.get('mask_path') or o.get('mask_file') or o.get('path')
        maskB = basename_or_none(mask)

        if name:
            if inst is not None and imgB:
                putA(imgB, inst, name)
            if imgB and maskB:
                putB(imgB, maskB, name)
            if maskB:
                putC(maskB, name)

    def walk(x, img_ctx=None):
        if isinstance(x, dict):
            # 优先识别 frames/by_image 等容器
            if 'frames' in x and isinstance(x['frames'], list):
                for fr in x['frames']:
                    img_in = fr.get('image') or fr.get('file') or fr.get('name') or img_ctx
                    imgB = basename_or_none(img_in)
                    visit_obj(imgB, fr)
                    for o in fr.get('objects', fr.get('instances', [])):
                        visit_obj(imgB, o)
                return
            if 'by_image' in x and isinstance(x['by_image'], dict):
                for img_in, lst in x['by_image'].items():
                    imgB = basename_or_none(img_in)
                    if isinstance(lst, list):
                        for o in lst:
                            visit_obj(imgB, o)
                return
            # 普通 dict：遍历
            for k,v in x.items():
                # 如果这是一个 object
                if isinstance(v, (dict, list)):
                    walk(v, img_ctx)
                else:
                    pass
            # 也把自身当成一个 obj 看一眼
            visit_obj(img_ctx, x)
        elif isinstance(x, list):
            for v in x:
                walk(v, img_ctx)

    walk(sem, img_ctx=None)
    return A, B, C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--story', required=True)
    ap.add_argument('--semantics', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    story = json.load(open(args.story,'r'))
    sem   = json.load(open(args.semantics,'r'))

    A,B,C = extract_sem_tuples(sem)

    filled = 0
    triedA = triedB = triedC = 0
    for fr in story.get('frames', []):
        imgB = basename_or_none(fr.get('image'))
        for o in fr.get('objects', []):
            # 已有名字则跳过
            if o.get('name') or o.get('label') or o.get('category') or o.get('class'):
                continue
            # story 自带 inst 或 mask?
            inst = o.get('inst') or o.get('instance') or o.get('id') or o.get('instance_id')
            inst = to_int(inst)
            maskB = basename_or_none(o.get('mask'))

            # 策略 A: (image, inst)
            if imgB and inst is not None:
                triedA += 1
                name = A.get((imgB, inst))
                if name:
                    o['name'] = name
                    filled += 1
                    continue
            # 策略 B: (image, maskB)
            if imgB and maskB:
                triedB += 1
                name = B.get((imgB, maskB))
                if name:
                    o['name'] = name
                    filled += 1
                    continue
            # 策略 C: maskB 兜底
            if maskB:
                triedC += 1
                name = C.get(maskB)
                if name:
                    o['name'] = name
                    filled += 1
                    continue

    json.dump(story, open(args.out,'w'), ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out}; filled={filled} (tried A:{triedA} B:{triedB} C:{triedC})  |  sem_sizes: A={len(A)} B={len(B)} C={len(C)}")

if __name__=='__main__':
    main()
