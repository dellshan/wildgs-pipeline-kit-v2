#!/usr/bin/env python3
import os, os.path as op, argparse, glob, re

def clamp01(x):
    return 0.0 if x != x else 1.0 if x>1 else 0.0 if x<0 else x  # 处理 NaN/越界

def parse_line(line):
    # 允许逗号/多空格/制表符
    toks = re.split(r'[,\s]+', line.strip())
    toks = [t for t in toks if t!='']
    if not toks: return None, None
    # 类别可能是浮点写法，转 int
    try:
        cls = int(float(toks[0]))
    except:
        return None, None
    nums=[]
    for t in toks[1:]:
        try:
            nums.append(float(t))
        except:
            # 非数值直接丢弃
            pass
    return cls, nums

def nums_to_bbox(nums):
    """
    nums 可能是:
    - 4 个数:  cx cy w h (已是 bbox)
    - >=6 且为偶数个: x1 y1 x2 y2 ... (分割多边形, 归一化)
    - 其它: 无效
    返回 (cx, cy, w, h) 或 None
    """
    if len(nums)==4:
        cx, cy, w, h = nums
        return clamp01(cx), clamp01(cy), max(1e-6, clamp01(w)), max(1e-6, clamp01(h))
    if len(nums)>=6:
        # 若为奇数个，尝试去掉最后一个“脏”值
        if len(nums)%2==1:
            nums = nums[:-1]
        if len(nums)<6 or len(nums)%2==1:
            return None
        xs = nums[0::2]
        ys = nums[1::2]
        # 若坐标是像素而非归一化，也能工作——最后 clamp 到 [0,1]
        xmin = min(xs); xmax = max(xs)
        ymin = min(ys); ymax = max(ys)
        w = max(1e-6, xmax - xmin)
        h = max(1e-6, ymax - ymin)
        cx = xmin + w/2.0
        cy = ymin + h/2.0
        # 统一 clamp 到 [0,1]，过界的直接截断
        return clamp01(cx), clamp01(cy), clamp01(w), clamp01(h)
    return None

def fix_file(p):
    changed = 0
    kept = 0
    lines_out = []
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read().splitlines()
    for ln in raw:
        cls, nums = parse_line(ln)
        if cls is None or nums is None:
            continue
        bb = nums_to_bbox(nums)
        if bb is None:
            # 丢弃无效行
            continue
        cx, cy, w, h = bb
        lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        # 如果原来不是 4 个数，或做了 clamp/裁剪，就算 changed
        if len(nums)!=4 or any((v<0 or v>1) for v in nums) or len(raw)!=len(lines_out):
            changed += 1
        kept += 1
    # 空文件也保留（负样本）
    with open(p, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines_out))
    return kept, changed, len(raw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_root", required=True, help="path to labels root (has train/, val/)")
    args = ap.parse_args()

    files = sorted(glob.glob(op.join(args.labels_root, "**/*.txt"), recursive=True))
    total_kept = total_changed = total_raw = 0
    for i, p in enumerate(files, 1):
        kept, changed, rawn = fix_file(p)
        total_kept += kept; total_changed += changed; total_raw += rawn
        if i % 500 == 0:
            print(f"[{i}/{len(files)}] kept={total_kept} changed={total_changed} raw={total_raw}")
    print(f"[DONE] files={len(files)} | kept_lines={total_kept} | changed_lines={total_changed} | raw_lines={total_raw}")

if __name__ == "__main__":
    main()
