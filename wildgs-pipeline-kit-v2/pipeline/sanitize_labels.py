#!/usr/bin/env python3
import json, argparse, re, collections, os.path as op

# Chinese → English (extend as needed)
CN2EN = {
    "机房":"server room","会议室":"meeting room","仓库":"warehouse","梯子":"ladder",
    "桌子":"desk","椅子":"chair","垃圾桶":"trash can","实验室":"laboratory",
    "门":"door","箱子":"box","两个人":"two people","人":"person","服务器机柜":"server rack"
}

# Unify English synonyms / fix noise
SYN = {
    "trash bin":"trash can","garbage can":"trash can","rubbish bin":"trash can","bin":"trash can",
    "cab":"cabinet","cupboard":"cabinet","workbench":"table","table desk":"desk",
    "like":None,"looks":None,"present":None  # remove meaningless tokens
}

ASCII = re.compile(r"[^\x00-\x7F]+")
def to_ascii(s:str) -> str:
    s = s.lower().strip().replace("_"," ")
    s = re.sub(r"—|–","-",s)
    s = re.sub(r"[^0-9a-z \-/]", "", s)   # keep letters, digits, space, -, /
    s = re.sub(r"\s+"," ",s).strip()
    return s

def normalize(name):
    if not name: return None
    n = str(name).strip()
    if n in CN2EN: n = CN2EN[n]
    n = to_ascii(n)
    if n in SYN:
        n = SYN[n] if SYN[n] is not None else ""
    if not n or ASCII.search(n):   # must be pure ASCII
        return None
    return n

def process_story(J):
    for fr in J.get("frames", []):
        for o in fr.get("objects", []):
            n = normalize(o.get("name") or o.get("label") or o.get("category"))
            if n: o["name"] = n
            else:
                for k in ("name","label","category"): o.pop(k, None)
    return J

def process_obst(O):
    for fr in O.get("frames", []):
        for o in fr.get("obstacles", []):
            n = normalize(o.get("name"))
            o["name"] = n or "object"
    return O

def summary(J):
    c = collections.Counter()
    for fr in J.get("frames", []):
        for o in fr.get("objects", []):
            n=o.get("name"); 
            if n: c[n]+=1
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story_in", required=True)
    ap.add_argument("--obst_in",  required=True)
    ap.add_argument("--story_out", required=True)
    ap.add_argument("--obst_out",  required=True)
    args = ap.parse_args()

    S = json.load(open(args.story_in, "r", encoding="utf-8"))
    O = json.load(open(args.obst_in,  "r", encoding="utf-8"))

    S = process_story(S)
    O = process_obst(O)

    json.dump(S, open(args.story_out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(O, open(args.obst_out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    print("[OK] wrote", args.story_out, "and", args.obst_out)
    print("Top labels:", summary(S).most_common(10))

if __name__ == "__main__":
    main()
