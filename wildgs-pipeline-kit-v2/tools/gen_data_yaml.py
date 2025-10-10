#!/usr/bin/env python3
import os, os.path as op, argparse, yaml
def read_names(p): return [ln.strip() for ln in open(p) if ln.strip() and not ln.lstrip().startswith("#")]
ap=argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--names", required=True)
ap.add_argument("--out", default=None)
a=ap.parse_args()
names=read_names(a.names)
data={"path":a.root,"train":"images/train","val":"images/val","test":"images/test","nc":len(names),"names":names}
out=a.out or op.join(a.root,"data.yaml")
os.makedirs(op.dirname(out), exist_ok=True)
with open(out,"w") as f: yaml.safe_dump(data,f,sort_keys=False)
print("[OK] wrote", out)
