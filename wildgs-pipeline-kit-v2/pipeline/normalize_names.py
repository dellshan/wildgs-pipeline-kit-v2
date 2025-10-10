import json, re, sys, os.path as op
mp = {
  "人":"person","两个人":"two people","盒子":"box","垃圾桶":"trash can",
  "办公室":"office","仓库":"warehouse","looks":None,"a":None
}
inp, outp = sys.argv[1], sys.argv[2]
J=json.load(open(inp,'r'))
for fr in J.get('frames',[]):
  for o in fr.get('objects',[]):
    n=(o.get('name') or o.get('label') or o.get('category'))
    if not n: continue
    n = str(n).strip()
    n = re.sub(r'[^a-zA-Z\u4e00-\u9fa5\s_-]','',n)  # 去掉奇怪符号
    if n in mp:
      n = mp[n]
    if not n: 
      o.pop('name',None); o.pop('label',None); o.pop('category',None); continue
    o['name']=n
json.dump(J, open(outp,'w'), ensure_ascii=False, indent=2)
print("[OK] normalized ->", outp)
