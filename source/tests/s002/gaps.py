import json,sys
d=json.load(open('ntff.json'))
ins=d['instruction']
t0=min(i['timestamp'] for i in ins)
g=[i for i in ins if i['label'] in(sys.argv[1],) and i['subgraph']=='sg01']
g.sort(key=lambda i:i['timestamp'])
prev=None; tot=0
for i in g:
    t=(i['timestamp']-t0)/1e3
    if prev is not None and t-prev>float(sys.argv[2]):
        tot+=t-prev
        if 150<t<float(sys.argv[3]): print(f"gap {prev:8.2f} -> {t:8.2f}  {i['opcode'][:16]} {i['operands'][:110]}")
    prev=t+i['duration']/1e3
print("total gap", tot)
