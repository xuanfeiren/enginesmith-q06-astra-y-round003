import json,collections,sys
SG=sys.argv[5] if len(sys.argv)>5 else 'sg01'
d=json.load(open(sys.argv[1])); B=float(sys.argv[2]); t1=float(sys.argv[3]); t2=float(sys.argv[4])
ins=[i for i in d['instruction'] if i['subgraph']==SG]; dma=[x for x in d['dma'] if x['subgraph']==SG]
t0=min([i['timestamp'] for i in d['instruction']]+[x['timestamp'] for x in d['dma']])
nb=int((t2-t1)/B)+1
def acc(ev,key):
    a=[0.0]*nb
    for e in ev:
        s=(e['timestamp']-t0)/1e3; du=e['duration']/1e3
        if s+du<t1 or s>t2: continue
        v=key(e)
        # spread across buckets
        x=s
        while x<s+du:
            b=int((x-t1)/B); be=t1+(b+1)*B; seg=min(be,s+du)-x
            if 0<=b<nb: a[b]+=v*seg/du if du>0 else v
            x=be if du>0 else s+du+1
    return a
dm=acc(dma,lambda e:e['transfer_size'])
eng={}
for lab in ['Tensor','TensorMatrix','Scalar','Vector','GpSimd']:
    eng[lab]=acc([i for i in ins if i['label']==lab],lambda e:e['duration']/1e3)
print("t(us)   GB/s  PE%  PEm%  ACT%  DVE%  POOL%")
for b in range(nb):
    print(f"{t1+b*B:7.1f} {dm[b]/B/1e3:6.0f} "+" ".join(f"{eng[l][b]/B*100:4.0f}" for l in ['Tensor','TensorMatrix','Scalar','Vector','GpSimd']))
