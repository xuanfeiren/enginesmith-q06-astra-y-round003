import nki
import nki.isa as ni
from p_sbhint import _layer_sb

@nki.jit
def pf_group(plen,s_h,ln1,ln2,qn,kn,wqkv,wo,wgu,wd,rot,cosT,sinT,kcache,vcache):
    L=ln1.shape[0]
    for l in range(L):
        out=_layer_sb(plen,s_h,ln1[l:l+1],ln2[l:l+1],qn[l:l+1],kn[l:l+1],wqkv[l:l+1],wo[l:l+1],
                      wgu[l:l+1],wd[l:l+1],rot,cosT,sinT,kcache[l:l+1],vcache[l:l+1])
        if l+1<L:
            ni.core_barrier(data=s_h,cores=(0,1))
    return out
