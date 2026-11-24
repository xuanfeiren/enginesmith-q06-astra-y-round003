import os
import nki
import nki.language as nl
import nki.isa as ni
MODE=int(os.environ.get('NORM_MODE','0'))
@nki.jit
def norm(x, y):
    pid=nl.program_id(0)
    C=x.shape[-1]
    one=nl.ndarray((128,128),dtype=nl.bfloat16,buffer=nl.sbuf)
    ni.memset(dst=one,value=1.0)
    of=nl.ndarray((1,128),dtype=nl.float32,buffer=nl.sbuf)
    ni.memset(dst=of,value=1.0)
    for r in range(x.shape[1]):
        a=nl.ndarray((128,C),dtype=nl.bfloat16,buffer=nl.sbuf)
        ni.dma_copy(dst=a,src=x[pid,r])
        P=128
        if MODE:
            P=1
        ps=nl.ndarray((P,C),dtype=nl.float32,buffer=nl.psum)
        ni.nc_matmul(dst=ps,stationary=one[:,0:P],moving=a)
        rs=nl.ndarray((P,C),dtype=nl.float32,buffer=nl.sbuf)
        ni.activation(dst=rs,op=nl.rsqrt,data=ps,scale=1.0/128,bias=1e-6)
        if MODE:
            bc=nl.ndarray((128,C),dtype=nl.float32,buffer=nl.sbuf)
            if MODE==1:
                pp=nl.ndarray((128,C),dtype=nl.float32,buffer=nl.psum)
                ni.nc_matmul(dst=pp,stationary=of,moving=rs,is_stationary_onezero=True)
                ni.tensor_copy(dst=bc,src=pp)
            elif MODE==2:
                for q in range(4):
                    ni.nc_stream_shuffle(dst=bc[q*32:(q+1)*32,:],src=rs,shuffle_mask=[0]*32)
            else:
                ni.dma_copy(dst=bc,src=rs.ap(pattern=[[0,128],[1,C]],offset=0))
            ni.dma_copy(dst=y[pid,r],src=bc)
        else:
            ni.dma_copy(dst=y[pid,r],src=rs)
    z=nl.ndarray((1,1),dtype=nl.int32,buffer=nl.sbuf)
    ni.memset(dst=z,value=0)
    out=nl.ndarray((1,1),dtype=nl.int32,buffer=nl.shared_hbm)
    ni.dma_copy(dst=out,src=z)
    return out
