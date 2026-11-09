import numpy as np

def residual(obs, base_action, hidden, memory):
    v=np.asarray(obs[26:]); out=np.zeros(19)
    if not memory:
        memory.update(refvy=float(v[1]),timer=0.0,n=0)
    if abs(v[1]-memory['refvy'])>0.9 and abs(v[3])>2.5 and memory['n']>30 and memory['timer']<=0:
        memory['targetvy']=memory['refvy']; memory['timer']=1.0
    if memory['timer']>0:
        g=min(1.0,memory['timer']/0.3)
        lateral=float(np.clip(-0.12*(v[1]-memory['targetvy']),-0.24,0.24))*g
        out[1]=lateral; out[6]=lateral
        memory['timer']-=0.04
    memory['refvy']+=0.12*(v[1]-memory['refvy']); memory['n']+=1
    return out
