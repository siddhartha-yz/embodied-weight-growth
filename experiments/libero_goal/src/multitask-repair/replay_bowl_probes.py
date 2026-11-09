"""Recover exact stock observation/noise pairs from already-recorded trajectories."""
from pathlib import Path
import argparse
import numpy as np
import torch
from matrix import SmolActor, head, probes, read, save, public_observation
from layouts import BowlOffsetEnv

def replay(actor, env, folder, fault):
    row=read(folder/"result.json"); trace=read(folder/"trace.json")
    actor.reset(row["seed"])
    obs=public_observation(env.reset(goal="bowl",init_index=row["init_index"],seed=row["seed"],
                                     init_source=row["init_source"],layout=row["layout"]))
    with np.load(folder/"initial-public.npz",allow_pickle=False) as ref:
        assert np.array_equal(env.sim_state(),ref["sim_state"])
        assert all(np.array_equal(v,ref[k]) for k,v in obs.items())
    lift=((row["first_lift_step"]-1)//10)*10
    contact=((row["last_contact"]-1)//10)*10 if fault else ((row["first_native_goal"]-1)//10)*10
    ticks={max(10 if fault else 0,lift-10),max(0,contact-10),contact}
    cache={}
    for t in range(max(ticks)+1):
        if t%10==0:
            noise=actor.make_noise()
            if t in ticks:
                batch=actor.prepare(obs,row["instruction"])
                activ=[]
                with torch.inference_mode():
                    hook=head(actor).register_forward_pre_hook(lambda _,args:activ.append(args[0][:,:10].clone()))
                    try:
                        actual=actor.predict_action_chunk(batch,noise=noise.clone())[0,:10].cpu().numpy()
                    finally: hook.remove()
                expected=np.asarray([x["action"] for x in trace[t:t+10]])
                assert np.allclose(actual[:len(expected)],expected,atol=1e-6,rtol=0), (row["id"],t,float(np.max(np.abs(actual[:len(expected)]-expected))))
                cache[t]=(batch,noise,actual,torch.cat(activ).float().cpu().numpy())
        step=trace[t]
        raw,_,done,_=env.step(step["action"]); obs=public_observation(raw)
        status=env.status()
        assert np.array_equal(status["object_xyz_m"]["bowl"],step["target"]) and bool(done)==step["native"], (row["id"],t)
    return row,cache,trace

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--render-gpu",type=int,default=5)
    a=p.parse_args()
    actor=SmolActor(n_action_steps=10)
    with BowlOffsetEnv(pair="objects",camera_size=256,render_gpu_device_id=a.render_gpu) as env:
        f=replay(actor,env,a.run/"stock/bowl-rim-dev",True)
        c=replay(actor,env,a.run/"stock/clean-bowl-2",False)
    print("Exact recorded action and image/noise replay matched at all six stock replan contexts",flush=True)
    result=probes(actor,f,c)
    save(a.run/"bowl_probes.json",result)
if __name__=="__main__": main()
