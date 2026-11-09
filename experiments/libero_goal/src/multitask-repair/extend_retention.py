"""Recheck a positive-but-too-short ungrasped tail from the same fixed policy."""
import argparse
from pathlib import Path
import imageio.v2 as iio
import numpy as np
from matrix import SmolActor, edit_head, public_observation, PROMPT, read, save
from layouts import BowlOffsetEnv

def one(actor,env,folder,out):
    result=read(folder/"result.json"); trace=read(folder/"trace.json")
    if not (result["native_success"] and result["final_native"] and not result["stable_released_40"]):
        raise ValueError("Extend only native-positive end states with an incomplete 40-step release window")
    out.mkdir()
    arm=folder.parent
    edits=[{k:e[k] for k in ("row","column","delta")} for e in read(arm/"weight_edits.json")]
    g=result["goal"]; n=len(trace); chunk_start=(n//10)*10
    history=result.get("prompt_mode")=="historical_bottle"
    tail=[]; streak=0; hit=None
    with edit_head(actor,edits), iio.get_writer(out/"rollout.mp4",fps=20,codec="libx264") as video:
        obs=public_observation(env.reset(goal=g,init_index=result["init_index"],seed=result["seed"],
            init_source=result.get("init_source","bowl"),layout=result["layout"]))
        with np.load(folder/"initial-public.npz",allow_pickle=False) as ref:
            assert np.array_equal(env.sim_state(),ref["sim_state"]) and all(np.array_equal(v,ref[k]) for k,v in obs.items())
        actor.reset(result["seed"])
        for t in range(n+40):
            instruction=PROMPT if history and t<80 else env.language
            if t<chunk_start:
                if t%10==0: actor.make_noise()
                a=trace[t]["action"]
            else:
                a=actor.act(obs,instruction)
                if t<n:
                    assert np.array_equal(a,trace[t]["action"]), (folder,t)
            raw,_,done,_=env.step(a); obs=public_observation(raw)
            status=env.status(); held=g in status["grasped_objects"]; xyz=status["object_xyz_m"][g]
            if t<n:
                assert bool(done)==trace[t]["native"] and np.array_equal(xyz,trace[t]["target"]) and status["grasped_objects"]==trace[t]["held"], (folder,t)
            if t>=max(0,result["first_native_goal"]-13):
                video.append_data(np.ascontiguousarray(obs["agentview_image"][::-1,::-1]))
            streak=streak+1 if done and not held else 0
            if t>=n:
                tail.append({"step":t+1,"native":bool(done),"held":held,"target":xyz,"released_streak":streak})
                if streak>=40 and hit is None: hit=t+1
    r={"case":result["id"],"arm":arm.name,"source":str(folder),"original_last_step":n,
       "first_stable_released_40":hit,"verified_stable40":hit is not None,
       "final_native":tail[-1]["native"],"prefix_exact":True,"tail":tail}
    save(out/"result.json",r)
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--render-gpu",type=int,default=5)
    p.add_argument("folders",type=Path,nargs="+")
    a=p.parse_args(); a.out.mkdir(); results=[]
    actor=SmolActor(n_action_steps=10)
    with BowlOffsetEnv(pair="objects",camera_size=256,render_gpu_device_id=a.render_gpu) as env:
        for src in a.folders:
            r=one(actor,env,src,a.out/(src.parent.name+"--"+src.name))
            results.append(r)
            save(a.out/"summary.json",results)
            print(r["arm"],r["case"],r["first_stable_released_40"],r["final_native"],flush=True)
if __name__=="__main__": main()
