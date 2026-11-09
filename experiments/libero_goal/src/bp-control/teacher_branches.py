"""Replay the exact stock grasp and validate local counterfactual action chunks in LIBERO."""
import argparse
from contextlib import nullcontext
from itertools import product
from pathlib import Path
import json
import os
import sys

import imageio.v2 as iio
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"multitask-repair"),str(ROOT)]
from matrix import public_observation, rim_contacts
from layouts import BowlOffsetEnv

DEV={
    "bottle":"../results/repair-series/replay-stock/historical-bottle",
    "bowl":"../results/repair-series/replay-stock/bowl-rim-dev",
}

def read(path):
    return json.loads(path.read_text())

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def reset(env,case):
    return env.reset(goal=case["goal"],init_index=case["init_index"],seed=case["seed"],
                     init_source=case.get("init_source","bowl"),layout=case["layout"])

def candidate(stock,start,dy,dz,dx=0.,span=20,first_only_z=False):
    """Move farther inward and delay descent before resuming recorded stock actions."""
    actions=np.array([r["action"] for r in stock],dtype=np.float64)
    actions[start:start+span,0]+=dx
    actions[start:start+span,1]=np.maximum(-1.,actions[start:start+span,1]+dy)
    z_stop=start+min(10,span) if first_only_z else start+span
    actions[start:z_stop,2]=np.minimum(1.,actions[start:z_stop,2]+dz)
    return actions

def replay(env,case,ref,stock,actions,start,video=None):
    obs=reset(env,case)
    assert np.array_equal(env.sim_state(),ref["sim_state"])
    if video:
        assert all(np.array_equal(v,ref[k]) for k,v in public_observation(obs).items())
    goal=case["goal"]
    other="bottle" if goal=="bowl" else "bowl"
    rows=[]; first_hit=None; streak=0; stable=None
    def frame(o):
        return np.ascontiguousarray(o["agentview_image"][::-1,::-1])
    with iio.get_writer(video,fps=20,codec="libx264") if video else nullcontext() as writer:
        if writer: writer.append_data(frame(obs))
        for t,a in enumerate(actions[:max(190,start+80)]):
            obs,_,done,_=env.step(a)
            s=env.status(); xyz=s["object_xyz_m"]
            held=s["grasped_objects"]
            if t<start:
                assert np.array_equal(a,stock[t]["action"]),t
                assert np.array_equal(xyz[goal],stock[t]["target"]),t
                assert np.array_equal(obs["robot0_eef_pos"],stock[t]["eef"]),t
                assert held==stock[t]["held"] and bool(done)==stock[t]["native"],t
            if t==start-1:
                assert goal in held, (case["id"],t+1)
            row={"step":t+1,"action":a.tolist(),"eef":np.asarray(obs["robot0_eef_pos"]).tolist(),
                 "target":xyz[goal],"other":xyz[other],"held":held,"native":bool(done),
                 "other_native":s[f"{other}_goal"]}
            if writer:
                row["cabinet_contacts"]=rim_contacts(env,goal)
                writer.append_data(frame(obs))
            rows.append(row)
            if done and first_hit is None: first_hit=t+1
            streak=streak+1 if done and goal not in held else 0
            if streak>=40:
                stable=t+1
                break
    return rows,first_hit,stable

def search_options(goal):
    # Put moderate, short changes before larger ones; branch at an actual replan.
    starts=(90,100,80) if goal=="bottle" else (100,110,90)
    values=[]
    for start,dy,dz,dx,span,zfirst in product(starts,(-.12,-.22,-.36,-.48),(.14,.23,.35,.55),
                                              (0.,-.08),(10,20),(True,False)):
        if span==10 and not zfirst: continue
        values.append((start,dy,dz,dx,span,zfirst))
    return sorted(values,key=lambda p:p[4]*(p[1]**2+p[2]**2+p[3]**2)
                  +.001*starts.index(p[0])+.01*(not p[5]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--render-gpu",type=int,default=0)
    p.add_argument("--goal",choices=("bottle","bowl"),required=True)
    p.add_argument("--max-trials",type=int,default=60)
    p.add_argument("--try-first",type=float,nargs=5,metavar=("T","DY","DZ","DX","SPAN"))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    g=args.goal; source=ROOT/DEV[g]
    case=read(source/"result.json"); stock=read(source/"trace.json")
    video=args.output/"rollout.mp4"
    traces=[]
    options=search_options(g)
    if args.try_first:
        t,dy,dz,dx,span=args.try_first
        options.insert(0,(int(t),dy,dz,dx,int(span),True))
    with np.load(source/"initial-public.npz",allow_pickle=False) as ref:
        with BowlOffsetEnv(pair="objects",camera_size=256,render=False) as env:
            for spec in options[:args.max_trials]:
                t,dy,dz,dx,span,zfirst=spec
                actions=candidate(stock,t,dy,dz,dx,span,zfirst)
                rows,hit,stable=replay(env,case,ref,stock,actions,t)
                traces.append({"params":list(spec),"first_native_goal":hit,"stable_released_at":stable})
                if stable:
                    print(g,"selected",spec,"first goal",hit,"stable",stable,flush=True)
                    break
                if len(traces)%10==0:
                    print(g,"tested",len(traces),"hits",sum(x["first_native_goal"] is not None for x in traces),flush=True)
            else:
                save(args.output/"search.json",traces)
                raise SystemExit(f"No stable {g} branch found in {len(traces)} attempts")
        save(args.output/"search.json",traces)
        with BowlOffsetEnv(pair="objects",camera_size=256,render_gpu_device_id=args.render_gpu) as env:
            filmed,hit2,stable2=replay(env,case,ref,stock,actions,t,video)
        assert hit==hit2 and stable==stable2 and [r["target"] for r in rows]==[r["target"] for r in filmed]
    save(args.output/"trace.json",filmed)
    # Include all modified ten-control windows; trace contains the unmodified tail and prefix.
    item={"case":case,"source":DEV[g],"branch_t":t,"windows":[{"t":k,"actions":[r["action"] for r in filmed[k:k+10]]}
              for k in range(t,t+span,10)],"first_native_goal":hit,"stable_released_at":stable,
          "physics_and_initial_observations_match_stock_prefix":True,
          "trace":os.path.relpath(args.output/"trace.json", ROOT),
          "video":os.path.relpath(video, ROOT)}
    save(args.output/"teacher.json",item)

if __name__=="__main__": main()
