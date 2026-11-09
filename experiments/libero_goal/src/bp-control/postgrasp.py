"""Diagnostic: let each locked actor take over at the identical recorded stock grasp."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sequential-repair"))
from stage import assigned, BowlOffsetEnv, SmolActor, head, read, save
from matrix import public_observation

def replay_and_run(env, actor, item, arm, edits):
    case=item["case"]; source=ROOT/item["source"]
    prefix=read(source/"trace.json")
    t0=item["branch_t"]; g=case["goal"]
    obs=public_observation(env.reset(goal=g,init_index=case["init_index"],seed=case["seed"],
                          init_source=case.get("init_source","bowl"),layout=case["layout"]))
    with np.load(source/"initial-public.npz",allow_pickle=False) as ref:
        assert np.array_equal(env.sim_state(),ref["sim_state"])
        assert all(np.array_equal(v,ref[k]) for k,v in obs.items())
    actor.reset(case["seed"])
    for t in range(t0):
        if t%10==0: actor.make_noise()
        row=prefix[t]; raw,_,done,_=env.step(row["action"])
        obs=public_observation(raw); s=env.status()
        assert np.array_equal(s["object_xyz_m"][g],row["target"]) and bool(done)==row["native"], t
        assert s["grasped_objects"]==row["held"] and np.array_equal(obs["robot0_eef_pos"],row["eef"]),t
    assert t0%10==0 and g in env.status()["grasped_objects"]
    rows=[]; hit=stable=None; streak=0
    with assigned(head(actor),edits):
        for step in range(t0+1,381):
            if step>300 and hit is None: break
            if hit is not None and step>hit+80: break
            a=actor.act(obs,case["instruction"])
            if arm=="stock" and step<=len(prefix):
                assert np.allclose(a,prefix[step-1]["action"],atol=1e-6,rtol=0),(g,step)
            raw,_,done,_=env.step(a); obs=public_observation(raw); s=env.status()
            held=g in s["grasped_objects"]
            if done and hit is None and step<=300: hit=step
            streak=streak+1 if done and not held else 0
            if stable is None and streak>=40: stable=step
            rows.append({"step":step,"action":a.tolist(),"xyz":s["object_xyz_m"][g],"held":held,"native":bool(done)})
    return {"arm":arm,"goal":g,"takeover_after":t0,"first_native_goal":hit,
            "stable_released_at":stable,"final_native":rows[-1]["native"],"trace":rows}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    args=p.parse_args()
    items=read(args.run/"teachers.json")
    actor=SmolActor(n_action_steps=10)
    summary=[]
    with BowlOffsetEnv(pair="objects",camera_size=256,render_gpu_device_id=0) as env:
        for arm in ("stock","bp","wsp"):
            edits=read(args.run/f"{arm}.json")["edits"] if arm!="stock" else []
            for item in items.values():
                r=replay_and_run(env,actor,item,arm,edits)
                save(args.run/f"postgrasp-{r['goal']}-{arm}.json",r)
                summary.append({k:v for k,v in r.items() if k!="trace"})
                print(summary[-1],flush=True)
    save(args.run/"postgrasp-summary.json",summary)

if __name__=="__main__":
    main()
