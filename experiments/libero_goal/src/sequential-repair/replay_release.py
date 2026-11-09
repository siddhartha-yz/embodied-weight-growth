"""Probe the verified successful v2 bowl carry/release, even without a two-finger lift flag."""
import argparse
from pathlib import Path

import numpy as np
from stage import assigned, BowlOffsetEnv, head, probes, read, save, SmolActor
from matrix import Observer, public_observation

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("episode",type=Path)
p.add_argument("edits",type=Path)
p.add_argument("output",type=Path)
a=p.parse_args()
result=read(a.episode/"result.json")
trace=read(a.episode/"trace.json")
assert result["goal"]=="bowl" and result["stable_released_40"]
chunk=(result["first_native_goal"]-1)//10*10
ticks=(chunk-30,chunk-10,chunk)
edits=read(a.edits)["edits"]
actor=SmolActor(n_action_steps=10); layer=head(actor)
stock=[{**e,"value":float(layer.weight[e["row"],e["column"]])} for e in edits]
with assigned(layer,edits):
    with BowlOffsetEnv(pair="objects",camera_size=256,render_gpu_device_id=0) as env:
        obs=public_observation(env.reset(goal="bowl",init_index=result["init_index"],seed=result["seed"],
                init_source=result["init_source"],layout=result["layout"]))
        with np.load(a.episode/"initial-public.npz",allow_pickle=False) as ref:
            assert np.array_equal(env.sim_state(),ref["sim_state"])
            assert all(np.array_equal(v,ref[k]) for k,v in obs.items())
        actor.reset(result["seed"])
        observer=Observer(actor,True)
        for t,row in enumerate(trace[:max(ticks)+1]):
            action=observer.act(obs,result["instruction"],t)
            assert np.allclose(action,row["action"],atol=1e-6,rtol=0),(t,action,row["action"])
            raw,_,done,_=env.step(row["action"])
            obs=public_observation(raw)
            assert np.array_equal(env.status()["object_xyz_m"]["bowl"],row["target"]) and bool(done)==row["native"],t
        obs={f"bowl/fault/successful_carry_{t}":(t,*observer.caches[t]) for t in ticks}
        out=probes(actor,obs,edits,stock)
        # These are supplemental measurements for weights already edited or tested;
        # stage 3 still chooses coordinates only from its main current-model table.
        out["replay_matches_recorded_actions_and_physics"]=True
        out["note"]="Successful bowl transport and cabinet release despite no two-finger sustained lift flag."
        save(a.output,out)
