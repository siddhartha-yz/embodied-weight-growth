"""Render the existing matched-grasp diagnostic, without running any policy again."""
import argparse
import shutil
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sequential-repair"))
from stage import BowlOffsetEnv, read, save
from matrix import public_observation

def render(env, run, item, arm):
    case=item["case"]; goal=case["goal"]; source=ROOT/item["source"]
    continuation=read(run/f"postgrasp-{goal}-{arm}.json")
    t0=continuation["takeover_after"]
    expected=read(source/"trace.json")[:t0]+continuation["trace"]
    out=run/f"postgrasp-video-{goal}"/arm
    out.mkdir(parents=True)
    obs=env.reset(goal=goal,init_index=case["init_index"],seed=case["seed"],
                  init_source=case.get("init_source","bowl"),layout=case["layout"])
    with np.load(source/"initial-public.npz",allow_pickle=False) as ref:
        assert np.array_equal(env.sim_state(),ref["sim_state"])
        assert all(np.array_equal(v,ref[k]) for k,v in public_observation(obs).items())
    shutil.copyfile(source/"initial-public.npz",out/"initial-public.npz")
    rows=[]
    with iio.get_writer(out/"rollout.mp4",fps=20,codec="libx264") as writer:
        frame=lambda x: np.ascontiguousarray(x["agentview_image"][::-1,::-1])
        writer.append_data(frame(obs))
        for row in expected:
            obs,_,done,_=env.step(row["action"])
            s=env.status(); held=s["grasped_objects"]
            xyz=s["object_xyz_m"][goal]
            assert bool(done)==row["native"] and np.array_equal(xyz,row.get("xyz",row.get("target"))), (arm,row["step"])
            rows.append({"step":row["step"],"held":held,"native":bool(done)})
            writer.append_data(frame(obs))
    hit=continuation["first_native_goal"]
    result=case|{"steps":len(rows),"first_native_goal":hit,"native_success":hit is not None,
                 "stable_released_at":continuation["stable_released_at"],"takeover_after":t0}
    save(out/"trace.json",rows); save(out/"result.json",result)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--goal",choices=("bottle","bowl"),default="bottle")
    a=p.parse_args()
    item=read(a.run/"teachers.json")[a.goal]
    with BowlOffsetEnv(pair="objects",camera_size=256,render_gpu_device_id=0) as env:
        for arm in ("stock","bp","wsp"): render(env,a.run,item,arm)
