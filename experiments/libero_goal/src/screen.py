"""Screen a paired LIBERO-Goal scene before attempting any policy repair."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np

from paired_env import PairedEnv

GOALS = ("bowl", "bottle")
PUBLIC = ("agentview_image", "robot0_eye_in_hand_image",
          "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def public_observation(raw):
    return {k: np.asarray(raw[k]).copy() for k in PUBLIC}


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value,np.generic):
        return value.item()
    return value


def save(path, data):
    path.write_text(json.dumps(jsonable(data),ensure_ascii=False,allow_nan=False,indent=2)+"\n")


def run_episode(env, actor, goal, init_index, seed, max_steps, output, *, video=False, init_source=None, layout="canonical"):
    out = output / f"init-{init_index:03d}-{goal}"
    out.mkdir()
    kwargs = {"layout":layout}
    if init_source is not None:
        kwargs["init_source"] = init_source
    obs = public_observation(env.reset(goal=goal, init_index=init_index, seed=seed, **kwargs))
    initial_physics = env.sim_state()
    initial_xyz = env.status()["object_xyz_m"]
    initial = {k:v.copy() for k,v in obs.items()}
    np.savez_compressed(out/"initial-public.npz", **initial, sim_state=initial_physics)
    actor.reset(seed)
    if video:
        import imageio.v2 as iio
        writer = iio.get_writer(str(out/"rollout.mp4"), fps=20, codec="libx264")
    else:
        writer = nullcontext(None)
    trace = []
    native_done = False
    first_correct_step = first_other_step = None
    first_lift = first_lift_step = None
    sustained = {name:0 for name in GOALS}
    other = next(x for x in GOALS if x != goal)
    with writer as recording:
        if recording is not None:
            recording.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        for step in range(1,max_steps+1):
            a = np.asarray(actor.act(obs,env.language), dtype=np.float64)
            if a.shape != (7,) or not np.isfinite(a).all():
                raise ValueError("SmolVLA did not produce seven finite native commands")
            new_obs, reward, native_done, _info = env.step(a)
            obs = public_observation(new_obs)
            status = env.status()
            for name in GOALS:
                lifted = (name in status["grasped_objects"]
                          and status["object_xyz_m"][name][2] - initial_xyz[name][2] >= .025)
                sustained[name] = sustained[name] + 1 if lifted else 0
                if first_lift is None and sustained[name] >= 3:
                    first_lift, first_lift_step = name, step
            if status[f"{goal}_goal"] and first_correct_step is None:
                first_correct_step = step
            if status[f"{other}_goal"] and first_other_step is None:
                first_other_step = step
            trace.append({"step":step,"action":a.tolist(),"eef_pos":obs["robot0_eef_pos"].tolist(),
                          "reward":float(reward),"status":status})
            if recording is not None:
                recording.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if first_correct_step is not None or native_done:
                break
    final = env.status()
    r = {"init_index":init_index,"seed":seed,"goal":goal,"instruction":env.language,"layout":layout,
         "steps":len(trace),"native_done":bool(native_done),
         "first_grasp":final["first_grasp"],"first_grasp_step":final["first_grasp_step"],
         "first_lift":first_lift,"first_lift_step":first_lift_step,"wrong_object_lifted_first":first_lift==other,
         "first_intended_goal_step":first_correct_step,"first_other_goal_step":first_other_step,
         "success":first_correct_step is not None,"wrong_object_first":final["first_grasp"]==other,
         "completed_other_goal":first_other_step is not None}
    save(out/"result.json",r)
    save(out/"trace.json",trace)
    return r,initial,initial_physics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--init",type=int,nargs="+",default=[0,1,2,3])
    p.add_argument("--device",default="cuda")
    p.add_argument("--render-gpu",type=int,default=-1)
    p.add_argument("--max-steps",type=int,default=300)
    p.add_argument("--init-source",choices=GOALS)
    p.add_argument("--no-video",action="store_true")
    p.add_argument("--n-action-steps",type=int)
    p.add_argument("--layouts",choices=("canonical","swap_xy"),nargs="+",default=["canonical"])
    args=p.parse_args()
    args.run.mkdir()
    from smol_actor import SmolActor
    actor=SmolActor(args.checkpoint,device=args.device,n_action_steps=args.n_action_steps)
    env=PairedEnv(pair="objects",camera_size=256,render_gpu_device_id=args.render_gpu)
    result={"max_steps":args.max_steps,"n_action_steps":actor.policy.config.n_action_steps,"paired":[],
            "criterion":"both official goals should be possible before interpreting wrong first grasp as target-selection failure"}
    try:
        for layout in args.layouts:
            layout_out = args.run / layout
            layout_out.mkdir()
            for index in args.init:
                # Equal robot/noise seed and exact same native initial state across the pair.
                seed=1000+index
                cases={}
                reference=None
                ref_state=None
                for goal in GOALS:
                    row,public,state=run_episode(env,actor,goal,index,seed,args.max_steps,layout_out,
                        video=not args.no_video,init_source=args.init_source,layout=layout)
                    if reference is None:
                        reference,ref_state=public,state
                    elif not np.array_equal(state,ref_state) or any(not np.array_equal(public[k],reference[k]) for k in PUBLIC):
                        raise AssertionError(f"Native reset differs between prompts ({layout} init {index})")
                    cases[goal]=row
                    print(json.dumps(row,ensure_ascii=False),flush=True)
                result["paired"].append({"init_index":index,"layout":layout,"same_initial_public_obs":True,
                    "both_goals":all(r["success"] for r in cases.values()),
                    "both_lift_correct_first":all(k==r["first_lift"] for k,r in cases.items()),"arms":cases})
                save(args.run/"summary.json",result)
    finally:
        env.close()


if __name__=="__main__":
    main()
