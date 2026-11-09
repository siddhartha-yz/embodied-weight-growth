"""Independent stock LeRobot LIBERO control for checkpoint competence."""
import argparse
import json
from pathlib import Path

import numpy as np

from paired_env import _runtime
from smol_actor import SmolActor


def raw_observation(stock):
    return {"agentview_image":stock["pixels"]["image"],
            "robot0_eye_in_hand_image":stock["pixels"]["image2"],
            "robot0_eef_pos":stock["robot_state"]["eef"]["pos"],
            "robot0_eef_quat":stock["robot_state"]["eef"]["quat"],
            "robot0_gripper_qpos":stock["robot_state"]["gripper"]["qpos"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--size",type=int,default=256)
    args=p.parse_args()
    args.run.mkdir()
    _runtime(None)
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, TASK_SUITE_MAX_STEPS
    actor=SmolActor(n_action_steps=10)
    rows=[]
    cases=[("libero_goal",4,0),("libero_goal",2,0),
           ("libero_10",0,0),("libero_10",4,0),
           ("libero_10",0,1),("libero_10",4,1)]
    for suite,task,row in cases:
        env=LiberoEnv(benchmark.get_benchmark_dict()[suite](),task,suite,obs_type="pixels_agent_pos",
            init_states=True,episode_index=row,observation_width=args.size,observation_height=args.size,
            control_mode="relative")
        try:
            seed=1000+row
            obs,_=env.reset(seed=seed)
            actor.reset(seed)
            success=False
            for step in range(1,TASK_SUITE_MAX_STEPS[suite]+1):
                action=actor.act(raw_observation(obs),env.task_description)
                obs,_reward,term,trunc,info=env.step(np.asarray(action))
                success=bool(info.get("is_success",False))
                if term or trunc or success:
                    break
            r={"suite":suite,"task":task,"init":row,"instruction":env.task_description,
               "size":args.size,"n_action_steps":10,"success":success,"steps":step}
            rows.append(r)
            (args.run/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
            print(json.dumps(r),flush=True)
        finally:
            env.close()


if __name__=="__main__":
    main()
