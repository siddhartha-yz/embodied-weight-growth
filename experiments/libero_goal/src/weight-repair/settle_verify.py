"""Verify physical retention after the official first-success boundary, without an action override."""
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import imageio.v2 as iio
import numpy as np
from repair import EpisodeActor, PairedEnv, SmolActor, save
from screen import public_observation

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--case",type=Path,required=True)
    p.add_argument("--patch",type=Path,required=True)
    p.add_argument("--render-gpu",type=int,default=5)
    a=p.parse_args(); a.out.mkdir()
    episode=json.loads((a.case/"result.json").read_text())
    prefix=json.loads((a.case/"trace.json").read_text())
    edits=json.loads(a.patch.read_text())["edits"]
    score=episode["first_intended_goal_step"]
    assert episode["success"] and score==len(prefix)
    smol=SmolActor(n_action_steps=10)
    with ExitStack() as scope, PairedEnv(pair="objects",camera_size=256,render_gpu_device_id=a.render_gpu) as env, iio.get_writer(a.out/"continuation.mp4",fps=20,codec="libx264") as video:
        actor=EpisodeActor(smol,80,late_edits=edits,edit_stack=scope)
        obs=public_observation(env.reset(goal="bottle",init_source="bowl",init_index=2,seed=1002,layout="swap_xy"))
        actor.reset(1002); post=[]
        for step in range(1,score+41):
            action=actor.act(obs,env.language)
            raw,_,done,_=env.step(action)
            obs=public_observation(raw); status=env.status()
            if step<=score:
                ref=prefix[step-1]
                if not np.array_equal(action,ref["action"]) or not np.array_equal(obs["robot0_eef_pos"],ref["eef_pos"]) or status!=ref["status"]:
                    raise RuntimeError(f"Patched rollout prefix differs at step {step}")
            if step>=95:
                video.append_data(np.ascontiguousarray(obs["agentview_image"][::-1,::-1]))
            if step>=score:
                post.append({"step":step,"goal":done,"bottle_xyz_m":status["object_xyz_m"]["bottle"],
                             "bottle_grasped":"bottle" in status["grasped_objects"],"gripper_command":float(action[6])})
        final={"first_native_success":score,"prefix_reproduced_exactly":True,"extra_controls":40,
               "goal_true_for_all_40_extra_steps":all(x["goal"] for x in post[1:]),
               "goal_true_extra_steps":sum(x["goal"] for x in post[1:]),
               "first_lost_goal_step":next((x["step"] for x in post if not x["goal"]),None),
               "goal_at_end":post[-1]["goal"],"still_held_at_end":post[-1]["bottle_grasped"],"post":post}
        save(a.out/"settling.json",final)
        print(json.dumps({k:v for k,v in final.items() if k!="post"}),flush=True)
if __name__=="__main__": main()
