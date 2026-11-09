"""Paired, fixed-from-start SmolVLA multi-repair eval and bowl-only forward probes."""
from contextlib import nullcontext
from pathlib import Path
import argparse
import json
import math
import sys

import imageio.v2 as iio
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"weight-repair")]
from paired_env import PairedEnv, _runtime, OBJECTS
from repair import edit_head, head, PROMPT
from screen import public_observation, save
from smol_actor import SmolActor
from stock_control import raw_observation

ARMS=("stock","bottle","bowl","merged")
EXTRA=40
AXES=(0,1,2,6)
def read(p): return json.loads(Path(p).read_text())

def merge_edits(a,b):
    keyed={}
    for edits in (a,b):
        for row in edits:
            k=(row["row"],row["column"])
            keyed[k]=keyed.get(k,0.)+row["delta"]
    return [{"row":r,"column":c,"delta":d} for (r,c),d in sorted(keyed.items()) if d!=0]

def checked(path,layer):
    edits=read(path)["edits"]
    if not 1<=len(edits)<=8: raise ValueError("Expected at most eight existing output-head scalars")
    keys=set()
    for e in edits:
        r,c,d=e["row"],e["column"],e["delta"]
        if type(r) is not int or type(c) is not int or r not in AXES or not -1<=c<layer.in_features or type(d) not in (int,float) or not math.isfinite(d) or not 0<abs(d)<=(1.25 if c==-1 else .65) or (r,c) in keys:
            raise ValueError(f"Invalid/duplicate edit: {e}")
        keys.add((r,c))
    return edits

def rim_contacts(env,goal):
    native=env._env.env
    model=native.sim.model
    data=native.sim.data
    geom={model.geom_name2id(x) for x in native.get_object(OBJECTS[goal]).contact_geoms}
    result=set()
    for c in data.contact[:data.ncon]:
        other=c.geom2 if c.geom1 in geom else c.geom1 if c.geom2 in geom else None
        if other is not None:
            name=model.geom_id2name(other) or ""
            if "cabinet" in name:
                result.add(name)
    return sorted(result)

class Observer:
    """Records the unmodified head input on native chunk boundaries during the stock probes."""
    def __init__(self,actor,record=False):
        self.actor=actor; self.record=record; self.caches={}
    def act(self,obs,prompt,t):
        if not self.record or t%10:
            return self.actor.act(obs,prompt)
        # Use stock select_action's queue path; intercept the actual batch and same RNG draw.
        policy=self.actor.policy
        saved=policy._get_action_chunk
        activ=[]
        def call(batch,noise=None,**kwargs):
            before=dict(batch)
            out=saved(batch,noise=noise,**kwargs)
            self.caches[t]=(before,noise.clone(),self.actor.postprocessor(out)[0,:10].cpu().numpy(),torch.cat(activ).float().cpu().numpy())
            return out
        with torch.inference_mode():
            hook=head(self.actor).register_forward_pre_hook(lambda _,a:activ.append(a[0][:,:10].clone()))
            policy._get_action_chunk=call
            try: return self.actor.act(obs,prompt)
            finally:
                policy._get_action_chunk=saved
                hook.remove()

def run_pair(actor,env,case,out,record=False,*,goal_tail=EXTRA):
    out.mkdir(parents=True)
    i,g=case["init_index"],case["goal"]
    obs=public_observation(env.reset(goal=g,init_index=i,seed=case["seed"],init_source=case.get("init_source","bowl"),layout=case["layout"]))
    initial={**obs,"sim_state":env.sim_state()}
    zs={k:v[2] for k,v in env.status()["object_xyz_m"].items()}
    np.savez_compressed(out/"initial-public.npz",**initial)
    actor.reset(case["seed"])
    observer=Observer(actor,record)
    first_lift=first_lift_step=first_goal=first_other=first_open=None
    lift={k:0 for k in ("bowl","bottle")}; rows=[]; released_streak=0
    stable_step=None; budget=case.get("max_steps",300)
    other="bottle" if g=="bowl" else "bowl"
    history=case.get("prompt_mode")=="historical_bottle"
    with iio.get_writer(out/"rollout.mp4",fps=20,codec="libx264") as video:
        video.append_data(np.ascontiguousarray(obs["agentview_image"][::-1,::-1]))
        for step in range(1,budget+goal_tail+1):
            if step>budget and first_goal is None: break
            if first_goal is not None and step>first_goal+goal_tail: break
            instruction=PROMPT if history and step<=80 else env.language
            a=observer.act(obs,instruction,step-1)
            obs,_,done,_=env.step(a); obs=public_observation(obs); status=env.status()
            z=status["object_xyz_m"]; held=status["grasped_objects"]
            for name in lift:
                lift[name]=lift[name]+1 if name in held and z[name][2]-zs[name]>=.025 else 0
                if first_lift is None and lift[name]>=3:
                    first_lift,first_lift_step=name,step
            if step<=budget and done and first_goal is None: first_goal=step
            if first_other is None and status[f"{other}_goal"]: first_other=step
            if first_open is None and first_lift==g and step>first_lift_step and float(a[6])<0: first_open=step
            released_streak=released_streak+1 if done and g not in held else 0
            if released_streak>=EXTRA and stable_step is None: stable_step=step
            rows.append({"step":step,"action":a.tolist(),"eef":obs["robot0_eef_pos"].tolist(),
                "target":z[g],"other":z[other],"held":held,"native":bool(done),
                "other_native":status[f"{other}_goal"],"cabinet_contacts":rim_contacts(env,g)})
            video.append_data(np.ascontiguousarray(obs["agentview_image"][::-1,::-1]))
    held=[r for r in rows if g in r["held"] and first_lift==g and r["step"]>=first_lift_step]
    touches=[r["step"] for r in rows if first_lift==g and r["step"]>=first_lift_step and r["cabinet_contacts"]]
    result={**case,"instruction":env.language,"first_prompt":PROMPT if history else env.language,
            "first_lift":first_lift,"first_lift_step":first_lift_step,"first_open_after_lift":first_open,
            "first_native_goal":first_goal,"first_other_goal":first_other,"stable_released_at":stable_step,
            "native_success":first_goal is not None,"stable_released_40":stable_step is not None,
            "first_cabinet_contact_after_lift":touches[0] if touches else None,
            "last_contact":held[-1]["step"] if held else None,"last_contact_xyz":held[-1]["target"] if held else None,
            "final_xyz":rows[-1]["target"],"final_native":rows[-1]["native"],"steps":len(rows)}
    save(out/"result.json",result); save(out/"trace.json",rows)
    return result,observer.caches,rows

def run_other(actor,case,out):
    _runtime(None)
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    out.mkdir(parents=True); suite=case["suite"]; task=case["task"]; row=case["init_index"]
    env=LiberoEnv(benchmark.get_benchmark_dict()[suite](),task,suite,obs_type="pixels_agent_pos",
        init_states=True,episode_index=row,observation_width=256,observation_height=256,
        control_mode="relative")
    try:
        obs,_=env.reset(seed=case["seed"]); actor.reset(case["seed"]); raw=raw_observation(obs)
        np.savez_compressed(out/"initial-public.npz",**raw,sim_state=env._env.get_sim_state())
        rows=[]; success=False
        with iio.get_writer(out/"rollout.mp4",fps=20,codec="libx264") as video:
            video.append_data(np.ascontiguousarray(raw["agentview_image"][::-1,::-1]))
            for step in range(1,case["max_steps"]+1):
                a=actor.act(raw,env.task_description)
                obs,_,done,trunc,info=env.step(a); raw=raw_observation(obs)
                success=bool(info.get("is_success",False))
                rows.append({"step":step,"action":a.tolist(),"native":success})
                video.append_data(np.ascontiguousarray(raw["agentview_image"][::-1,::-1]))
                if done or trunc or success: break
        result={**case,"instruction":env.task_description,"native_success":success,
                "first_native_goal":step if success else None,"steps":step}
        save(out/"trace.json",rows); save(out/"result.json",result)
        return result
    finally: env.close()

def capture_summary(rows,goal):
    return [{"t":r["step"],"eef":np.round(r["eef"],3).tolist(),"xyz":np.round(r["target"],3).tolist(),
             "held":goal in r["held"],"cabinet":r["cabinet_contacts"],"action":np.round(r["action"],3).tolist()}
             for r in rows if r["step"]%5==0]

def probes(actor,fault,clean):
    fr,fc,ft=fault; cr,cc,ct=clean
    last=fr["last_contact"]; lift=fr["first_lift_step"]
    if fr["first_lift"]!="bowl" or fr["native_success"] or last is None:
        raise ValueError("Bowl dev probe must be a real post-grasp failure")
    t=max(0,((last-1)//10)*10)
    ct0=((cr["first_lift_step"]-1)//10)*10
    ct1=((cr["first_native_goal"]-1)//10)*10
    xs={}
    for name,key,c in [("fault_early",max(10,((lift-1)//10-1)*10),fc),("fault_carry",max(0,t-10),fc),
                       ("fault_rim",t,fc),("clean_early",max(0,ct0-10),cc),
                       ("clean_carry",max(0,ct1-10),cc),("clean_place",ct1,cc)]:
        xs[name]=(key,*c[key])
    # Select channel candidates by signed activation contrast, then do true one-edit forward interventions.
    late=xs["fault_rim"][-1].mean(axis=(0,1))
    ref=np.stack([v[-1].mean(axis=(0,1)) for k,v in xs.items() if k.startswith("clean") or k=="fault_early"])
    relevance=abs(late)/(0.50+np.sqrt(np.mean(ref**2,axis=0)))
    channels=[int(c) for c in np.argsort(relevance)[-6:][::-1]]
    if 89 not in channels: channels.append(89)
    if 371 not in channels: channels.append(371)
    summary={k:{"t":v[0],"means":v[-2][:,[0,1,2,6]].mean(axis=0).round(4).tolist(),
        "late_grip":v[-2][5:,6].round(3).tolist(),
        "h":{c:round(float(v[-1][:,:,c].mean()),4) for c in channels}} for k,v in xs.items()}
    output=[]
    for r in AXES:
        for c in [*channels[:4],89,371]:
            if c in channels[:4] and c in (89,371) and any(q["row"]==r and q["column"]==c for q in output): continue
            d=.30
            original=float(head(actor).weight[r,c])
            entry={"row":r,"column":c,"delta":d,"original":original,"effects":{}}
            with edit_head(actor,[{"row":r,"column":c,"delta":d}]):
                for k,(tick,b,n,a,h) in xs.items():
                    got=actor.predict_action_chunk(b,noise=n.clone())[0,:10].cpu().numpy()
                    err=got-a
                    entry["effects"][k]=[round(float(err[:5,r].mean()),4),round(float(err[5:,r].mean()),4)]
            output.append(entry)
    return {"fault":fr,"clean":cr,"fault_trace":capture_summary(ft,"bowl"),
        "clean_trace":capture_summary(ct,"bowl"),"contexts":summary,"edit_probes":output}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--a",type=Path); p.add_argument("--b",type=Path)
    p.add_argument("--arms",nargs="+",choices=ARMS,required=True)
    p.add_argument("--render-gpu",type=int,default=0); p.add_argument("--probe",nargs=2)
    args=p.parse_args(); args.out.mkdir()
    cases=read(args.cases); save(args.out/"cases.json",cases)
    actor=SmolActor(n_action_steps=10); layer=head(actor)
    a=checked(args.a,layer) if args.a else []
    b=checked(args.b,layer) if args.b else []
    deltas={"stock":[],"bottle":a,"bowl":b,"merged":merge_edits(a,b)}
    assert all(arm=="stock" or deltas[arm] for arm in args.arms)
    if args.probe and "stock" not in args.arms: p.error("Probes need stock")
    env_cls=PairedEnv
    if any(c.get("layout","canonical") not in ("canonical","swap_xy") for c in cases):
        from layouts import BowlOffsetEnv
        env_cls=BowlOffsetEnv
    output=[]; saved={}; resets={}
    def report(arm,case,out,result):
        reset=out/"initial-public.npz"
        if case["id"] not in resets:
            resets[case["id"]]=reset
        else:
            with np.load(reset,allow_pickle=False) as got, np.load(resets[case["id"]],allow_pickle=False) as prior:
                if got.files!=prior.files or any(not np.array_equal(got[k],prior[k]) for k in got.files):
                    raise RuntimeError(f"Unmatched initial simulator/observation state for {arm}/{case['id']}")
        output.append({"arm":arm,**result})
        save(args.out/"summary.json",output)
        print(arm,case["id"],result["native_success"],result.get("stable_released_40"),flush=True)

    for arm in args.arms:
        (args.out/arm).mkdir()
    def active_weights(arm):
        patch=deltas[arm]
        applied=[{**e,"original":float(layer.bias[e["row"]] if e["column"]==-1 else layer.weight[e["row"],e["column"]])} for e in patch]
        return patch,applied
    def log_weights(arm,applied):
        for e in applied:
            e["applied"]=float(layer.bias[e["row"]] if e["column"]==-1 else layer.weight[e["row"],e["column"]])
        save(args.out/arm/"weight_edits.json",applied)

    paired=[c for c in cases if "suite" not in c]
    others=[c for c in cases if "suite" in c]
    if paired:
        with env_cls(pair="objects",camera_size=256,render_gpu_device_id=args.render_gpu) as env:
            for arm in args.arms:
                patch,applied=active_weights(arm)
                with edit_head(actor,patch):
                    log_weights(arm,applied)
                    for case in paired:
                        out=args.out/arm/case["id"]
                        record=bool(args.probe and arm=="stock" and case["id"] in args.probe)
                        result,cache,trace=run_pair(actor,env,case,out,record)
                        if record: saved[case["id"]]=(result,cache,trace)
                        report(arm,case,out,result)
    # Never reuse the paired renderer after creating/closing another LIBERO environment:
    # this robosuite version invalidates the active EGL context when that renderer closes.
    if others:
        for arm in args.arms:
            patch,applied=active_weights(arm)
            with edit_head(actor,patch):
                log_weights(arm,applied)
                for case in others:
                    out=args.out/arm/case["id"]
                    report(arm,case,out,run_other(actor,case,out))
    if args.probe:
        save(args.out/"bowl_probes.json",probes(actor,saved[args.probe[0]],saved[args.probe[1]]))

if __name__=="__main__": main()
