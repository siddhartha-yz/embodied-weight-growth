"""Evaluate fixed absolute SmolVLA edits and collect matched interventions for the next round."""
from contextlib import contextmanager
from pathlib import Path
import argparse
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "multitask-repair")]
from matrix import (SmolActor, capture_summary, head, read, run_other, run_pair, save)
from layouts import BowlOffsetEnv

DEV = ROOT / "multitask-repair"
AXES = (0, 1, 2, 6)
FOCUS = {
    "bottle": ("historical-bottle", "clean-bottle-2"),
    "bowl": ("bowl-rim-dev", "clean-bowl-2"),
}

@contextmanager
def assigned(layer, edits):
    with torch.no_grad():
        old = [(e["row"], e["column"], layer.weight[e["row"], e["column"]].clone()) for e in edits]
        for e in edits:
            layer.weight[e["row"], e["column"]] = e["value"]
    try:
        yield
    finally:
        with torch.no_grad():
            for r, c, v in old:
                layer.weight[r, c] = v

def get_cases():
    by_id = {c["id"]: c for name in ("historical-clean", "bowl-development")
             for c in read(DEV / f"{name}.json")}
    ids = [c for cases in FOCUS.values() for c in cases]
    return [by_id[c] for c in ids], [c for i, c in by_id.items() if i not in ids]

def contexts(result, cache, trace):
    def replanned(step):
        return min(max(cache), max(0, (step - 1) // 10 * 10))
    goal = result["goal"]
    lift = result["first_lift_step"] if result["first_lift"] == goal else None
    if not lift:
        ticks = {"start": 0, "approach": 20}
        window = [q for q in trace[20:160] if q["action"][6] >= 0]
        if window:
            closest = min(window, key=lambda q: np.linalg.norm(np.array(q["eef"])-q["target"]))
            ticks["attempted_grasp"] = replanned(closest["step"])
        if result["first_lift_step"]:
            ticks["wrong_object"] = replanned(result["first_lift_step"])
    else:
        place = result["first_native_goal"] or result["last_contact"] or lift
        ticks = {"pregrasp": max(0, replanned(lift)-10), "grasp": replanned(lift),
                 "carry": max(replanned(lift), replanned(place)-10), "place": replanned(place)}
    return {k:(t,*cache[t]) for k,t in ticks.items() if t in cache}

def probes(actor, obs, edits, stock_edits=()):
    """Single-weight forward intervention on identical actual observations and noise."""
    layer = head(actor)
    refs = [v[-1].mean(axis=(0,1)) for k,v in obs.items() if "place" not in k]
    ref = np.sqrt(np.mean(np.square(refs), axis=0))
    cols = {89,371,8,483}
    for name, v in obs.items():
        if name.endswith(("/place","/attempted_grasp","/wrong_object")) and "/fault/" in name:
            score = abs(v[-1].mean(axis=(0,1))) / (.45 + ref)
            cols.update(int(c) for c in np.argsort(score)[-4:])
    keys = {(r,c) for r in AXES for c in cols}
    keys.update((e["row"],e["column"]) for e in edits)
    def halves(a):
        a = a[:,AXES]
        return np.round(np.stack([a[:5].mean(0),a[5:10].mean(0)]),4).tolist()
    current = {k:{"t":v[0],"actions":halves(v[-2])} for k,v in obs.items()}
    rows=[]
    for r,c in sorted(keys):
        value = float(layer.weight[r,c])
        test = dict(row=r,column=c,value=value+.30)
        with assigned(layer,[test]):
            effects = {}
            for name, (_,batch,noise,actions,_) in obs.items():
                out=actor.predict_action_chunk(batch,noise=noise.clone())[0,:10].cpu().numpy()
                effects[name]=halves(out-actions)
        rows.append({"row":r,"column":c,"current":value,"trial_offset":.30,"effects":effects})
    result={"contexts":current,"edits_tested":rows}
    if stock_edits:
        with assigned(layer,stock_edits):
            for name, (_,batch,noise,actions,_) in obs.items():
                out=actor.predict_action_chunk(batch,noise=noise.clone())[0,:10].cpu().numpy()
                current[name]["stock_minus_current_same_input"]=halves(out-actions)
        result["stock_values_for_prior_assignments"]=stock_edits
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--edits",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--vlm-assets",type=Path)
    p.add_argument("--next",type=int,choices=(1,2,3))
    p.add_argument("--split",choices=("dev","preserve","heldout"),default="dev")
    p.add_argument("--heldout",type=Path)
    p.add_argument("--render-gpu",type=int,default=5)
    a=p.parse_args()
    if a.checkpoint and a.edits: p.error("Choose a checkpoint or in-memory assignments, not both")
    if a.next and a.split!="dev": p.error("Only development evidence can be sent to Astra")
    dev,preserve=get_cases()
    cases = dev if a.split=="dev" else preserve if a.split=="preserve" else read(a.heldout)
    edits=read(a.edits)["edits"] if a.edits else []
    selected = FOCUS.keys() if a.next==3 else ("bottle",) if a.next==1 else ("bowl",) if a.next==2 else ()
    looking = {cid:(goal, "fault" if i==0 else "clean") for goal in selected
               for i,cid in enumerate(FOCUS[goal])}
    a.out.mkdir()
    save(a.out/"cases.json",cases)
    actor=SmolActor(checkpoint_dir=a.checkpoint,vlm_assets_dir=a.vlm_assets,n_action_steps=10)
    layer=head(actor); outcomes=[]; cached={}; details={}
    stock_edits = [{**e,"value":float(layer.weight[e["row"],e["column"]])} for e in edits] if a.next==3 else []
    with assigned(layer,edits):
        paired=[c for c in cases if "suite" not in c]
        if paired:
            with BowlOffsetEnv(pair="objects",camera_size=256,render_gpu_device_id=a.render_gpu) as env:
                for case in paired:
                    cid=case["id"]
                    result,cache,trace=run_pair(actor,env,case,a.out/cid,cid in looking,goal_tail=80)
                    outcomes.append(result)
                    print(cid,result["first_lift"],result["first_lift_step"],
                          result["first_native_goal"],result["stable_released_at"],flush=True)
                    if cid in looking:
                        g,role=looking[cid]
                        observed=contexts(result,cache,trace)
                        cached.update({f"{g}/{role}/{k}":v for k,v in observed.items()})
                        focus= [result[k] for k in ("first_lift_step","first_cabinet_contact_after_lift",
                                                    "last_contact","first_native_goal","stable_released_at") if result[k] is not None]
                        sampled=capture_summary(trace,g)
                        focus.extend(t for t,*_ in observed.values())
                        details[cid]={"result":result,"trace":[q for q in sampled if q["t"]<=35 or
                                      any(abs(q["t"]-e)<=15 for e in focus) or q["t"]>=len(trace)-5]}
                    del cache
        for case in cases:
            if "suite" in case:
                r=run_other(actor,case,a.out/case["id"])
                outcomes.append(r); print(case["id"],r["native_success"],flush=True)
        save(a.out/"summary.json",outcomes)
        if a.next:
            save(a.out/"probes.json",{"evaluated":details,**probes(actor,cached,edits,stock_edits)})
            print("Interventions recorded",flush=True)

if __name__=="__main__":
    main()
