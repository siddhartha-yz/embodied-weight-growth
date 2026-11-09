"""Direct action-output projection surgery on frozen SmolVLA, not an action wrapper."""
from contextlib import ExitStack, contextmanager
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paired_env import PairedEnv
from screen import PUBLIC, run_episode, save
from smol_actor import SmolActor

PROMPT = "pick up the wine bottle, then put it on top of the cabinet"
CASES = {"fault": (2,"bottle","swap_xy",80), "clean2": (2,"bottle","canonical",0),
         "clean3": (3,"bottle","canonical",0), "cleanB": (3,"bowl","canonical",0)}
HOLDOUT = {"heldout0": (0,"bottle","canonical",0), "heldout1": (1,"bowl","canonical",0)}
CAPTURE = {"fault": (50,70,80,90,100,110,120), "clean2": (20,40,60,70,80),
           "clean3": (30,50,70,80), "cleanB": (30,50,70,80)}
DIMS = {0:"x",1:"y",2:"z",6:"gripper"}

def head(actor):
    return actor.policy.model.action_out_proj

@contextmanager
def edit_head(actor, edits):
    layer=head(actor); saved=[]
    try:
        with torch.inference_mode():
            for item in edits:
                r,c,d=item["row"],item["column"],item["delta"]
                p=layer.bias if c==-1 else layer.weight
                k=r if c==-1 else (r,c)
                saved.append((p,k,p[k].clone()))
                p[k]+=d
                if torch.equal(saved[-1][2],p[k]):
                    raise ValueError("Numerical edit did not change the actual parameter")
        yield
    finally:
        with torch.inference_mode():
            for p,k,v in reversed(saved): p[k]=v

class EpisodeActor:
    def __init__(self, actor, switch, steps_to_capture=(), diagnostic=None, late_edits=(), edit_stack=None):
        self.actor,self.switch,self.capture,self.diagnostic=actor,switch,steps_to_capture,diagnostic
        self.late_edits,self.edit_stack=late_edits,edit_stack

    def reset(self,seed):
        self.actor.reset(seed); self.t=0; self.cache={}; self.last=None

    def act(self,obs,instruction):
        t=self.t; self.t+=1
        used_prompt=PROMPT if t<self.switch else instruction
        if t%10==0:
            if self.late_edits and t==self.switch:
                self.edit_stack.enter_context(edit_head(self.actor,self.late_edits))
            batch=self.actor.prepare(obs,used_prompt)
            noise=self.actor.make_noise()
            raw=[]
            with torch.inference_mode():
                hook=head(self.actor).register_forward_pre_hook(lambda m,a:raw.append(a[0][:,:10].clone())) if t in self.capture else None
                try: self.last=self.actor.predict_action_chunk(batch,noise=noise.clone())[0,:10].cpu().numpy()
                finally:
                    if hook is not None: hook.remove()
                if t in self.capture:
                    self.cache[t]=(batch,noise,self.last.copy(),torch.cat(raw).float().cpu().numpy())
        a=self.last[t%10].copy()
        if self.diagnostic in ("hold","carry") and 100<=t<125: a[6]=max(a[6],1)
        if self.diagnostic=="carry" and 100<=t<116:
            a[0]=min(a[0],-.15); a[1]=min(a[1],-.72)
            if t<106: a[2]=max(a[2],.40)
        return a

def episode(env, actor, output, case, refroot, *, edits=(), capture=False, diagnostic=None):
    i,g,layout,switch=(CASES|HOLDOUT)[case]
    output.mkdir()
    late=edits if case=="fault" else ()
    with ExitStack() as scope:
        wrapper=EpisodeActor(actor,switch,CAPTURE[case] if capture else (),diagnostic,late,scope)
        if edits and not late: scope.enter_context(edit_head(actor,edits))
        result,obs,state=run_episode(env,wrapper,g,i,1000+i,300,output,video=True,init_source="bowl",layout=layout)
    n=f"init-{i:03d}-{g}"
    ref=refroot/("instruction-control/phase-switch" if case=="fault" else "screen-stock-prompt/canonical")/n
    original=json.loads((ref/"trace.json").read_text())
    trace=json.loads((output/n/"trace.json").read_text())
    with np.load(ref/"initial-public.npz",allow_pickle=False) as expected:
        if not np.array_equal(state,expected["sim_state"]) or any(not np.array_equal(obs[k],expected[k]) for k in PUBLIC):
            raise AssertionError("Original experiment reset not reproduced")
    if not edits:
        keep=100 if diagnostic else len(original)
        if len(trace)<keep or trace[:keep]!=original[:keep]:
            raise AssertionError(f"Stock trajectory not reproduced for {case}/{diagnostic}")
    if late and trace[:switch]!=original[:switch]:
        raise AssertionError("Patch deployment altered the matched acquisition prefix")
    last_hold=max((q["step"] for q in trace if g in q["status"]["grasped_objects"]), default=None)
    first_open_after_hold=next((q["step"] for q in trace if result["first_lift_step"] is not None and q["step"]>result["first_lift_step"] and q["action"][6]<0),None)
    edge = {q["step"]:{"target_xyz":q["status"]["object_xyz_m"][g],"grasped":g in q["status"]["grasped_objects"],"action":q["action"]} for q in trace if q["step"] in (80,90,100,105,106,110,115,120,125,130)}
    result.update({"last_two_pad_contact":last_hold,"first_open_after_lift":first_open_after_hold,"milestones":edge})
    return result,wrapper.cache

def probes(actor, caches):
    # Rank channels only to decide what to probe, not to calculate a repair.
    target=caches["fault"][100][3][:,5:,:].mean(axis=(0,1))
    clean=np.stack([record[3].mean(axis=(0,1)) for case in ("clean2","clean3","cleanB") for record in caches[case].values()])
    control=caches["fault"][80][3].mean(axis=(0,1))
    score=abs(target)/(0.75 + np.sqrt((clean**2).mean(axis=0)) + abs(control)*.4)
    channels=np.argsort(score)[-4:][::-1].tolist()
    directions=[]
    for r,sign in [(0,1),(1,1),(2,-1),(6,-1)]:
        directions.append((r,-1,sign*.65))
        for c in channels[:(4 if r==6 else 2)]:
            directions.append((r,c,float(sign*.24*np.sign(target[c]))))
    samples=[("fault",80),("fault",90),("fault",100),("clean2",70),("clean2",80),
             ("clean3",70),("cleanB",70)]
    samples=[(case,t) for case,t in samples if t in caches[case]]
    rows=[]
    for r,c,d in directions:
        p=head(actor).bias[r] if c<0 else head(actor).weight[r,c]
        row={"edit":{"row":r,"column":c,"delta":d},"old":float(p),"chunks":{}}
        with edit_head(actor,[row["edit"]]):
            for case,t in samples:
                b,n,base,h=caches[case][t]
                out=actor.predict_action_chunk(b,noise=n.clone())[0,:10].cpu().numpy()
                arr=out-base
                row["chunks"][f"{case}@{t}"]={
                    "mean_delta_xyz_gripper":arr[:,[0,1,2,6]].mean(axis=0).round(4).tolist(),
                    "first5_mean_delta_in_edited_axis":round(float(arr[:5,r].mean()),4),
                    "last5_mean_delta_in_edited_axis":round(float(arr[5:,r].mean()),4),
                    "gripper_chunk":out[:,6].round(3).tolist(),
                    "unedited_rms":round(float(np.sqrt(np.mean(np.delete(arr,r,axis=1)**2))),4),
                }
        rows.append(row)
    base={}
    for case,t in samples:
        b,n,a,h=caches[case][t]
        base[f"{case}@{t}"]={"xyz_first5_mean":a[:5,:3].mean(axis=0).round(4).tolist(),
            "xyz_last5_mean":a[5:,:3].mean(axis=0).round(4).tolist(),"gripper":a[:,6].round(3).tolist(),
            "meanh":{c:round(float(h[:,:,c].mean()),4) for c in channels}}
    return {"parameter":"model.action_out_proj", "shape":list(head(actor).weight.shape),
            "dtype":str(head(actor).weight.dtype),"probes":rows,"stock_chunks":base}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--render-gpu",type=int,default=5)
    p.add_argument("--patch",type=Path,action="append"); p.add_argument("--screen",type=Path,default=Path.cwd())
    a=p.parse_args(); a.out.mkdir()
    actor=SmolActor(n_action_steps=10); summary={}
    with PairedEnv(pair="objects",camera_size=256,render_gpu_device_id=a.render_gpu) as env:
        if a.patch is None:
            caches={}
            for case in CASES:
                summary[case],caches[case]=episode(env,actor,a.out/case,case,a.screen,capture=True)
                save(a.out/"episodes.json",summary)
                print(case,summary[case]["success"],flush=True)
            for diag in ("hold","carry"):
                summary[diag],_=episode(env,actor,a.out/diag,"fault",a.screen,diagnostic=diag)
                save(a.out/"episodes.json",summary)
                print(diag,summary[diag]["success"],flush=True)
            save(a.out/"probes.json",probes(actor,caches))
        else:
            for spec in a.patch:
                out=a.out/spec.stem; out.mkdir(); summary={}
                patch=json.loads(spec.read_text())["edits"]
                if not 0<len(patch)<=8: raise ValueError("expected 1–8 existing scalar changes")
                keys=set()
                for e in patch:
                    r,c,d=e["row"],e["column"],e["delta"]
                    if type(r)!=int or type(c)!=int or r not in DIMS or not -1<=c<head(actor).in_features or type(d) not in (float,int) or not math.isfinite(d) or not 0<abs(d)<= (1.25 if c==-1 else .65) or (r,c) in keys:
                        raise ValueError("edit outside approved scalar budget")
                    keys.add((r,c))
                actual=[]
                layer=head(actor)
                for e in patch:
                    r,c=e["row"],e["column"]
                    old=float(layer.bias[r] if c<0 else layer.weight[r,c])
                    actual.append({**e,"before":old})
                with edit_head(actor,patch):
                    for e in actual:
                        r,c=e["row"],e["column"]
                        e["after"]=float(layer.bias[r] if c<0 else layer.weight[r,c])
                save(out/"patch.json",actual)
                for case in (CASES|HOLDOUT):
                    summary[case],_=episode(env,actor,out/case,case,a.screen,edits=patch)
                    save(out/"episodes.json",summary)
                    print(spec.stem,case,summary[case]["success"],flush=True)
if __name__=="__main__": main()
