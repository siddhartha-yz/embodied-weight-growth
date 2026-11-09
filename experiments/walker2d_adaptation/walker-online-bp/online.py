"""Teacher-free, matched-feedback online Walker2d actor updates (CPU, native MuJoCo)."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from walk_plant import WalkPlant
from walk_policy import FEATURES, JOINT_ORDER, compile_weights, inputs
from walk_server import Steps
from bp_control import Net

BATCHES = 5
TRIALS = 8
TICKS = 250
BP_STEPS = 12
SIGMA = .045
LR = .00030
ANCHOR = .24
GAMMA, GAE = .993, .96
AUDIT = list(range(500, 512))
OLD = None
ANCHORS = (390, 391, 392)
REGIONS = ("all", "first", "middle", "head", "hips", "knees", "ankles")


def read(p):
    return json.loads(p.read_text())


def write(p, d):
    p.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")


def vec(s):
    x = inputs(s,.5)
    return [x[k] for k in FEATURES]


def run(policy, seed, friction, noise=None, trace=False):
    plant=WalkPlant(seed,"train")
    # Higher-priority floor owns contact friction; all other contact settings match.
    if friction is not None:
        plant.model.geom_priority[plant.floor]=1
        plant.model.geom_friction[plant.floor,0]=friction
    s=plant.reset()
    steps=Steps(s)
    xx, aa, uu, rr = [], [], [], []
    path=[{"tick":0,**s}] if trace else []
    fall=None
    t0=s["t"]
    x0=s["x"]
    min_z,max_pitch=s["z"],abs(s["pitch"])
    last_feet=None
    slips=[]
    for i in range(TICKS):
        mean=np.asarray(policy(s,.5))
        a=mean if noise is None else mean+noise[i]
        u=np.clip(a,-1,1)
        x=s["x"]
        xx.append(vec(s));aa.append(a);uu.append(mean)
        s=plant.step(u)
        fall=plant.fall_reason(s)
        if not fall: steps.update(s)
        # Reward comes from the interaction itself, not target actions.
        rr.append(s["x"]-x + (.006 if not fall else -.75) - .0004*float(np.square(u).mean()))
        min_z,max_pitch=min(min_z,s["z"]),max(max_pitch,abs(s["pitch"]))
        feet=[steps.foot_x(s,j) for j in range(2)]
        if last_feet is not None:
            slips.extend(abs(a-b)/.02 for a,b,c0,c1 in zip(feet,last_feet[0],s["contact"],last_feet[1]) if c0 and c1)
        last_feet=(feet,s["contact"])
        if trace:
            path.append({"tick":i+1,**s,"u":u.tolist(),"fall_reason":fall,
                         "alternating_steps":steps.alternating_steps})
        if fall: break
    n=len(rr)
    w={"distance_m":s["x"]-x0,"elapsed_s":s["t"]-t0,"alternating_steps":steps.alternating_steps,
       "fall_reason":fall,"touchdowns":steps.touchdowns}
    return {
        "stats":{"pass":fall is None and n==TICKS and w["distance_m"]>=1 and w["alternating_steps"]>=4,
                 **{k:v for k,v in w.items() if k!="touchdowns"},"ticks":n,"return":sum(rr),
                 "min_z_m":min_z,"max_abs_pitch_rad":max_pitch,
                 "stance_slip_p90_m_s":float(np.quantile(slips,.9)) if slips else 0.,
                 "touchdowns":[[e["foot"],round(e["time_s"],2)] for e in steps.touchdowns]},
        "data":{"x":np.asarray(xx),"a":np.asarray(aa),"mean":np.asarray(uu),"r":np.asarray(rr)},
        "replay":{"walk":w,"trace":path,"success":fall is None and n==TICKS and w["distance_m"]>=1 and w["alternating_steps"]>=4,
                  "scene":{"initial_pitch_rad":.02}} if trace else None,
    }


def rewards_to_adv(rolls):
    x=np.concatenate([q["x"] for q in rolls])
    returns=[]
    for q in rolls:
        r=q["r"]; g=0.; ret=np.zeros_like(r)
        for i in range(len(r)-1,-1,-1):
            g=r[i]+GAMMA*g
            ret[i]=g
        returns.append(ret)
    s=np.maximum(x.std(0),.05)
    c=x.mean(0)
    f=np.column_stack([np.ones(len(x)),(x-c)/s])
    v=f@np.linalg.solve(f.T@f+8*np.eye(f.shape[1]),f.T@np.concatenate(returns))
    adv=[]
    k=0
    for q in rolls:
        n=len(q["r"])
        vk=v[k:k+n]; k+=n
        td=q["r"]+GAMMA*np.r_[vk[1:],0.]-vk
        e=0.; z=np.empty_like(td)
        for i in range(n-1,-1,-1):
            e=td[i]+GAMMA*GAE*e
            z[i]=e
        adv.append(z)
    a=np.concatenate(adv)
    return x, (a-a.mean())/(a.std()+1e-8)


def bundle(data):
    return [torch.as_tensor(data[k],dtype=torch.float64) for k in ("x","a","mean","adv")]


def objective(net, tensors, xh,yh):
    x,action,old,adv=tensors
    mu=net(x)
    logratio=((action-old).square()-(action-mu).square()).sum(1)/(2*SIGMA**2)
    ratio=logratio.clamp(-12.,12.).exp()
    surrogate=torch.minimum(adv*ratio,adv*ratio.clamp(.8,1.2))
    stay=(net(xh)-yh).square().mean()/(SIGMA**2)
    return -surrogate.mean()+ANCHOR*stay


def mask_gradients(net, region):
    rows={"hips":(0,3),"knees":(1,4),"ankles":(2,5)}
    select={"first":0,"middle":1,"head":2,**{k:2 for k in rows}}
    if region=="all": return
    for j,(w,b) in enumerate(zip(net.dw,net.db)):
        if j!=select[region]:
            w.grad.zero_(); b.grad.zero_()
        elif region in rows:
            other=[k for k in range(6) if k not in rows[region]]
            w.grad[other]=0.; b.grad[other]=0.


def norm_info(net,tensors,xh,yh):
    loss=objective(net,tensors,xh,yh)
    net.zero_grad()
    loss.backward()
    d={}
    with torch.no_grad():
        for j,(w,b) in enumerate(zip(net.dw,net.db)):
            g=w.grad*getattr(net,f"mask{j}")
            d[("first","middle","head")[j]]=float((g.square().sum()+b.grad.square().sum()).sqrt())
            if j==2:
                for k,v in (("hips",[0,3]),("knees",[1,4]),("ankles",[2,5])):
                    d[k]=float((g[v].square().sum()+b.grad[v].square().sum()).sqrt())
    net.zero_grad()
    return d


def config(root):
    return read(root/"config.json")


def pack(root):
    a=np.load(root/"anchor.npz")
    return a["center"],a["scale"],torch.from_numpy(a["x"]),torch.from_numpy(a["y"])


def setup(root,check,terrain):
    if not 0<terrain<1: raise ValueError("new ground sliding coefficient must be between zero and one")
    orig=read(check)
    write(root/"original.json",orig)
    c={"terrain_friction":terrain,"old_floor_override":OLD,"batches":BATCHES,"trials_per_batch":TRIALS,"ticks_per_trial":TICKS,
       "max_bp_per_batch":BP_STEPS,"noise_sd":SIGMA,"learning_rate":LR,"anchor_weight":ANCHOR,
       "audit_seeds":AUDIT,"trial_seeds":[400+i for i in range(BATCHES*TRIALS)],
       "update_rule":"PPO clipped Gaussian latent log-probability, GAE ridge-value baseline, same old-floor action anchor",
       "reward":"delta x + .006 per non-fall step - .75 on fall - .0004 mean normalized torque^2"}
    write(root/"config.json",c)
    actor=compile_weights(orig)
    old=[run(actor,s,OLD) for s in ANCHORS]
    x=np.concatenate([d["data"]["x"] for d in old])
    y=np.concatenate([d["data"]["mean"] for d in old])
    np.savez(root/"anchor.npz",center=x.mean(0),scale=np.maximum(x.std(0),.07),x=x,y=y)
    write(root/"anchor.json",{"seeds":ANCHORS,"stats":[d["stats"] for d in old]})
    n=Net(orig,x.mean(0),np.maximum(x.std(0),.07))
    np.testing.assert_allclose(n(torch.from_numpy(x)).detach().numpy(),y,atol=1e-12,rtol=1e-12)
    for name in ("bp","astra"):
        p=root/name; p.mkdir()
        write(p/"ckpt-0.json",orig)


def collect(root,arm,batch):
    c=config(root); p=root/arm; t=time.perf_counter()
    spec=read(p/f"ckpt-{batch-1}.json")
    policy=compile_weights(spec)
    found=[]
    seeds=c["trial_seeds"][(batch-1)*TRIALS:batch*TRIALS]
    for seed in seeds:
        rng=np.random.default_rng(199000+seed)
        found.append(run(policy,seed,c["terrain_friction"],rng.normal(0,SIGMA,(TICKS,6))))
    z=[d["data"] for d in found]
    x,adv=rewards_to_adv(z)
    data={"x":x,"a":np.concatenate([d["a"] for d in z]),"mean":np.concatenate([d["mean"] for d in z]),"adv":adv}
    np.savez(p/f"batch-{batch}.npz",**data)
    center,scale,xh,yh=pack(root)
    model=Net(spec,center,scale)
    block=norm_info(model,bundle(data),xh,yh)
    healthy_err=float((model(xh)-yh).detach().square().mean().sqrt())
    rows=[{"seed":s,**f["stats"]} for s,f in zip(seeds,found)]
    write(p/f"feedback-{batch}.json",{"batch":batch,"rows":rows,"passes":sum(r["pass"] for r in rows),
          "physical_ticks":sum(r["ticks"] for r in rows),"gradient_l2":block,
          "rms_old_floor_action_drift_before_update":healthy_err,"lab_seconds":time.perf_counter()-t})
    print(arm,batch,"noisy passes",sum(r["pass"] for r in rows),"ticks",sum(r["ticks"] for r in rows),flush=True)


def params(spec):
    return {(l,i,t["feature"]):t["coef"] for l,x in enumerate(spec["layers"]) for i,r in enumerate(x["rows"]) for t in r["terms"]} | {(l,i,"bias"):r["bias"] for l,x in enumerate(spec["layers"]) for i,r in enumerate(x["rows"])}


def accepted_region(changed,region):
    if region=="all": return True
    if region=="none": return not changed
    rows={"hips":(0,3),"knees":(1,4),"ankles":(2,5)}
    layer={"first":0,"middle":1,"head":2,**{k:2 for k in rows}}
    return all(i==layer[region] and (region not in rows or row in rows[region]) for i,row,_ in changed)


def apply(root,arm,batch,region,reason):
    p=root/arm; t=time.perf_counter()
    spec=read(p/f"ckpt-{batch-1}.json")
    updated=spec
    c={"action": "skip" if region=="none" else "update", "block":region,"reason":reason,
       "optimizer_steps":0,"proposal_scale":0.}
    if region!="none":
        data=np.load(p/f"batch-{batch}.npz")
        x,action,old,adv=bundle(data)
        center,scale,xh,yh=pack(root)
        net=Net(spec,center,scale)
        opt=torch.optim.Adam(net.parameters(),lr=LR)
        rng=torch.Generator().manual_seed(81000+batch)
        for _ in range(BP_STEPS):
            i=torch.randint(len(x),(256,),generator=rng)
            j=torch.randint(len(xh),(256,),generator=rng)
            loss=objective(net,(x[i],action[i],old[i],adv[i]),xh[j],yh[j])
            opt.zero_grad()
            loss.backward()
            mask_gradients(net,region)
            torch.nn.utils.clip_grad_norm_(net.parameters(),.8)
            opt.step()
        # Both arms share an action trust region; interpolation does not use extra rollouts.
        with torch.no_grad():
            original_h=net.original(xh)
            factor=1.
            for attempt in range(8):
                new=net(x)
                new_h=net(xh)
                rms=float((new-old).square().mean().sqrt())
                peak=float((new-old).abs().max())
                change_h=float((new_h-original_h).square().mean().sqrt())
                if rms<=.018 and peak<=.085 and change_h<=.012:
                    break
                factor*=.5
                for v in net.parameters(): v.mul_(.5)
            else:
                raise RuntimeError("bounded PPO update could not satisfy action trust region")
            updated=net.export()
            idx=list(range(0,len(x),max(1,len(x)//19)))[:19]
            evaluator=compile_weights(updated)
            # Independent comparison of deployed actor to torch before accepting it.
            fidx=[k for k in idx]
            ref=net(x[fidx]).numpy()
            arrays=[(x[k].numpy(),ref[j]) for j,k in enumerate(fidx)]
            for v,y in arrays:
                s={key:float(v[i]) for i,key in enumerate(FEATURES) if key not in ("1","target_v")
                   and not re.fullmatch(r"(?:d?q|contact)\d",key)}
                s.update(q=v[6:12].tolist(),dq=v[12:18].tolist(),contact=v[18:20].tolist())
                np.testing.assert_allclose(evaluator(s,.5),y,atol=1e-12,rtol=1e-12)
        c.update(optimizer_steps=BP_STEPS,proposal_scale=factor,batch_action_delta_rms=rms,
                 batch_action_delta_peak=peak,old_floor_rms_change_this_update=change_h)
    a,b=params(spec),params(updated)
    assert a.keys()==b.keys()
    diff=[k for k in a if a[k]!=b[k]]
    assert accepted_region(diff,region)
    c.update(changed_parameters=len(diff),lab_seconds=time.perf_counter()-t)
    write(p/f"ckpt-{batch}.json",updated)
    write(p/f"update-{batch}.json",c)
    print(arm,batch,"apply",region,c["optimizer_steps"],"BP steps",len(diff),"params",flush=True)


def auto(root):
    for b in range(1,BATCHES+1):
        collect(root,"bp",b)
        apply(root,"bp",b,"all","fixed all-parameter update")


def audit(root):
    c=config(root); t=time.perf_counter()
    data={"terrain_friction":c["terrain_friction"],"seeds":AUDIT,"arms":{}}
    for arm in ("bp","astra"):
        a=root/arm
        if not (a/f"ckpt-{BATCHES}.json").exists(): raise ValueError(f"{arm} not locked")
        curve=[]; n=0
        for b in range(BATCHES+1):
            policy=compile_weights(read(a/f"ckpt-{b}.json"))
            runs=[run(policy,s,c["terrain_friction"])["stats"] for s in AUDIT]
            n+=0 if not b else read(a/f"feedback-{b}.json")["physical_ticks"]
            curve.append({"batch":b,"physical_train_ticks":n,"trial_budget_used":b*TRIALS,
                          "pass":sum(r["pass"] for r in runs),"n":len(runs),
                          "mean_time_s":sum(r["elapsed_s"] for r in runs)/len(runs),
                          "mean_distance_m":sum(r["distance_m"] for r in runs)/len(runs),"rows":runs})
        prev=compile_weights(read(a/"ckpt-0.json"))
        final=compile_weights(read(a/f"ckpt-{BATCHES}.json"))
        old_b=[run(prev,s,OLD)["stats"] for s in AUDIT]
        old_a=[run(final,s,OLD)["stats"] for s in AUDIT]
        data["arms"][arm]={"curve":curve,
             "return_to_old_ground":{"before":sum(r["pass"] for r in old_b),"after":sum(r["pass"] for r in old_a),
                     "lost":sum(x["pass"] and not y["pass"] for x,y in zip(old_b,old_a)),"n":len(AUDIT),
                     "rows":[{"seed":s,"before":x,"after":y} for s,x,y in zip(AUDIT,old_b,old_a)]},
             "updates":[read(a/f"update-{b}.json") for b in range(1,BATCHES+1)],
             "feedback":[read(a/f"feedback-{b}.json") for b in range(1,BATCHES+1)]}
    assert data["arms"]["bp"]["curve"][0]==data["arms"]["astra"]["curve"][0]
    assert data["arms"]["bp"]["feedback"][0]["rows"]==data["arms"]["astra"]["feedback"][0]["rows"]
    for arm in ("bp","astra"):
        curve=data["arms"][arm]["curve"]
        data["arms"][arm]["first_sustained_8_of_12_after_trials"]=next(
            (x["trial_budget_used"] for x,y in zip(curve,curve[1:]) if x["pass"]>=8 and y["pass"]>=8),None)
    data["audit_seconds"]=time.perf_counter()-t
    write(root/"results.json",data)
    print({arm:( [x["pass"] for x in v["curve"]],v["return_to_old_ground"]["after"]) for arm,v in data["arms"].items()})


def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("op",choices=["setup","collect","apply","auto","audit"])
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--terrain",type=float)
    p.add_argument("--arm",choices=["bp","astra"])
    p.add_argument("--batch",type=int,choices=range(1,BATCHES+1))
    p.add_argument("--region",choices=(*REGIONS,"none"))
    p.add_argument("--reason",default="")
    a=p.parse_args()
    if a.op=="setup": setup(a.root,a.checkpoint,a.terrain)
    elif a.op=="collect": collect(a.root,a.arm,a.batch)
    elif a.op=="apply": apply(a.root,a.arm,a.batch,a.region,a.reason)
    elif a.op=="auto": auto(a.root)
    else: audit(a.root)


if __name__=="__main__": main()
