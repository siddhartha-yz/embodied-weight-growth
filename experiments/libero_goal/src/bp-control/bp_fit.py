"""Fit 16 existing SmolVLA coefficients through fixed-noise native action sampling."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"multitask-repair")]
from matrix import head, read, save, public_observation, PROMPT
from layouts import BowlOffsetEnv
from smol_actor import SmolActor
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

REPAIR_AXES=(1,2)
PRESERVE_AXES=tuple(range(7))
GROUP_WEIGHTS={"repair-bottle":.5,"repair-bowl":.5,"preserve":1.}
OLD={"bottle":"clean-bottle-2","bowl":"clean-bowl-2"}
BOUND=.65


def cpu_tree(item):
    if torch.is_tensor(item):
        return item.detach().cpu().clone()
    if isinstance(item,dict):
        return {k:cpu_tree(v) for k,v in item.items()}
    if isinstance(item,list):
        return [cpu_tree(v) for v in item]
    if isinstance(item,tuple):
        return tuple(cpu_tree(v) for v in item)
    return item


def gpu_tree(item,device):
    if torch.is_tensor(item):
        return item.to(device)
    if isinstance(item,dict):
        return {k:gpu_tree(v,device) for k,v in item.items()}
    if isinstance(item,list):
        return [gpu_tree(v,device) for v in item]
    if isinstance(item,tuple):
        return tuple(gpu_tree(v,device) for v in item)
    return item


def cat_tree(items):
    first=items[0]
    if torch.is_tensor(first):
        return torch.cat(items,dim=0)
    if isinstance(first,dict):
        return {k:cat_tree([it[k] for it in items]) for k in first}
    if isinstance(first,tuple):
        return tuple(cat_tree([it[i] for it in items]) for i in range(len(first)))
    raise TypeError(type(first))


@torch.no_grad()
def prefix(actor,batch):
    """The unchanged model's cache for one observation; action-head edits don't touch it."""
    policy=actor.policy
    model=policy.model
    if not model.config.use_cache or policy.config.adapt_to_pi_aloha or model._rtc_enabled():
        raise ValueError("Expected the fixed LIBERO sampler with prefix cache and no RTC/adaptation")
    batch=policy._prepare_batch(dict(batch))
    images,img_masks=policy.prepare_images(batch)
    state=policy.prepare_state(batch)
    emb,pad,att=model.embed_prefix(images,img_masks,batch[OBS_LANGUAGE_TOKENS],
                                   batch[OBS_LANGUAGE_ATTENTION_MASK],state=state)
    _, cache=model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad,att),
        position_ids=pad.cumsum(1)-1,past_key_values=None,inputs_embeds=[emb,None],
        use_cache=True,fill_kv_cache=True)
    return pad,cache


def batch_samples(rows,batch_size):
    # A tokenizer may produce more than one prefix length; concatenate only equal shapes.
    buckets=defaultdict(list)
    for row in rows:
        buckets[(row["group"],row["input"]["prefix"][0].shape[1])].append(row)
    return [bucket[i:i+batch_size] for bucket in buckets.values() for i in range(0,len(bucket),batch_size)]


def denoise(actor,prefix_data,noise,affine):
    """Same Euler integration and native affine output as sample_actions + postprocess."""
    model=actor.policy.model
    pad,cache=prefix_data
    dt=-1./model.config.num_steps
    a=noise
    for j in range(model.config.num_steps):
        t=torch.tensor(1.+j*dt,dtype=torch.float32,device=a.device).expand(a.shape[0])
        a=a+dt*model.denoise_step(prefix_pad_masks=pad,past_key_values=cache,x_t=a,timestep=t)
    zero,scale=affine
    return a[:,:10,:7]*scale+zero


def native_affine(actor):
    """Obtain checkpoint's per-axis mean/std directly; avoid CPU transfer inside gradients."""
    from safetensors.torch import load_file
    d=load_file(ROOT/"models/smolvla_libero/policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    return d["action.mean"].to(actor.device),d["action.std"].to(actor.device)


def src_chunk(trace,t):
    a=np.asarray([r["action"] for r in trace[t:t+10]],dtype=np.float32)
    if a.shape!=(10,7):
        raise ValueError(f"incomplete native action chunk at {t}")
    return a


def replay(actor,env,result,trace,initial,requests,affine):
    """Reconstruct camera/proprio inputs and seeded replan noise while replaying actions."""
    actor.reset(result["seed"])
    reset=env.reset(goal=result["goal"],init_index=result["init_index"],seed=result["seed"],
                    init_source=result.get("init_source","bowl"),layout=result["layout"])
    obs=public_observation(reset)
    with np.load(initial,allow_pickle=False) as ref:
        assert np.array_equal(env.sim_state(),ref["sim_state"])
        assert all(np.array_equal(v,ref[k]) for k,v in obs.items())
    rows=[]
    for t in range(max(requests)+1):
        if t%10==0:
            noise=actor.make_noise()
        if t in requests:
            req=requests[t]
            prompt=PROMPT if t<80 and result.get("prompt_mode")=="historical_bottle" else result["instruction"]
            batch=actor.prepare(obs,prompt)
            # Inference tensors cannot be saved by autograd; clone frozen cache outside inference.
            pref=prefix(actor,batch)
            official=actor.predict_action_chunk(batch,noise.clone())[0,:10].detach().cpu()
            with torch.no_grad():
                ours=denoise(actor,pref,noise,affine)[0].cpu()
            diff=(ours-official).abs().max().item()
            if diff>1e-6:
                raise ValueError(f"native sampler mismatch {req['id']} {diff}")
            if req["group"]=="preserve" and np.max(np.abs(official.numpy()-req["target"]))>1e-6:
                raise ValueError(f"stock recorded action mismatch {req['id']}")
            row={**req,"t":t,"case":result["id"],"goal":result["goal"],
                 "stock":official.tolist(),"target":req["target"].tolist(),
                 "observation":{"eef_m":obs["robot0_eef_pos"].tolist(),
                                "target_m":env.status()["object_xyz_m"][result["goal"]],
                                "grasped":env.status()["grasped_objects"],"instruction":prompt},
                 "input":{"prefix":cpu_tree(pref),"noise":cpu_tree(noise)}}
            rows.append(row)
        # Each target input is captured before taking the next teacher action.
        step=trace[t]
        raw,_,done,_=env.step(np.asarray(step["action"],dtype=np.float64))
        obs=public_observation(raw)
        status=env.status()
        if (not np.array_equal(status["object_xyz_m"][result["goal"]],step["target"])
                or not np.array_equal(obs["robot0_eef_pos"],step["eef"])
                or status["grasped_objects"]!=step["held"] or bool(done)!=step["native"]):
            raise ValueError(f"physical replay diverged: {result['id']} t={t} group={list(requests.values())[0]['group']}")
    return rows


def canonical_preserve_ticks(result,dev):
    # Each training case contributes approach and acquisition; known good canonical also placement.
    grasp=((result["first_lift_step"]-1)//10)*10
    ticks={10,max(0,grasp-10),grasp}
    if not dev:
        place=((result["first_native_goal"]-1)//10)*10
        ticks|={max(grasp,place-10),place}
    return sorted(ticks)


def verify_teacher(branch,teacher):
    stable=teacher["stable_released_at"]
    g=teacher["case"]["goal"]
    if type(stable) is not int or stable<40 or stable>len(branch):
        raise ValueError(f"no verified 40-step teacher placement for {g}")
    if not all(row["native"] and g not in row["held"] for row in branch[stable-40:stable]):
        raise ValueError(f"teacher trace has no 40-step release for {g}")


def collect(actor,args):
    aff=native_affine(actor)
    coords_in=read(args.coords)["edits"]
    coords=sorted({(e["row"],e["column"]) for e in coords_in})
    if len(coords)!=16:
        raise ValueError("The registered experiment must expose 16 unique coordinates")
    base=head(actor).weight
    coord_meta=[{"row":r,"column":c,"stock":float(base[r,c])} for r,c in coords]
    teachers=read(args.teachers)
    if set(teachers)!={"bottle","bowl"}:
        raise ValueError("Both verified teachers are needed for the joint objective")
    rows=[]
    with BowlOffsetEnv(pair="objects",camera_size=256,render_gpu_device_id=args.render_gpu) as env:
        for g in ("bottle","bowl"):
            teach=teachers[g]
            source=ROOT/teach["source"]
            result=read(source/"result.json")
            assert all(result[k]==teach["case"][k] for k in ("id","seed","layout","init_index","goal"))
            stock=read(source/"trace.json")
            branch=read(ROOT/teach["trace"])
            verify_teacher(branch,teach)
            requests={}
            for w in teach["windows"]:
                t=w["t"]
                assert t%10==0 and t not in requests
                target=src_chunk(branch,t)
                assert np.array_equal(target,np.asarray(w["actions"],dtype=np.float32))
                requests[t]={"id":f"{result['id']}/repair/{t}","group":f"repair-{g}",
                             "target":target,"axes":REPAIR_AXES}
            t_first=min(requests)
            assert all(np.array_equal(a["action"],b["action"]) for a,b in zip(stock[:t_first],branch[:t_first]))
            assert g in branch[t_first-1]["held"]
            rows+=replay(actor,env,result,branch,source/"initial-public.npz",requests,aff)
            for base_dir,dev in ((source,True),(args.stock_dir/OLD[g],False)):
                result=read(base_dir/"result.json")
                trace=read(base_dir/"trace.json")
                ticks=canonical_preserve_ticks(result,dev)
                requests={t:{"id":f"{result['id']}/preserve/{t}","group":"preserve",
                             "target":src_chunk(trace,t),"axes":PRESERVE_AXES} for t in ticks}
                rows+=replay(actor,env,result,trace,base_dir/"initial-public.npz",requests,aff)
            print(f"Replay verified teacher and preservation for {g}",flush=True)
    return {"coordinates":coord_meta,"contexts":rows}


class SparseProjection(nn.Module):
    """Optimize a vector of offsets; all other projection weights and bias stay frozen."""
    def __init__(self,layer,coords):
        super().__init__()
        self.register_buffer("stock",layer.weight.detach())
        self.register_buffer("bias",layer.bias.detach())
        self.register_buffer("rows",torch.tensor([c["row"] for c in coords],device=layer.weight.device))
        self.register_buffer("cols",torch.tensor([c["column"] for c in coords],device=layer.weight.device))
        self.register_buffer("start",self.stock[self.rows,self.cols])
        self.delta=nn.Parameter(torch.zeros_like(self.start))
        self.absolute=None
    def forward(self,x):
        values=self.start+self.delta if self.absolute is None else self.absolute
        weight=self.stock.index_put((self.rows,self.cols),values)
        return F.linear(x,weight,self.bias)


def batches(dataset,actor,size):
    counts=Counter(r["group"] for r in dataset["contexts"])
    groups=[]
    for rows in batch_samples(dataset["contexts"],size):
        g=rows[0]["group"]
        a=torch.tensor(rows[0]["axes"],device=actor.device)
        chunk=cat_tree([r["input"] for r in rows])
        chunk=gpu_tree(chunk,actor.device)
        target=torch.tensor([r["target"] for r in rows],device=actor.device)[:,:,a]
        groups.append((rows,g,a,chunk,target,len(rows)/counts[g]))
    return groups


def output(actor,group,aff):
    return denoise(actor,group[3]["prefix"],group[3]["noise"],aff)


@torch.no_grad()
def predict_all(actor,groups,aff):
    return {row["id"]:action
            for group in groups for row,action in zip(group[0],output(actor,group,aff).cpu())}


def report(dataset,pred):
    grouped=defaultdict(list); details={}
    for r in dataset["contexts"]:
        e=(pred[r["id"]][:,r["axes"]]-torch.tensor(r["target"])[:,r["axes"]]).square().mean().item()
        grouped[r["group"]].append(e); details[r["id"]]=e
    m={k:sum(v)/len(v) for k,v in grouped.items()}
    return {"mse":m,"rmse":{k:math.sqrt(v) for k,v in m.items()},
            "objective":sum(m[g]*w for g,w in GROUP_WEIGHTS.items()),"context_mse":details}


def prepare(args,actor):
    if args.dataset.exists():
        raise FileExistsError(args.dataset)
    dataset=collect(actor,args)
    torch.save(dataset,args.dataset)
    aff=native_affine(actor)
    groups=batches(dataset,actor,args.batch)
    projection=SparseProjection(head(actor),dataset["coordinates"])
    actor.policy.model.action_out_proj=projection
    stock=predict_all(actor,groups,aff)
    errors={r["id"]:(stock[r["id"]]-torch.tensor(r["stock"])).abs().max().item() for r in dataset["contexts"]}
    print("Batched reproduction max native absolute error",max(errors.values()),flush=True)
    if max(errors.values())>1e-4:
        print("Using single-context forwards for exact native equality",flush=True)
        groups=batches(dataset,actor,1)
        stock=predict_all(actor,groups,aff)
        errors={r["id"]:(stock[r["id"]]-torch.tensor(r["stock"])).abs().max().item() for r in dataset["contexts"]}
    if max(errors.values())>1e-6 and len(groups)==len(dataset["contexts"]):
        raise ValueError(f"Single-context stock reproduction error {max(errors.values())}")
    probes=[]
    for i,c in enumerate(dataset["coordinates"]):
        with torch.no_grad():
            projection.delta[i]=.20
            post=predict_all(actor,groups,aff)
            projection.delta.zero_()
        probes.append({**c,"step":.20,"effects":{k:np.round((v-stock[k]).numpy().astype(np.float64),6).tolist()
                                                for k,v in post.items()}})
    payload={"coordinates":dataset["coordinates"],"contexts":[{k:v for k,v in r.items() if k!="input"}
                           for r in dataset["contexts"]],"probes":probes,
             "objective":{"formula":"0.5*repair_bottle_YZ_MSE + 0.5*repair_bowl_YZ_MSE + preserve_7axes_MSE",
                          "group_weights":GROUP_WEIGHTS,"bound_from_stock":BOUND,
                          "step_index":"zero-based t before action at control t+1"},
             "stock_fit":report(dataset,stock),"same_context_max_native_error":max(errors.values())}
    save(args.shared,payload)
    print("Shared data and forward probes ready",flush=True)


def fit(args,actor):
    dataset=torch.load(args.dataset,weights_only=True)
    groups=batches(dataset,actor,args.batch)
    aff=native_affine(actor)
    projection=SparseProjection(head(actor),dataset["coordinates"])
    actor.policy.model.action_out_proj=projection
    stock=predict_all(actor,groups,aff)
    if max((stock[r["id"]]-torch.tensor(r["stock"])).abs().max().item() for r in dataset["contexts"])>1e-4:
        groups=batches(dataset,actor,1)
        stock=predict_all(actor,groups,aff)
    before=report(dataset,stock)
    if args.score_edits:
        d=read(args.score_edits)["edits"]
        assigned={(q["row"],q["column"]):q["value"] for q in d}
        available={(q["row"],q["column"]) for q in dataset["coordinates"]}
        if not set(assigned)<=available or len(assigned)!=len(d):
            raise ValueError("Unexpected or duplicate coordinate in score-only manifest")
        values=[]
        for c in dataset["coordinates"]:
            val=assigned.get((c["row"],c["column"]),c["stock"])
            if not math.isfinite(val) or abs(val-c["stock"])>BOUND+1e-6:
                raise ValueError(f"Out-of-bounds score-only edit: {c} {val}")
            values.append(val)
        projection.absolute=torch.tensor(values,dtype=projection.start.dtype,device=actor.device)
        after=report(dataset,predict_all(actor,groups,aff))
        save(args.out,{"stock":before,"candidate":after})
        print(json.dumps({"stock":before["mse"],"candidate":after["mse"]}),flush=True)
        return
    opt=torch.optim.Adam([projection.delta],lr=.035)
    log=[]; best=before["objective"]; best_delta=projection.delta.detach().clone()
    start=time.perf_counter()
    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        total=0.
        for group in groups:
            _,g,axes,_,target,fraction=group
            actions=output(actor,group,aff)[:,:,axes]
            loss=F.mse_loss(actions,target)*GROUP_WEIGHTS[g]*fraction
            total+=float(loss.detach())
            loss.backward()
        grad=projection.delta.grad
        if grad is None or not bool(grad.isfinite().all()) or (i==0 and not bool((grad!=0).any())):
            raise ValueError("Missing, zero initial, or nonfinite gradients")
        log.append({"step":i,"objective":total,"grad_l2":float(grad.norm()),"gradient_nonzero":int((grad!=0).sum())})
        if total<best:
            best=total; best_delta=projection.delta.detach().clone()
        opt.step()
        with torch.no_grad():
            projection.delta.clamp_(-BOUND,BOUND)
        if i%5==0:
            print(f"Step {i} loss {total:.7f} |grad| {grad.norm():.6f}",flush=True)
    last=report(dataset,predict_all(actor,groups,aff))
    if last["objective"]<best:
        best=last["objective"]; best_delta=projection.delta.detach().clone()
    with torch.no_grad():
        projection.delta.copy_(best_delta)
    after=report(dataset,predict_all(actor,groups,aff))
    values=(projection.start+projection.delta.detach()).cpu().tolist()
    edits=[{"row":q["row"],"column":q["column"],"value":v} for q,v in zip(dataset["coordinates"],values)]
    save(args.out,{"edits":edits,"stock_fit":before,"bp_fit":after,"history":log,
                   "training_seconds":time.perf_counter()-start,
                   "training_batch_size":max(len(g[0]) for g in groups),"steps":args.steps})
    print(json.dumps({"stock":before["mse"],"BP":after["mse"],"wall_s":time.perf_counter()-start}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest="mode",required=True)
    prep=s.add_parser("prepare")
    prep.add_argument("--teachers",type=Path,required=True)
    prep.add_argument("--coords",type=Path,required=True)
    prep.add_argument("--stock-dir",type=Path,required=True)
    prep.add_argument("--dataset",type=Path,required=True)
    prep.add_argument("--shared",type=Path,required=True)
    prep.add_argument("--render-gpu",type=int,default=0)
    prep.add_argument("--batch",type=int,default=4)
    f=s.add_parser("fit")
    f.add_argument("--dataset",type=Path,required=True)
    f.add_argument("--out",type=Path,required=True)
    f.add_argument("--batch",type=int,default=4)
    f.add_argument("--steps",type=int,default=32)
    f.add_argument("--score-edits",type=Path)
    a=p.parse_args()
    actor=SmolActor(n_action_steps=10)
    if a.mode=="prepare": prepare(a,actor)
    else: fit(a,actor)

if __name__=="__main__":
    main()
