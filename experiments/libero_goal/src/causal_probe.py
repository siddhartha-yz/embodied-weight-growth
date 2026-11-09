"""Diagnostic only: a same-instruction, language-row activation transplant."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch

from paired_env import PairedEnv
from screen import PUBLIC, run_episode, save
from smol_actor import SmolActor


def reset_snapshot(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in PUBLIC}


@contextmanager
def language_value_hook(actor, layer, replacement=None):
    model=actor.policy.model
    projection=model.vlm_with_expert.lm_expert.layers[layer].self_attn.v_proj
    embed=model.embed_prefix
    saved={"calls":0}

    def embed_and_mark(images,img_masks,lang_tokens,lang_masks,state=None):
        out=embed(images,img_masks,lang_tokens,lang_masks,state)
        state_positions=torch.nonzero(out[2][0],as_tuple=True)[0]
        if not len(state_positions):
            raise RuntimeError("Did not find the appended state token in the prefix")
        start=int(state_positions[0])-lang_tokens.shape[1]
        rows=torch.nonzero(lang_masks[0],as_tuple=True)[0]+start
        if start<0 or not len(rows):
            raise RuntimeError("Invalid language token span")
        saved["rows"]=rows
        saved["prefix"]=out[0].shape[1]
        saved["tokens"]=lang_tokens[0,lang_masks[0]].detach().cpu().tolist()
        return out

    def value(_layer,_args,out):
        rows=saved["rows"]
        if out.ndim!=3 or out.shape[0]!=1 or out.shape[1]!=saved["prefix"]:
            raise RuntimeError(f"Expected cached prefix V projection [1,P,D], got {out.shape}")
        vals=out[:,rows,:]
        saved["calls"]+=1
        if "value" not in saved:
            saved["value"]=vals.detach().clone()
        elif not torch.equal(vals,saved["value"]):
            raise RuntimeError("Cross-attention V changed across denoising steps")
        if replacement is not None:
            if saved["tokens"]!=replacement["tokens"] or vals.shape!=replacement["value"].shape:
                raise RuntimeError("Donor/recipient language span mismatch")
            copy=out.clone()
            copy[:,rows,:]=replacement["value"].to(copy)
            return copy
        return None

    model.embed_prefix=embed_and_mark
    handle=projection.register_forward_hook(value)
    try:
        yield saved
    finally:
        handle.remove()
        model.embed_prefix=embed
    if not saved["calls"]:
        raise RuntimeError("Selected expert projection was never called")


class FirstChunkOnly:
    def __init__(self,actor,layer,donor):
        self.actor,self.layer,self.donor=actor,layer,donor

    def reset(self,seed):
        self.actor.reset(seed)
        self.position=0
        self.chunk=None
        self.audit=None

    def act(self,obs,instruction):
        if self.position==0:
            with language_value_hook(self.actor,self.layer,self.donor) as audit:
                self.chunk=self.actor.predict_action_chunk(self.actor.prepare(obs,instruction))[0]
            self.audit={"projection_calls":audit["calls"],"language_rows":len(audit["rows"])}
        n=self.actor.policy.config.n_action_steps
        pos=self.position
        self.position+=1
        if pos<n:
            return self.chunk[pos].cpu().numpy()
        return self.actor.act(obs,instruction)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--init",type=int,required=True)
    p.add_argument("--goal",choices=("bowl","bottle"),required=True)
    p.add_argument("--render-gpu",type=int,default=5)
    args=p.parse_args()
    args.run.mkdir()
    target=f"init-{args.init:03d}-{args.goal}"
    clean=args.screen/"canonical"/target
    shifted=args.screen/"swap_xy"/target
    a=json.loads((clean/"result.json").read_text())
    b=json.loads((shifted/"result.json").read_text())
    if not a["success"] or a["instruction"]!=b["instruction"] or not b["wrong_object_lifted_first"]:
        raise ValueError("Require a solved donor and an observed wrong-object lift for the exact same instruction")
    seed=b["seed"]
    actor=SmolActor(n_action_steps=10)
    donor_batch=actor.prepare(reset_snapshot(clean/"initial-public.npz"),a["instruction"])
    receiver_batch=actor.prepare(reset_snapshot(shifted/"initial-public.npz"),b["instruction"])
    actor.reset(seed)
    noise=actor.make_noise()
    rows=[]
    with PairedEnv(pair="objects",camera_size=256,render_gpu_device_id=args.render_gpu) as env:
        for layer in (5,9,13):
            with language_value_hook(actor,layer) as donor:
                actor.predict_action_chunk(donor_batch,noise=noise.clone())
            with language_value_hook(actor,layer) as recipient:
                base=actor.predict_action_chunk(receiver_batch,noise=noise.clone())
            with language_value_hook(actor,layer,recipient) as _sham:
                sham=actor.predict_action_chunk(receiver_batch,noise=noise.clone())
            if not torch.equal(base,sham):
                raise RuntimeError("Self-transplant should preserve the action chunk bitwise")
            out=args.run/f"expert-layer-{layer:02d}"
            out.mkdir()
            control=FirstChunkOnly(actor,layer,donor)
            row,obs,state=run_episode(env,control,args.goal,args.init,seed,300,out,
                                       video=True,init_source="bowl",layout="swap_xy")
            with np.load(shifted/"initial-public.npz",allow_pickle=False) as source:
                if not np.array_equal(state,source["sim_state"]) or any(not np.array_equal(obs[k],source[k]) for k in PUBLIC):
                    raise RuntimeError("Recipient native initial state differed from baseline")
            rows.append({**row,"expert_layer":layer,"self_transplant_bitwise_equal":True,**control.audit})
            save(args.run/"summary.json",{"donor":a,"receiver":b,"diagnostic":rows})
            print(rows[-1],flush=True)


if __name__=="__main__":
    main()
