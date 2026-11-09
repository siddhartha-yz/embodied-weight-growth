"""Extend language-value intervention from one chunk through target acquisition."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from causal_probe import language_value_hook
from paired_env import PairedEnv
from screen import PUBLIC, run_episode, save
from smol_actor import SmolActor

LAYERS=(5,13)
CASES=((2,"bottle"),(3,"bowl"))


def check_initial(out, ref):
    with np.load(out/"initial-public.npz",allow_pickle=False) as got, np.load(ref/"initial-public.npz",allow_pickle=False) as base:
        for k in (*PUBLIC,"sim_state"):
            if not np.array_equal(got[k],base[k]):
                raise AssertionError(f"Reset {k} differs from the paired screen")


class Capture:
    """Capture normal donor language values at each replan, without altering it."""

    def __init__(self, actor, plan_count):
        self.actor, self.plan_count = actor, plan_count

    def reset(self,seed):
        self.actor.reset(seed)
        self.plans={l:[] for l in LAYERS}

    def act(self, obs, instruction):
        if self.actor.queued_actions or len(self.plans[LAYERS[0]])>=self.plan_count:
            return self.actor.act(obs,instruction)
        with ExitStack() as stack:
            active={l:stack.enter_context(language_value_hook(self.actor,l)) for l in LAYERS}
            action=self.actor.act(obs,instruction)
        for l,capture in active.items():
            self.plans[l].append({"value":capture["value"],"tokens":capture["tokens"],
                                  "calls":capture["calls"]})
        return action


class Apply:
    def __init__(self, actor, layer, donor, plan_count):
        self.actor,self.layer,self.donor,self.plan_count=actor,layer,donor,plan_count

    def reset(self,seed):
        self.actor.reset(seed)
        self.audit=[]

    def act(self,obs,instruction):
        i=len(self.audit)
        if self.actor.queued_actions or i>=self.plan_count:
            return self.actor.act(obs,instruction)
        ordinal=min(i,len(self.donor)-1)
        with language_value_hook(self.actor,self.layer,self.donor[ordinal]) as h:
            result=self.actor.act(obs,instruction)
        self.audit.append({"step":i*self.actor.policy.config.n_action_steps,
                           "donor_step":ordinal*self.actor.policy.config.n_action_steps,
                           "projection_calls":h["calls"],"language_rows":len(h["rows"])})
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--render-gpu",type=int,default=5)
    args=p.parse_args()
    args.run.mkdir()
    actor=SmolActor(n_action_steps=10)
    cadence=actor.policy.config.n_action_steps
    result={"intervention":"same instruction, only expert V projection's valid language rows, through reach",
            "cases":[]}
    with PairedEnv(pair="objects",camera_size=256,render_gpu_device_id=args.render_gpu) as env:
        for init,goal in CASES:
            name=f"init-{init:03d}-{goal}"
            canonical=args.screen/"canonical"/name
            receiver=args.screen/"swap_xy"/name
            don_ref=json.loads((canonical/"result.json").read_text())
            rec_ref=json.loads((receiver/"result.json").read_text())
            if not don_ref["success"] or not rec_ref["wrong_object_lifted_first"] or don_ref["instruction"]!=rec_ref["instruction"]:
                raise AssertionError("Only compare real successful donors and wrong-object recipients")
            seed=rec_ref["seed"]
            prefix=(don_ref["first_lift_step"]-1)//cadence+1
            reach=(rec_ref["first_lift_step"]+cadence-1)//cadence+1
            group=args.run/name
            group.mkdir()
            out=group/"donor"
            out.mkdir()
            cap=Capture(actor,prefix)
            don,_,_=run_episode(env,cap,goal,init,seed,300,out,init_source="bowl")
            check_initial(out/name,canonical)
            if don!=don_ref or len(cap.plans[LAYERS[0]])!=prefix:
                raise AssertionError("Native donor changed while capturing activation")
            original=json.loads((canonical/"trace.json").read_text())
            logged=json.loads((out/name/"trace.json").read_text())
            for actual,ref in zip(logged,original,strict=True):
                if actual["action"]!=ref["action"] or actual["eef_pos"]!=ref["eef_pos"]:
                    raise AssertionError("Read-only donor hooks changed the actual rollout")
            row={"canonical":don,"untreated":rec_ref,"donor_replans":prefix,
                 "intervention_replans":reach,"layers":[]}
            result["cases"].append(row)
            for layer in LAYERS:
                parent=group/f"layer-{layer:02d}"
                parent.mkdir()
                intervention=Apply(actor,layer,cap.plans[layer],reach)
                got,_,_=run_episode(env,intervention,goal,init,seed,300,parent,
                    video=True,init_source="bowl",layout="swap_xy")
                check_initial(parent/name,receiver)
                item={**got,"layer":layer,"replans":intervention.audit}
                row["layers"].append(item)
                save(args.run/"summary.json",result)
                print(json.dumps(item),flush=True)


if __name__=="__main__":
    main()
