"""Warm-start supervised BP with behavior retention, in the unchanged Walker2d policy."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import round2 as r2
from distill_v8 import states_array, teacher_torques, USE, STATE_SIGMA
from walk_plant import WalkPlant
from walk_policy import compile_policy
from walk_server import Session

p = r2.previous
# Final test remains round2's seed 320..331 plus the two named reference cases.
STABLE_TRAIN = [(271,.02),(272,.02),(307,.02),(308,.02),(309,.02),(310,.015),(311,.025)]
FAIL_TRAIN = [(271,.08),(302,.08),(304,.075),(305,.085),(306,.08),(307,.07),(308,.09)]
STABLE_DEV = r2.DEV_HEALTHY + [(312,.02),(313,.02)]
FAIL_DEV = r2.DEV_TARGET + [(314,.07),(315,.09)]
CHECKPOINTS = (200, 600, 1200)
LAMBDAS = (.70, 2.5, 8.0)
AGGREGATE = (FAIL_TRAIN[0],FAIL_TRAIN[1],FAIL_TRAIN[3],FAIL_TRAIN[5])


def wall():
    return time.perf_counter()


def play(arm, spec, case):
    seed, pitch = case
    plant=WalkPlant(seed,"train")
    plant.start_q[2]=pitch
    s=Session(arm,seed,"train",frozen_spec=spec,plant=plant)
    r=s.request({"op":"run","ticks":250})
    if not r["ok"]:
        raise RuntimeError(r["error"])
    r["scene"]["initial_pitch_rad"]=pitch
    return r


def available_states(r):
    # The last state has no subsequent observed action.
    return r["trace"][:-1]


def num(spec):
    return sum(len(r["terms"])+1 for l in spec["layers"] for r in l["rows"])


class Net(nn.Module):
    def __init__(self,spec,center,scale):
        super().__init__()
        self.source=spec
        self.register_buffer("center",torch.tensor(center))
        self.register_buffer("scale",torch.tensor(scale))
        self.dw=nn.ParameterList()
        self.db=nn.ParameterList()
        width=len(center)
        for i,l in enumerate(spec["layers"]):
            w=np.zeros((len(l["rows"]),width))
            mask=np.zeros_like(w)
            b=[]
            for j,r in enumerate(l["rows"]):
                b.append(r["bias"])
                for t in r["terms"]:
                    w[j,t["feature"]]=t["coef"]
                    mask[j,t["feature"]]=1
            for key,v in (("w",w),("mask",mask),("b",np.asarray(b))):
                self.register_buffer(f"{key}{i}",torch.tensor(v))
            self.dw.append(nn.Parameter(torch.zeros_like(getattr(self,f"w{i}"))))
            self.db.append(nn.Parameter(torch.zeros_like(getattr(self,f"b{i}"))))
            width=len(l["rows"])

    def weights(self, i):
        return self.dw[i]*getattr(self,f"mask{i}")

    def original(self,x):
        for i in range(3):
            x=F.linear(x,getattr(self,f"w{i}"),getattr(self,f"b{i}"))
            if i < 2: x=x.tanh()
        return x.clamp(-1,1)

    def forward(self,x):
        for i in range(3):
            base=F.linear(x,getattr(self,f"w{i}"),getattr(self,f"b{i}"))
            z=(x-self.center)/self.scale if i==0 else x
            x=base+F.linear(z,self.weights(i),self.db[i])
            if i < 2: x=x.tanh()
        return x.clamp(-1,1)

    def export(self):
        layers=[]
        for i,l in enumerate(self.source["layers"]):
            dw=self.weights(i).detach()
            db=self.db[i].detach()
            if i==0:
                dw=dw/self.scale
                db=db-dw@self.center
            w=(getattr(self,f"w{i}")+dw).numpy()
            b=(getattr(self,f"b{i}")+db).numpy()
            rows=[{"bias":float(b[j]),"terms":[{"feature":t["feature"],"coef":float(w[j,t["feature"]])}
                   for t in row["terms"]]} for j,row in enumerate(l["rows"])]
            layers.append({"rows":rows})
        return {"layers":layers}


def augment(raw,rng,repeats,sigma):
    x=np.repeat(raw,repeats,axis=0)
    x[:,USE[:15]]+=rng.normal(size=(len(x),15))*STATE_SIGMA*sigma
    return np.concatenate((raw,x))


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(9008)
    rng=np.random.default_rng(9008)
    spec=p.read(args.checkpoint)
    teacher=p.read(args.teacher)["spec"]
    oracle=compile_policy("code",teacher)
    student=compile_policy("weights",spec)
    timing={"teacher_and_dataset_s":0.,"optimizer_s":0.,"development_replay_s":0.}
    counts={"dataset_rollouts":0,"development_rollouts":0,"training_steps":0}
    train_start=wall()
    t0=wall()
    stable=[p.roll(spec,c) for c in STABLE_TRAIN]
    failures=[p.roll(spec,c) for c in FAIL_TRAIN]
    demos=[play("code",teacher,c) for c in FAIL_TRAIN]
    counts["dataset_rollouts"]+=len(stable)+len(failures)+len(demos)
    assert p.stats(stable[0])["pass"] and not p.stats(failures[0])["pass"] and p.stats(demos[0])["pass"]
    successful_stable=[r for r in stable if p.stats(r)["pass"]]
    stable_raw=[s for r in successful_stable for s in available_states(r)]
    fail_raw=[s for r in failures for s in available_states(r)]
    demo_raw=[s for r in demos if p.stats(r)["pass"] for s in available_states(r)]
    raw_labelled=fail_raw+demo_raw
    real_x=states_array(raw_labelled)
    # The existing vectorized code-v8 evaluator is checked against the actual code DSL
    # for every real training state; augmentation is labeled by the same equations.
    np.testing.assert_allclose(teacher_torques(real_x),np.asarray([oracle(s,.5) for s in raw_labelled]),atol=1e-12,rtol=1e-12)
    hn=augment(states_array(stable_raw),rng,3,.30)
    ft=augment(real_x,rng,3,.45)
    center=np.mean(np.concatenate((hn,ft)),axis=0)
    scale=np.maximum(np.std(np.concatenate((hn,ft)),axis=0),.08)
    h=torch.from_numpy(hn)
    n0=Net(spec,center,scale)
    h_labels=n0.original(h).detach()
    # A zero update must exactly implement the old controller before fitting.
    torch.testing.assert_close(n0(h[:32]),h_labels[:32],rtol=0.,atol=0.)
    np.testing.assert_allclose(n0.original(torch.from_numpy(states_array(stable_raw[:40]))).numpy(),
                               [student(s,.5) for s in stable_raw[:40]],atol=1e-12,rtol=1e-12)
    timing["teacher_and_dataset_s"]+=wall()-t0
    t0=wall()
    dev_cases=STABLE_DEV+FAIL_DEV
    dev_base={c:p.roll(spec,c) for c in dev_cases}
    counts["development_rollouts"]+=len(dev_base)
    dev_healthy=[dev_base[c] for c in STABLE_DEV if p.stats(dev_base[c])["pass"]]
    dev_states=[s for r in dev_healthy for s in r["trace"]]
    vh=torch.from_numpy(states_array(dev_states))
    vh_y=n0.original(vh)
    timing["development_replay_s"]+=wall()-t0
    records=[]
    best=None
    best_spec=None

    def score(rec):
        h,d=rec["nominal"],rec["target"]
        pass_both=h["pass"] and d["pass"]
        return (pass_both, -rec["healthy_losses"], d["pass"], rec["dev_challenge_pass"],
                rec["dev_stable_pass"], -rec["nominal_max_torque_delta"],
                d["elapsed_s"], d["distance_m"])

    for stage in (1,2):
        if stage==2:
            t0=wall()
            dr=[p.roll(best_spec,c) for c in AGGREGATE]
            counts["dataset_rollouts"]+=len(dr)
            online=[s for r in dr for s in available_states(r)]
            ax=states_array(online)
            np.testing.assert_allclose(teacher_torques(ax),[oracle(s,.5) for s in online],atol=1e-12,rtol=1e-12)
            ft=np.concatenate((ft,augment(ax,rng,3,.35)))
            raw_labelled.extend(online)
            timing["teacher_and_dataset_s"]+=wall()-t0
        t0=wall()
        x=torch.from_numpy(ft)
        y=torch.from_numpy(teacher_torques(ft))
        timing["teacher_and_dataset_s"]+=wall()-t0
        for lam in LAMBDAS:
            t0=wall()
            net=Net(spec,center,scale)
            opt=torch.optim.Adam(net.parameters(),lr=.0010)
            timing["optimizer_s"]+=wall()-t0
            for i in range(CHECKPOINTS[-1]):
                t0=wall()
                fi=torch.randint(len(x),(384,))
                hi=torch.randint(len(h),(384,))
                fit=(net(x[fi])-y[fi]).square().mean()
                retain=(net(h[hi])-h_labels[hi]).square().mean()
                shrink=sum(t.square().mean() for t in (*net.dw,*net.db))
                loss=fit+lam*retain+.025*shrink
                opt.zero_grad()
                loss.backward()
                opt.step()
                timing["optimizer_s"]+=wall()-t0
                counts["training_steps"]+=1
                if i+1 not in CHECKPOINTS: continue
                t0=wall()
                candidate=net.export()
                assert num(candidate)==num(spec)
                cand_roll={c:p.roll(candidate,c) for c in dev_cases}
                counts["development_rollouts"]+=len(cand_roll)
                m={c:p.stats(rr) for c,rr in cand_roll.items()}
                error=(net(vh)-vh_y).detach()
                record={"stage":stage,"lambda":lam,"step":i+1,"nominal":m[p.NOMINAL],"target":m[p.TARGET],
                        "healthy_losses":sum(p.stats(dev_base[c])["pass"] and not m[c]["pass"] for c in STABLE_DEV),
                        "dev_stable_pass":sum(m[c]["pass"] for c in STABLE_DEV),
                        "dev_challenge_pass":sum(m[c]["pass"] for c in FAIL_DEV),
                        "nominal_rms_torque_delta":float(error.square().mean().sqrt()),
                        "nominal_max_torque_delta":float(error.abs().max()),
                        "case_outcomes":[{"seed":c[0],"pitch":c[1],"baseline_pass":p.stats(dev_base[c])["pass"],
                                          **m[c]} for c in dev_cases]}
                records.append(record)
                if best is None or score(record)>score(best):
                    best,best_spec=record,candidate
                timing["development_replay_s"]+=wall()-t0
                print(f"stage {stage} lambda {lam:g} step {i+1} dev target={record['target']['elapsed_s']:.2f}s "
                      f"normal losses={record['healthy_losses']} hard={record['dev_challenge_pass']}/4",
                      flush=True)
        # Both stages start each candidate from the unchanged checkpoint. Stage 2
        # adds labels on stage-1 candidate-visited states only.
    p.write(args.out/"bp-selected.json",best_spec)
    p.write(args.out/"selection.json",{"candidate":{"id":"bp-selected","qualified":best["nominal"]["pass"] and best["target"]["pass"] and not best["healthy_losses"]}})
    timing["data_to_locked_checkpoint_s"]=wall()-train_start
    counts.update({"normal_original_states":len(stable_raw),"normal_samples_with_augmentation":len(h),
                   "teacher_rollout_states":len(demo_raw),"teacher_labeled_real_states_total":len(raw_labelled),
                   "teacher_samples_with_augmentation":len(ft),"candidate_checkpoints":len(records)})
    audit={"timings":timing,"counts":counts,"selected":best,"checkpoints":records,
           "training":{"stable":STABLE_TRAIN,"fail":FAIL_TRAIN,"stable_dev":STABLE_DEV,"fail_dev":FAIL_DEV,
                       "gradient_objective":"teacher action MSE + lambda * original policy action MSE on healthy states + 0.025 delta-L2",
                       "teacher_target_demo":p.stats(demos[0]),"runtime":"one CPU torch thread; existing sparse edges only"}}
    p.write(args.out/"bp-fit.json",audit)
    print("Locked selected",best["stage"],best["lambda"],best["step"],timing,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=["train","final"])
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--teacher",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if a.mode=="train":
        if (a.out/"selection.json").exists(): raise ValueError("refusing to reselect a locked experiment")
        train(a)
    else:
        if (a.out/"final-results.json").exists(): raise ValueError("held-out evaluation already exists")
        t=wall()
        r2.final(p.read(a.checkpoint),a.out)
        p.write(a.out/"final-timing.json",{"matched_evaluation_s":wall()-t,"rollouts":52})


if __name__=="__main__":
    main()
