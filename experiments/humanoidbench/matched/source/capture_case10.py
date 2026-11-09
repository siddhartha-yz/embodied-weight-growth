"""Capture frozen matched case 10 with native foot contacts for visualization."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from actor_source import load
from compare import window
from edits import sparse_edit
from matched import (GatedPolicy, activity_stats, check_prefix, core_results,
                     recovery, write)
from push_env import DT, Scenario, _borrow_native_env, describe, reusable_native_env, rollout

site = Path(__file__).resolve().parent
prior = Path(sys.argv[1])
out = Path(sys.argv[2])
old = json.loads((site/"run/summary.json").read_text())
frozen = old["push"][9]
scenario = Scenario(**frozen["scenario"])
assert frozen["case"] == 10 and scenario.push_step == 788
out.mkdir()
actor = load(prior/"drq_checkpoint/actor_bundle.npz")
weights = sparse_edit((prior/"run/wsp/selected.json").read_text(), actor.params)
channels = describe()["channels"]
scales = np.asarray([(c["max_rad"]-c["min_rad"])/2 for c in channels])
cap = old["config"]["shared_l2_cap_rad"]
hips = (1, 6)


def collect(policy, scene, native):
    model, data = native.model, native.data
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    feet = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name): label
            for name, label in (("left_ankle_link", "left"), ("right_ankle_link", "right"))}
    assert min(floor,pelvis,*feet) >= 0
    wrench = np.empty(6)
    capture = []

    def monitor(obs):
        step = policy.step
        action = policy(obs)
        if step < scene_start:
            return action
        forces = {"left": 0., "right": 0.}
        nonfoot = {}
        supports = {}
        for i in range(data.ncon):
            c = data.contact[i]
            if c.efc_address < 0:
                continue
            if c.geom1 == floor:
                other = c.geom2
            elif c.geom2 == floor:
                other = c.geom1
            else:
                continue
            body = int(model.geom_bodyid[other])
            mujoco.mj_contactForce(model, data, i, wrench)
            f = float(wrench[0])
            if f <= 0.:
                continue
            if body in feet:
                key = feet[body]
                forces[key] += f
                supports[key] = supports.get(key, np.zeros(3)) + f*c.pos
            else:
                key = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body)
                nonfoot[key] = nonfoot.get(key,0.) + f
        delta = None
        if policy.activity and policy.activity[-1]["step"] == step:
            base, _ = actor.forward(obs)
            delta = ((action - base) * scales * 180/np.pi)[[1,6,10]].tolist()
        capture.append({"step": step, "time_s": step*DT,
                        "feet_force_n": forces, "other_floor_force_n": nonfoot,
                        "feet_contact_center_m": {k:(v/forces[k]).tolist() for k,v in supports.items()},
                        "feet_body_pos_m": {label:data.xpos[body].tolist() for body,label in feet.items()},
                        "body_com_m": data.subtree_com[pelvis].tolist(),
                        "delta_target_deg_hips_torso": delta})
        return action

    scene_start = scenario.push_step - 32
    r = rollout(monitor, scene, trace_every=1, record_support=True)
    r.update(gate=policy.gate.event, activity=policy.activity, activity_stats=activity_stats(policy.activity),
             contacts=capture)
    return r


with reusable_native_env():
    with _borrow_native_env() as env:
        native = env.unwrapped
    arms = {}
    for key in ("clean", "base", "wsp", "code"):
        p = GatedPolicy(actor, "base" if key == "clean" else key, scales, hips, cap, weights=weights)
        scene = replace(scenario, push_step=None, lateral_impulse_ns=0) if key == "clean" else scenario
        arms[key] = collect(p, scene, native)
    ref = window(arms["clean"]["trace"], scenario)
    assert ref.tolist() == frozen["reference"]
    base = arms["base"]
    for k,r in arms.items():
        if k != "clean":
            check_prefix(base,r,base["gate"],r["gate"])
            recovery(r,scenario,ref)
        assert core_results(r) == frozen["arms"][k], k
    write(out/"case10.json", {**frozen,"arms":arms,"source":"Exact frozen matched-case10 replay + native contacts"})
print("case10 verified against saved matched metrics and per-tick correction audits",flush=True)
