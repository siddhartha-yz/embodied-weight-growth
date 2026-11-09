"""One-shot, weight-only repair of the already trained Walker2d controller."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import re

from walk_plant import WalkPlant
from walk_policy import FEATURES, JOINT_ORDER
from walk_server import Session

NOMINAL = (271, 0.02)
TARGET = (271, 0.08)
CHALLENGE = [(s, 0.08) for s in (272, 273, 274, 275, 276, 277)] + [
    (s, p) for s, p in [(278, .07), (279, .07), (280, .07), (281, .09), (282, .09), (283, .09)]]
RETAIN = [(s, .02) for s in (272, 273, 274, 275, 276, 277)] + [
    (s, p) for s, p in [(278, .015), (279, .015), (280, .015), (281, .025), (282, .025), (283, .025)]]
MAX_EDITS = 12
MAX_DELTA = 0.30


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def roll(spec, case):
    seed, pitch = case
    plant = WalkPlant(seed, "train")
    plant.start_q[2] = pitch
    s = Session("weights", seed, "train", frozen_spec=spec, plant=plant)
    r = s.request({"op": "run", "ticks": 250})
    if not r["ok"]:
        raise RuntimeError(r["error"])
    r["scene"]["initial_pitch_rad"] = pitch
    return r


def stats(r):
    w = r["walk"]
    return {**{k: w[k] for k in ("distance_m", "elapsed_s", "alternating_steps", "fall_reason")},
            "pass": r["success"] and len(r["trace"]) == 251}


def patch(spec, proposal):
    fixed = copy.deepcopy(spec)
    edits = proposal["edits"]
    if not 1 <= len(edits) <= MAX_EDITS:
        raise ValueError("edit count outside registered budget")
    seen = set()
    changes = []
    for e in edits:
        key, unit, value = e["coefficient"], e["unit"], e["value"]
        match = re.fullmatch(r"L([123])\.(bias|w[0-9]+)", key)
        if not match or type(unit) is not int or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"malformed edit: {e!r}")
        layer, field = match.groups()
        rows = fixed["layers"][int(layer)-1]["rows"]
        if not 0 <= unit < len(rows) or (unit,key) in seen:
            raise ValueError(f"out-of-range or duplicate coordinate: {e!r}")
        seen.add((unit,key))
        row = rows[unit]
        if field == "bias":
            slot, name = row, "bias"
        else:
            idx = int(field[1:])
            slot = next((t for t in row["terms"] if t["feature"] == idx), None)
            if slot is None or field != f"w{idx}":
                raise ValueError(f"coordinate is not stored in original model: {e!r}")
            name = "coef"
        before = slot[name]
        delta = value - before
        if not 0 < abs(delta) <= MAX_DELTA + 1e-10:
            raise ValueError(f"bad delta: {e!r} old={before}")
        slot[name] = value
        changes.append({**e, "before": before, "delta": delta})
    return fixed, changes


def state_summary(r, steps):
    lines = ["t,x,z,pitch,vx,vz,pitch_v,contact_RL,q_RL,dq_RL,previous_torque_RL"]
    for i in sorted(set(steps) | {len(r["trace"])-1}):
        if i >= len(r["trace"]):
            continue
        row = r["trace"][i]
        groups = [[row[k] for k in ("t", "x", "z", "pitch", "vx", "vz", "pitch_v")],
                  row["contact"], row["q"], row["dq"], row["u"]]
        lines.append(" | ".join(",".join(f"{v:.3f}" for v in g) if g else "-" for g in groups))
    return "\n".join(lines)


def prepare(spec, out):
    base, target = roll(spec,NOMINAL), roll(spec,TARGET)
    assert stats(base)["pass"] and not stats(target)["pass"]
    raw = []
    for j, layer in enumerate(spec["layers"],1):
        raw.append(f"Layer L{j}")
        for i,row in enumerate(layer["rows"]):
            terms = " ".join(f'w{x["feature"]}={x["coef"]:.8g}' for x in row["terms"])
            raw.append(f'{i}: bias={row["bias"]:.8g} {terms}')
    steps = [0,5,10,20,30,40,45,50,55,58,60,62,64,66,68,70,72,75,78,80,82,84,86,90,95,100,103,110,120,140,170,200,249]
    prompt = f"""You are the expert editing an existing genuinely trained robot controller, not building a replacement.
This is a one-shot intervention with no training, tools, code, external information or rollout search.
Return at most {MAX_EDITS} direct replacements of stored weights or biases. A useful repair must complete
the failed rollout with the new initial pitch, also generalize to unseen nearby initial conditions,
and preserve the original 5-second walking capability. Prefer phase/state-sensitive small edits to
global changes that can destroy nominal gait. No scene ID is given to the policy.
Your JSON item {{ "unit": 0, "coefficient": "L3.w7", "value": 0.012 }} would REPLACE
the stored weight from L2 hidden unit 7 to output row 0 with 0.012, not add a delta.
Allowed keys L1.wN, L2.wN, L3.wN or L1.bias/L2.bias/L3.bias. Layer and row numbering
as listed below, rows/features zero-based; no adding absent terms. Absolute delta <= {MAX_DELTA}
from stored value for each edited coefficient. Include a brief diagnosis that distinguishes observed
failure from conjecture. No formula, only final numbers in edits.

Walker2d MuJoCo, sagittal right/left legs; q and dq are in rad and rad/s, angles are MuJoCo
joint coordinates; root pitch positive is +Y rotation (vertical torso tips toward +X).
Controls are six signed normalized joint torques clipped to [-1,1]. Joint order: {JOINT_ORDER}.
All six joint axes are (0,-1,0); positive torque increases that joint coordinate.
Outputs direct to unchanged motors, no gait or stabilizer beneath. dt 0.02 s; target vx .50 m/s.
Hidden layers use tanh(sum), final layer linear with clip. Inputs have their raw values.
Input order: {tuple(enumerate(FEATURES))}. q0..q5 and dq0..dq5 follow joint order.
contact0 is right floor contact, contact1 left. clock_sin,clock_cos are 1 Hz.
This network was distilled from successful control demonstrations at nominal start
and nearby state samples, not constructed for this failure. Before: nominal and target have the
same dynamics and joint perturbation; only root pitch starts at .02 vs .08 rad. Near-boundary
initial contact can differ naturally. Success requires 250 ticks with no fall, >=1m displacement,
and >=4 alternating advancing steps. Fall if |root pitch|>1 rad, height<.8 m or torso hits floor.

Nominal result: {json.dumps(stats(base))}; qualified touchdowns: {json.dumps(base["walk"]["touchdowns"])}.
{state_summary(base,steps)}

Target result: {json.dumps(stats(target))}; qualified touchdowns: {json.dumps(target["walk"]["touchdowns"])}.
{state_summary(target,steps)}

Stored network coefficients:
""" + "\n".join(raw) + "\n"
    (out/"prompt.txt").write_text(prompt)
    write(out/"target-before.json", target)


def evaluate(spec, proposal, out):
    fixed, changes = patch(spec, proposal)
    write(out/"patched.json", fixed)
    write(out/"weight-delta.json", changes)
    rows=[]
    for name,cases in (("nominal", [NOMINAL]), ("target", [TARGET]), ("heldout",CHALLENGE), ("retention",RETAIN)):
        for case in cases:
            before,after = roll(spec,case),roll(fixed,case)
            rows.append({"group":name,"seed":case[0],"pitch":case[1],"before":stats(before),"after":stats(after)})
            if name == "target":
                write(out/"target-traces.json", {"before":before,"after":after})
    totals={}
    for name in ("nominal","target","heldout","retention"):
        group=[x for x in rows if x["group"]==name]
        totals[name]={"n":len(group), **{k:sum(x[k]["pass"] for x in group) for k in ("before","after")},
                      "lost":sum(x["before"]["pass"] and not x["after"]["pass"] for x in group),
                      "gained":sum(not x["before"]["pass"] and x["after"]["pass"] for x in group)}
    count=sum(1+len(x["terms"]) for l in spec["layers"] for x in l["rows"])
    write(out/"results.json", {"weight_count":count,"edited":len(changes),"totals":totals,"cases":rows})
    print(json.dumps(totals))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=["prepare","evaluate"])
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--proposal",type=Path)
    args=ap.parse_args()
    spec=read(args.checkpoint)
    if args.command=="prepare":
        prepare(spec,args.out)
    else:
        evaluate(spec,read(args.proposal)["answer"],args.out)


if __name__ == "__main__":
    main()
