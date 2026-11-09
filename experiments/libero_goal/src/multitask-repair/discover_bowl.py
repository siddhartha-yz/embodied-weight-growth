"""Search a fixed small set of in-distribution-scale bowl reset offsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layouts import BowlOffsetEnv
from screen import run_episode, save
from smol_actor import SmolActor

# Fixed before simulation; new rows and new offsets are not inspected.
GRID = (
    (+.060, +.040),
    (+.000, +.070),
    (-.060, +.060),
    (-.040, -.065),
)
DEVELOPMENT_ROWS = (2, 3)
SAFE_REJECTIONS = (
    "Unsafe crossed-layout contact",
    "Both crossed objects must be supported",
    "Crossed layout did not settle",
    "Crossed layout cannot start",
)


def case(init, xy):
    dx, dy = xy
    return {"goal": "bowl", "init_index": init, "seed": 1000 + init,
            "init_source": "bowl", "layout": f"bowl_xy:{dx:+.3f},{dy:+.3f}"}


def trial_path(out, spec):
    layout = spec["layout"].split(":", 1)[1]
    label = layout.replace("+", "p").replace("-", "m").replace(".", "").replace(",", "-")
    return out / f"offset-{label}"


def classify(trace, row):
    first_lift = row["first_lift_step"]
    if row["success"]:
        return {"classification": "native_success"}
    if first_lift is None or row["first_lift"] != "bowl":
        return {"classification": "no_correct_sustained_lift"}
    held = [q["step"] for q in trace if q["step"] >= first_lift and "bowl" in q["status"]["grasped_objects"]]
    assert held, first_lift
    best = None
    rim = None
    last_seen_held = first_lift
    for q in trace[first_lift-1:]:
        step = q["step"]
        qheld = "bowl" in q["status"]["grasped_objects"]
        if qheld:
            last_seen_held = step
        if step-last_seen_held > 8:
            continue
        geom = q["status"]["cabinet_probe"]
        x,y,z = geom["target_in_site_m"]
        margin = geom["xy_margin_m"]
        # Track approach independently of strict qualification. Both criteria
        # refer to the native top site, not camera coordinates.
        distance = max(0., -margin) + abs(z)
        snap = {"step": step, "time_s": step/20, "held": qheld,
                "site_xyz_m": [x,y,z], "xy_margin_m": margin,
                "cabinet_contact": geom["contact"],
                "cabinet_contact_site_m": geom["contact_site_points_m"],
                "gripper_command": q["action"][6]}
        if best is None or distance < best[0]:
            best = distance, snap
        near_rim_contact = any(abs(c[2]) <= .035 for c in geom["contact_site_points_m"])
        if (geom["contact"] and near_rim_contact and -.035 <= margin <= .04
                and -.025 <= z <= .13):
            if rim is None or snap["held"] and not rim["held"]:
                rim = snap
    result = {"last_bowl_two_pad_step": held[-1], "closest_transport_state": best[1]}
    if rim is not None:
        result.update(classification="cabinet_rim_failure", rim_state=rim)
    elif best[0] <= .08:
        result.update(classification="near_cabinet_failure_without_rim_contact")
    else:
        result.update(classification="grasp_then_slip_or_missed_approach")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:5")
    p.add_argument("--render-gpu", type=int, default=5)
    p.add_argument("--continue-after-hit", action="store_true")
    args = p.parse_args()
    args.out.mkdir()
    plan = [case(i,xy) for xy in GRID for i in DEVELOPMENT_ROWS]
    save(args.out/"plan.json", {"cases": plan, "criterion": "Native bowl goal fails after a correct sustained lift; while held or within eight controls after last grip, bowl touches cabinet near top plane within 3.5 cm, root near top-site footprint boundary and height. No predicate or contact used by actor."})
    results=[]
    with torch.cuda.device(args.device):
        actor = SmolActor(device=args.device, n_action_steps=10)
        with BowlOffsetEnv(pair="objects", camera_size=256,
                           render_gpu_device_id=args.render_gpu, record_cabinet=True) as env:
            for spec in plan:
                path=trial_path(args.out,spec)
                path.mkdir(exist_ok=True)
                try:
                    row,_,_=run_episode(env,actor,output=path,max_steps=300,video=True,**spec)
                except RuntimeError as exc:
                    if not str(exc).startswith(SAFE_REJECTIONS):
                        raise
                    item={"case":spec,"classification":"unsafe_reset","reason":str(exc)}
                else:
                    trial=path/f"init-{spec['init_index']:03d}-bowl"
                    trace=json.loads((trial/"trace.json").read_text())
                    result=classify(trace,row)
                    item={"case":spec,"path":str(trial.relative_to(args.out)),"result":row,**result}
                    save(trial/"diagnosis.json",item)
                print(json.dumps(item,ensure_ascii=False),flush=True)
                results.append(item)
                save(args.out/"summary.json",{"cases":results})
                if item["classification"]=="cabinet_rim_failure":
                    if not (args.out/"first-qualified.json").exists():
                        save(args.out/"first-qualified.json",item)
                    if not args.continue_after_hit:
                        break

if __name__=="__main__":
    main()
