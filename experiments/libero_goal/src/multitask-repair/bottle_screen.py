"""Screen only preregistered development bottle offsets with the stock policy."""
from __future__ import annotations

import argparse
import json
from itertools import groupby
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layouts import BowlOffsetEnv
from screen import run_episode, save
from smol_actor import SmolActor


CASES = (
    ("dev02-right-back", 2, "bottle_xy:+0.040,-0.050"),
    ("dev02-left-front", 2, "bottle_xy:-0.040,+0.050"),
    ("dev03-right-front", 3, "bottle_xy:+0.030,+0.065"),
)
MICRO_CASES = (
    ("dev02-right-back-micro", 2, "bottle_xy:+0.020,-0.025"),
    ("dev02-left-front-micro", 2, "bottle_xy:-0.020,+0.025"),
    ("dev03-right-front-micro", 3, "bottle_xy:+0.015,+0.033"),
)


def describe(trace, canonical):
    def bottle_xyz(row):
        return np.asarray(row["status"]["object_xyz_m"]["bottle"])

    def has_grasp(row):
        return "bottle" in row["status"]["grasped_objects"]

    spans = []
    for held, rows in groupby(trace, key=has_grasp):
        rows = list(rows)
        if held:
            spans.append([rows[0]["step"], rows[-1]["step"]])

    peak = max(trace, key=lambda row: bottle_xyz(row)[2])
    after_grasp = trace[spans[0][0] - 1 :] if spans else trace
    goal_xyz = bottle_xyz(canonical[-1])
    closest = min(after_grasp, key=lambda row: np.linalg.norm(bottle_xyz(row)[:2] - goal_xyz[:2]))
    observations = {
        "bottle_grasp_spans": spans,
        "peak_bottle_step": peak["step"],
        "peak_bottle_xyz_m": bottle_xyz(peak),
        "final_bottle_xyz_m": bottle_xyz(trace[-1]),
        "canonical_success_bottle_xyz_m": goal_xyz,
        "closest_bottle_xy_to_canonical_final_m": float(
            np.linalg.norm(bottle_xyz(closest)[:2] - goal_xyz[:2])
        ),
        "closest_bottle_xy_step": closest["step"],
    }
    cabinet = [(r["step"], r["status"]["cabinet_probe"]) for r in trace
               if "cabinet_probe" in r["status"]]
    if cabinet:
        contacts = [t for t, probe in cabinet if probe["contact"]]
        margin_step, margin = max(cabinet, key=lambda item: item[1]["xy_margin_m"])
        observations.update({
            "first_cabinet_contact_step": contacts[0] if contacts else None,
            "cabinet_contact_control_count": len(contacts),
            "closest_native_xy_margin_m": margin["xy_margin_m"],
            "closest_native_xy_margin_step": margin_step,
        })
    return observations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path("screen-stock-prompt"))
    p.add_argument("--render-gpu", type=int, default=1)
    p.add_argument("--grid", choices=("offset", "micro"), default="offset")
    args = p.parse_args()
    args.run.mkdir()
    actor = SmolActor(n_action_steps=10)
    summary = {"replan_steps": 10, "max_steps": 300, "goal": "bottle", "cases": []}
    with BowlOffsetEnv(pair="objects", camera_size=256, render_gpu_device_id=args.render_gpu,
                       record_cabinet=args.grid == "micro") as env:
        for label, init, layout in MICRO_CASES if args.grid == "micro" else CASES:
            case = args.run / label
            case.mkdir()
            ref = args.reference / "canonical" / f"init-{init:03d}-bottle"
            stock = json.loads((ref / "result.json").read_text())
            if not stock["success"] or stock["instruction"] != env.instructions["bottle"]:
                raise ValueError(f"Reference is not a native stock success: {ref}")
            row, _, _ = run_episode(env, actor, "bottle", init, stock["seed"], 300, case,
                                    video=True, init_source="bowl", layout=layout)
            trace = json.loads((case / f"init-{init:03d}-bottle" / "trace.json").read_text())
            canonical = json.loads((ref / "trace.json").read_text())
            row.update(label=label, init_source="bowl", **describe(trace, canonical))
            summary["cases"].append(row)
            save(args.run / "summary.json", summary)
            print(json.dumps(row, default=lambda x: x.tolist()), flush=True)


if __name__ == "__main__":
    main()
