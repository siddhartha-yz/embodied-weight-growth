"""Set pulse difficulty with calibration only; seal fixed train/test scenarios."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from actor_source import load
from edits import Controller
from push_env import make_scenarios, rollout, reusable_native_env, single_support_scenarios


def evaluate(ctrl, scenario):
    ctrl.reset()
    return rollout(ctrl, scenario, trace_every=0)["metrics"]


def align(ctrl, scenarios):
    pattern = ("left_only", "right_only", "right_only", "left_only")
    sides = [pattern[i % len(pattern)] for i in range(len(scenarios))]
    return single_support_scenarios(ctrl, scenarios, even_steps=True, desired_supports=sides)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actor", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True)
    if (args.out / "plan.json").exists():
        raise ValueError("Calibration plan already sealed")
    actor = load(args.actor)
    ctrl = Controller(actor)
    cfg = dict(n_calibration=6, n_train=4, n_holdout=12, n_nominal=6)
    calibration = make_scenarios(1.0, **cfg)
    nominal_path = args.out / "nominal.json"
    if nominal_path.is_file():
        nominal = json.loads(nominal_path.read_text())
    else:
        nominal = [evaluate(ctrl, s) for s in calibration["nominal_train"][:3]]
        nominal_path.write_text(json.dumps(nominal, indent=2) + "\n")
    if not all(x["completed_horizon"] and not x["physically_down"] and x["mean_vx_mps"] > 0.65 for x in nominal):
        raise RuntimeError("Checkpoint did not reliably walk in calibration clean trials")
    aligned = align(ctrl, calibration["calibration"])
    history = []
    for magnitude in (58, 72, 90, 110, 52, 12, 6):
        scenarios = [replace(s, lateral_impulse_ns=s.lateral_impulse_ns * magnitude)
                     for s in aligned]
        metrics = [evaluate(ctrl, s) for s in scenarios]
        row = {"impulse_ns": magnitude, "fell": sum(m["physically_down"] for m in metrics),
               "recovered": sum(m["recovered_and_finished"] for m in metrics),
               "trials": len(metrics), "metrics": metrics}
        history.append(row)
        (args.out / "calibration.json").write_text(json.dumps(history, indent=2) + "\n")
        print(magnitude, row["fell"], row["recovered"], flush=True)
        if row["fell"] == len(metrics) // 2:
            break
    mixed = [x for x in history if 0 < x["fell"] < x["trials"]]
    if not mixed:
        raise RuntimeError("No pulse strength in calibrated grid yielded both survival and falls")
    best = min(mixed, key=lambda x: abs(x["fell"] / x["trials"] - .5))
    magnitude = best["impulse_ns"]
    plan = make_scenarios(magnitude, **cfg)
    plan["nominal_train"] = plan["nominal_train"][:2]
    for key in ("train", "holdout"):
        plan[key] = align(ctrl, plan[key])
    plan["calibration"] = [replace(s, lateral_impulse_ns=s.lateral_impulse_ns * magnitude)
                           for s in aligned]
    sealed = {k: [asdict(s) for s in items] for k, items in plan.items()}
    (args.out / "plan.json").write_text(json.dumps(sealed, indent=2) + "\n")
    print("Chosen N*s", magnitude, "calibration falls", best["fell"], "of", best["trials"], flush=True)


if __name__ == "__main__":
    with reusable_native_env():
        main()
