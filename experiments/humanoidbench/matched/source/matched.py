"""Re-evaluate frozen H1 repairs with one shared state gate and physical action cap."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics

import numpy as np

from actor_source import load
from compare import finite_json, window
from edits import residual_code, sparse_edit
from push_env import DT, Scenario, describe, reusable_native_env, rollout

STRIDE = 2
ACTOR_DT = STRIDE * DT
WINDOW_TICKS = 25
FADE_S = .3
OLD_CODE_NORMALIZED_LIMIT = .24
CODE_GAIN = -.12


def write(path, data):
    path.write_text(json.dumps(finite_json(data), ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def wrap_degrees(x):
    return (x + 180) % 360 - 180


class SameGate:
    """Same original code trigger, one use per native episode; no force or clock hint."""
    def __init__(self):
        self.ema = None
        self.event = None

    def visit(self, obs, step):
        if self.event is not None:
            return
        vy = float(obs[27])
        if self.ema is None:
            self.ema = vy
        if step // STRIDE > 30 and abs(vy - self.ema) > .9 and abs(float(obs[29])) > 2.5:
            self.event = {"step": step, "time_s": step * DT,
                          "ref_vy_mps": self.ema, "obs": obs.copy().tolist()}
        self.ema += .12 * (vy - self.ema)

    def fade(self, step):
        if self.event is None:
            return 0.
        age = (step - self.event["step"]) // STRIDE
        return min(1., (WINDOW_TICKS - age) * ACTOR_DT / FADE_S) if age < WINDOW_TICKS else 0.


def code_core(obs, reference_vy, hips):
    """Original code's signed gain; its old clip and timer move into shared host."""
    delta = np.zeros(19, dtype=np.float64)
    delta[list(hips)] = CODE_GAIN * (float(obs[27]) - reference_vy)
    return delta


class GatedPolicy:
    def __init__(self, actor, arm, scales, hips, cap, *, weights=None):
        self.actor, self.arm, self.scales, self.hips, self.cap = actor, arm, scales, hips, cap
        self.weights = weights
        self.gate = SameGate()
        self.step = 0
        self.held = None
        self.activity = []

    def __call__(self, obs):
        step = self.step
        if step % STRIDE == 0:
            raw, _ = self.actor.forward(obs)
            base = np.asarray(raw, dtype=np.float64)
            self.gate.visit(obs, step)
            fade = self.gate.fade(step)
            self.held = base
            if fade and self.arm != "base":
                if self.arm == "wsp":
                    a, _ = self.actor.forward(obs, self.weights)
                    raw_delta = np.asarray(a, dtype=np.float64) - base
                else:
                    raw_delta = code_core(obs, self.gate.event["ref_vy_mps"], self.hips)
                uncapped = float(np.linalg.norm(raw_delta * self.scales))
                gain = min(1., self.cap / uncapped) if uncapped else 1.
                self.held = np.clip(base + fade * gain * raw_delta, -1., 1.)
                actual_rad = (self.held - base) * self.scales
                norm = float(np.linalg.norm(actual_rad))
                if norm > self.cap * fade + 1e-12:
                    raise AssertionError("Emitted target correction exceeded the shared envelope")
                self.activity.append({"step": step, "fade": fade, "raw_l2_rad": uncapped,
                                      "actual_l2_rad": norm, "clipped": uncapped > self.cap,
                                      "actual_peak_joint_rad": float(np.max(np.abs(actual_rad)))})
        self.step += 1
        return self.held


def activity_stats(log):
    if not log:
        return {"ticks": 0, "peak_l2_rad": 0., "rms_l2_rad": 0., "integral_l2_rad_s": 0., "cap_hits": 0}
    norms = [a["actual_l2_rad"] for a in log]
    return {"ticks": len(log), "peak_l2_rad": max(norms),
            "rms_l2_rad": math.sqrt(statistics.fmean(x*x for x in norms)),
            "integral_l2_rad_s": sum(norms) * ACTOR_DT,
            "cap_hits": sum(a["clipped"] for a in log)}


def recovery(result, scenario, ref):
    m = result["metrics"]
    actual = window(result["trace"], scenario)
    data = {"met_tolerances": False}
    if actual is not None:
        dvx, dvy, yaw = actual - ref
        dyaw = wrap_degrees(math.degrees(yaw))
        data.update(vx_error_mps=float(dvx), vy_error_mps=float(dvy), yaw_error_deg=dyaw,
                    nominal_vx_mps=float(ref[0]),
                    met_tolerances=bool(abs(dvx) <= .2 and abs(dvy) <= .15 and abs(dyaw) <= 10.))
    m["matched_recovery"] = data
    m["recovered_and_finished"] = bool(m["completed_horizon"] and not m["physically_down"]
                                        and data["met_tolerances"])


def check_prefix(base, other, reference_event, other_event):
    if reference_event != other_event:
        raise AssertionError("Gate tick, state or lateral reference did not match")
    n = reference_event["step"] + 1 if reference_event else len(base["trace"])
    if len(base["trace"]) < n or len(other["trace"]) < n or base["trace"][:n] != other["trace"][:n]:
        raise AssertionError("Pre-trigger observations or actions did not match")
    if not reference_event and base["trace"] != other["trace"]:
        raise AssertionError("No-trigger trajectory changed")


def verify_code_core(path, hips, scales, cap):
    original = residual_code(path.read_text())
    for value in (-4., -2., -.91, 0., .91, 2., 4.):
        obs = np.zeros(51)
        obs[27] = value
        v = code_core(obs, 0., hips)
        norm = float(np.linalg.norm(v * scales))
        matched = v * (min(1., cap / norm) if norm else 1.)
        old = original(obs, np.zeros(19), np.zeros(512),
                       {"refvy": 0., "targetvy": 0., "timer": 1., "n": 31})
        np.testing.assert_allclose(matched, old, atol=1e-12, rtol=0)


def core_results(r):
    return {k: r[k] for k in ("metrics", "audit", "activity", "activity_stats", "gate") if k in r}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    out = args.run
    out.mkdir()
    plan = json.loads((args.prior / "calibrated/plan.json").read_text())
    previous = json.loads((args.prior / "run/heldout.json").read_text())
    actor = load(args.prior / "drq_checkpoint/actor_bundle.npz")
    weights = sparse_edit((args.prior / "run/wsp/selected.json").read_text(), actor.params)
    channels = describe()["channels"]
    hips = tuple(i for i, x in enumerate(channels) if x["joint"] in ("left_hip_roll", "right_hip_roll"))
    if hips != (1, 6):
        raise RuntimeError("Expected original H1 hip-roll channels")
    scales = np.asarray([(c["max_rad"] - c["min_rad"])/2 for c in channels])
    # Equal physical target-vector L2 cap, chosen to reproduce the original
    # two hip-roll commands' maximum (before fading and native saturation).
    cap = float(np.linalg.norm(OLD_CODE_NORMALIZED_LIMIT * scales[list(hips)]))
    verify_code_core(args.prior / "run/code/selected.py", hips, scales, cap)
    config = {"shared_l2_cap_rad": cap, "shared_l2_cap_degrees": math.degrees(cap),
              "actor_dt_s": ACTOR_DT, "native_dt_s": DT, "window_s": WINDOW_TICKS * ACTOR_DT,
              "fade_s": FADE_S, "detector": {"lateral_delta_mps": .9, "abs_roll_rate_radps": 2.5,
                                               "warmup_actor_ticks": 30, "ema_alpha": .12}}
    record = {"config": config, "push": [], "nominal": []}
    write(out / "summary.json", record)

    def trial(arm, scenario):
        ctrl = GatedPolicy(actor, arm, scales, hips, cap, weights=weights)
        r = rollout(ctrl, scenario, trace_every=1)
        r.update(gate=ctrl.gate.event, activity=ctrl.activity, activity_stats=activity_stats(ctrl.activity))
        return r

    for i, item in enumerate(plan["holdout"], 1):
        scenario = Scenario(**item)
        clean = trial("base", replace(scenario, push_step=None, lateral_impulse_ns=0.0))
        ref = window(clean["trace"], scenario)
        if ref is None:
            raise RuntimeError(f"Case {i} has no clean comparison window")
        arms = {"clean": clean, **{name: trial(name, scenario) for name in ("base", "wsp", "code")}}
        base = arms["base"]
        old = previous["base"]["push"][i-1]
        assert old["scenario"] == item
        for row in old["trace"]:
            assert base["trace"][row["step"]] == row, (i, row["step"])
        for name in ("base", "wsp", "code"):
            r = arms[name]
            check_prefix(base, r, base["gate"], r["gate"])
            recovery(r, scenario, ref)
        gate = base["gate"]
        item_out = {"case": i, "scenario": item, "trigger_step": gate["step"] if gate else None,
                    "prefix_identical": True, "reference": ref.tolist(),
                    "arms": {name: core_results(r) for name, r in arms.items()}}
        record["push"].append(item_out)
        write(out / "summary.json", record)
        if i == 11:
            write(out / "case11.json", {**item_out, "arms": arms})
        print("push", i, "trigger", item_out["trigger_step"], {
              k: (v["metrics"]["recovered_and_finished"], v["metrics"]["physically_down"])
              for k, v in arms.items() if k != "clean"}, flush=True)

    for i, item in enumerate(plan["nominal_holdout"], 1):
        scenario = Scenario(**item)
        arms = {name: trial(name, scenario) for name in ("base", "wsp", "code")}
        base = arms["base"]
        for r in arms.values():
            check_prefix(base, r, base["gate"], r["gate"])
        record["nominal"].append({"case": i, "scenario": item,
            "trigger_step": base["gate"]["step"] if base["gate"] else None, "prefix_identical": True,
            "arms": {name: core_results(r) for name, r in arms.items()}})
        write(out / "summary.json", record)


if __name__ == "__main__":
    with reusable_native_env():
        main()
