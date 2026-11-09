"""Compare temporary hidden-activation interventions in the frozen Walker2d policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

from walk_plant import DT, TARGET_V, WalkPlant
from walk_policy import FEATURES, JOINT_ORDER, compile_weights, inputs, output
from walk_server import Session, Steps


CASES = {"nominal": (271, 0.02), "target": (271, 0.08)}
TICKS = 250
MIN_TICK, MAX_TICK = 36, 79  # .72 to 1.58 s on the existing 50 Hz controller
MAX_WINDOW = 16
MAX_PROBES = 6
PROBE_KEYS = {"label", "layer", "unit", "start_s", "end_s", "delta"}
TORSO = ("x", "z", "pitch", "vx", "vz", "pitch_v")


@dataclass(frozen=True)
class Probe:
    label: str
    layer: int
    unit: int
    start: int
    end: int
    delta: float

    def definition(self):
        return {"label": self.label, "layer": self.layer, "unit": self.unit,
                "start_s": round(self.start * DT, 2), "end_s": round(self.end * DT, 2),
                "delta": self.delta}


def grid_tick(x):
    if type(x) not in (float, int) or not MIN_TICK * DT - 1e-9 <= x <= MAX_TICK * DT + 1e-9:
        raise ValueError("probe endpoints must be between 0.72 and 1.58 s")
    n = round(x / DT)
    if abs(x - n * DT) > 1e-9:
        raise ValueError("probe endpoints must be on the 0.02 s grid")
    return n


def parse_probes(audit):
    try:
        proposals = audit["answer"]["probes"]
    except (TypeError, KeyError) as exc:
        raise ValueError("expected SDK audit with answer.probes") from exc
    if not isinstance(proposals, list) or not 1 <= len(proposals) <= MAX_PROBES:
        raise ValueError(f"answer.probes must contain 1–{MAX_PROBES} interventions")
    probes, labels, settings = [], set(), set()
    for i, p in enumerate(proposals, 1):
        if not isinstance(p, dict) or set(p) != PROBE_KEYS:
            raise ValueError(f"probe {i}: expected exactly {', '.join(sorted(PROBE_KEYS))}")
        label, layer, unit, delta = p["label"], p["layer"], p["unit"], p["delta"]
        if not isinstance(label, str) or not label.strip() or label.strip() in labels:
            raise ValueError(f"probe {i}: label must be nonempty and unique")
        if type(layer) is not int or layer not in (1, 2) or type(unit) is not int or not 0 <= unit < 24:
            raise ValueError(f"probe {i}: layer must be 1 or 2 and unit must be 0–23")
        if type(delta) not in (int, float) or not 0.03 <= abs(delta) <= 0.8:
            raise ValueError(f"probe {i}: absolute delta must be 0.03–0.8")
        try:
            start, end = grid_tick(p["start_s"]), grid_tick(p["end_s"])
        except ValueError as exc:
            raise ValueError(f"probe {i}: {exc}") from exc
        if not 1 <= end - start <= MAX_WINDOW:
            raise ValueError(f"probe {i}: duration must be positive and at most 0.32 s")
        key = (layer, unit, start, end, delta)
        if key in settings:
            raise ValueError(f"probe {i}: duplicate intervention")
        labels.add(label.strip())
        settings.add(key)
        probes.append(Probe(label.strip(), layer, unit, start, end, delta))
    return probes


class ProbeNetwork:
    def __init__(self, spec):
        self.baseline = compile_weights(spec)
        if [len(layer["rows"]) for layer in spec["layers"]] != [24, 24, 6]:
            raise ValueError("expected the frozen 25→24→24→6 Walker2d checkpoint")
        # Same term order, fsum, tanh, inputs and clipped output as compile_weights.
        self.layers = tuple(tuple((tuple((t["feature"], t["coef"]) for t in row["terms"]), row["bias"])
                                  for row in layer["rows"]) for layer in spec["layers"])

    def intervene(self, state, target_v, probe):
        obs = inputs(state, target_v)
        z = [obs[k] for k in FEATURES]
        before = after = None
        for layer, rows in enumerate(self.layers, 1):
            z = [math.fsum(z[j] * coef for j, coef in terms) + bias for terms, bias in rows]
            if not all(math.isfinite(v) for v in z):
                raise ValueError("non-finite layer output")
            if layer < len(self.layers):
                z = [math.tanh(v) for v in z]
                if layer == probe.layer:
                    before = z[probe.unit]
                    after = max(-1.0, min(1.0, before + probe.delta))
                    z[probe.unit] = after
        return output(z), before, after

    def windowed(self, probe):
        def policy(state, target_v):
            tick = round(state["t"] / DT)
            if probe.start <= tick < probe.end:
                return self.intervene(state, target_v, probe)[0]
            return self.baseline(state, target_v)
        return policy


def rollout(spec, case, policy=None):
    seed, pitch = case
    plant = WalkPlant(seed, "train")
    plant.start_q[2] = pitch
    session = Session("weights", seed, "train", frozen_spec=spec, plant=plant)
    if policy is not None:
        session.policy = policy  # Local diagnostic; neither checkpoint nor simulator is changed.
    result = session.request({"op": "run", "ticks": TICKS})
    if not result["ok"]:
        raise RuntimeError(result["error"])
    result["scene"]["initial_pitch_rad"] = pitch
    return result


def rounded(value):
    return round(value, 6)


def rms(values):
    return math.sqrt(math.fsum(x * x for x in values) / len(values))


def fixed_state(network, probe, trace):
    rows = []
    # The terminal sample did not feed an action, even if its timestamp lies in a window.
    for state in trace[:-1]:
        if probe.start <= state["tick"] < probe.end:
            base = network.baseline(state, TARGET_V)
            changed, h_before, h_after = network.intervene(state, TARGET_V, probe)
            rows.append({"tick": state["tick"], "time_s": state["t"],
                         "activation_before": h_before, "activation_after": h_after,
                         "torque_before": base, "torque_after": changed,
                         "du": [b - a for a, b in zip(base, changed)]})
    if not rows:
        return {"n": 0}, rows
    peak = max(rows, key=lambda r: max(abs(x) for x in r["du"]))
    channels = list(zip(*(r["du"] for r in rows)))
    dh = [r["activation_after"] - r["activation_before"] for r in rows]
    activation = [r["activation_before"] for r in rows]
    summary = {"n": len(rows),
               "base_activation_range": [rounded(min(activation)), rounded(max(activation))],
               "applied_activation_delta_mean": rounded(math.fsum(dh) / len(rows)),
               "torque_delta_mean": [rounded(math.fsum(c) / len(c)) for c in channels],
               "torque_delta_rms": [rounded(rms(c)) for c in channels],
               "peak_torque_delta": {"time_s": rounded(peak["time_s"]),
                                     "du": [rounded(x) for x in peak["du"]]}}
    return summary, rows


def behavior(result, probe):
    trace, walk = result["trace"], result["walk"]
    next_edge = None
    for before, after in zip(trace, trace[1:]):
        # A state sample at start is prior to the first intervened action.
        if after["tick"] > probe.start and before["contact"][0] == 0 and after["contact"][0] == 1:
            next_edge = {"time_s": rounded(after["t"]), "right_foot_x_m": rounded(Steps.foot_x(after, 0)),
                         "torso": {key: rounded(after[key]) for key in TORSO},
                         "left_contact": bool(after["contact"][1]),
                         "qualified": any(event["foot"] == "R" and event["time_s"] == after["t"]
                                          for event in walk["touchdowns"])}
            break
    qualified = [{"foot": e["foot"], **{k: rounded(v) for k, v in e.items() if k != "foot"}}
                 for e in walk["touchdowns"] if round(e["time_s"] / DT) > probe.start]
    return {"next_raw_right_contact": next_edge,
            "next_qualified_right_touchdown": next((e for e in qualified if e["foot"] == "R"), None),
            "qualified_touchdowns_after_start": [[e["foot"], e["time_s"]] for e in qualified],
            "pass": result["success"] and len(trace) == TICKS + 1,
            "fall_reason": walk["fall_reason"], "elapsed_s": rounded(walk["elapsed_s"]),
            "distance_m": rounded(walk["distance_m"]), "alternating_steps": walk["alternating_steps"]}


def run(spec, probes, out):
    network = ProbeNetwork(spec)
    originals = {name: rollout(spec, case) for name, case in CASES.items()}
    summaries = []
    raw = {"baseline": originals, "probes": []}
    for probe in probes:
        policy = network.windowed(probe)
        summary = {"probe": probe.definition(), "cases": {}}
        raw_probe = {"probe": probe.definition(), "cases": {}}
        for name, case in CASES.items():
            original = originals[name]
            fixed, fixed_rows = fixed_state(network, probe, original["trace"])
            changed = rollout(spec, case, policy)
            if changed["trace"][:probe.start + 1] != original["trace"][:probe.start + 1]:
                raise RuntimeError(f"{probe.label}/{name} changed the trajectory before intervention")
            summary["cases"][name] = {"fixed_state": fixed,
                                      "before": behavior(original, probe), "after": behavior(changed, probe)}
            raw_probe["cases"][name] = {"fixed_state": fixed_rows, "intervention": changed}
        summaries.append(summary)
        raw["probes"].append(raw_probe)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe-traces.json").write_text(json.dumps(raw, allow_nan=False, separators=(",", ":")) + "\n")
    compact = {"definition": {"kind": "temporary post-tanh activation delta clipped to [-1,1]",
                              "window": "start-inclusive, end-exclusive; decisions every 0.02 s",
                              "fixed_state": "normalized torques on baseline input frames inside the window",
                              "next_contact": "right foot 0→1 after the first possible affected action, even if unqualified",
                              "torso_units": {"x": "m", "z": "m", "pitch": "rad", "vx": "m/s",
                                              "vz": "m/s", "pitch_v": "rad/s"}},
               "joint_order": JOINT_ORDER, "cases": CASES, "probes": summaries}
    (out / "probe-results.json").write_text(json.dumps(compact, allow_nan=False, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.checkpoint.read_text())
    audit = json.loads(args.proposal.read_text())
    try:
        probes = parse_probes(audit)
    except ValueError as exc:
        parser.error(str(exc))
    run(spec, probes, args.out)


if __name__ == "__main__":
    main()
