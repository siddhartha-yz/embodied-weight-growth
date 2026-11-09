"""Evaluate code-to-weight candidates and an ankle-role ablation in the original walker."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys

from audit_walk import differing_path, outcome, read, replay, write
from walk_policy import JOINT_ORDER, compile_policy


SCENES = ("train", "transfer")
SWING_ANKLES = (
    "        an0=0.3*(p-obs['q0']-obs['q1']-obs['q2'])",
    "        an1=0.3*(p-obs['q3']-obs['q4']-obs['q5'])",
)


def shared_ankle_code(source):
    """Remove only the two swing-ankle overrides; keep phase, hips and knees."""
    lines = source.splitlines()
    if any(lines.count(line) != 1 for line in SWING_ANKLES):
        raise ValueError("expected the two exact swing-ankle assignments of walking code v8")
    return "\n".join(line for line in lines if line not in SWING_ANKLES)


def teacher_error(arm, spec, teacher):
    """One-step errors on observed teacher states (separate from actual rollouts)."""
    policy = compile_policy(arm, spec)
    pairs = list(zip(teacher["trace"], teacher["trace"][1:]))
    errors = [[] for _ in JOINT_ORDER]
    response = {role: {"own_ankle_angle": [], "forward_speed": []} for role in ("stance", "swing")}
    for before, after in pairs:
        actual = policy(before, 0.5)
        for i, (a, b) in enumerate(zip(actual, after["u"])):
            errors[i].append(abs(a - b))
        phase = math.sin(7 * before["t"])
        if abs(phase) < 0.1:
            continue
        for index, swinging in ((2, phase > 0), (5, phase < 0)):
            # Exclude clipping by the teacher; infer the feedback law at a fixed state and phase.
            if abs(after["u"][index]) > 0.94:
                continue
            role = "swing" if swinging else "stance"
            for feature, step in (("own_ankle_angle", 0.02), ("forward_speed", 0.04)):
                left, right = dict(before), dict(before)
                if feature == "own_ankle_angle":
                    left["q"], right["q"] = list(before["q"]), list(before["q"])
                    left["q"][index] -= step
                    right["q"][index] += step
                else:
                    left["vx"] -= step
                    right["vx"] += step
                response[role][feature].append((policy(right, 0.5)[index] - policy(left, 0.5)[index]) / (2 * step))
    return {
        "states": len(pairs),
        "mae": sum(sum(e) for e in errors) / (len(pairs) * len(errors)),
        "rmse": math.sqrt(sum(sum(x*x for x in e) for e in errors) / (len(pairs) * len(errors))),
        "max_error": max(map(max, errors)),
        "mae_by_joint": {name: sum(err) / len(err) for name, err in zip(JOINT_ORDER, errors)},
        "mean_ankle_feedback_slope": {role: {k: sum(xs) / len(xs) if xs else None
                                             for k, xs in data.items()} for role, data in response.items()},
    }


def recorded(run, arm, scene):
    return read(run / arm / "frozen" / scene / "v08.json")


def evaluate(args, arm, spec, name, originals):
    report = {}
    for scene in SCENES:
        ref = originals["code"][scene]
        request = ref["request"]
        start, result = replay(args, arm, scene, ref["seed"], [request], spec)
        if (path := differing_path(start, ref["initial"])) is not None:
            raise ValueError(f"{name}/{scene}: initial snapshot differs from code v8 at {path}")
        if not result["ok"]:
            raise ValueError(f"{name}/{scene}: {result['error']}")
        write(args.out / name / f"{scene}.json", {
            "arm": arm, "name": name, "scene": scene, "seed": ref["seed"], "request": request,
            "initial": start, "result": result,
        })
        report[scene] = {**outcome(result), "teacher_torque_error": teacher_error(arm, spec, ref["result"])}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weights", action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--ablate-code", action="store_true")
    parser.add_argument("--ssh-host")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--remote-server")
    parser.add_argument("--remote-pythonpath")
    args = parser.parse_args()
    if args.ssh_host and not args.remote_server:
        parser.error("--ssh-host requires --remote-server")
    originals = {a: {s: recorded(args.run, a, s) for s in SCENES} for a in ("code", "weights")}
    summary = {"reference": {}, "new": {}}
    for arm, scenes in originals.items():
        spec = read(args.run / arm / "train" / "policy-v08.json")["spec"]
        summary["reference"][arm] = {
            s: {**outcome(r["result"]), "teacher_torque_error": teacher_error(arm, spec, originals["code"][s]["result"])}
            for s, r in scenes.items()
        }
    if args.ablate_code:
        name = "code-shared-ankles"
        spec = shared_ankle_code(read(args.run / "code" / "train" / "policy-v08.json")["spec"])
        write(args.out / name / "artifact.json", {"spec": spec})
        summary["new"][name] = evaluate(args, "code", spec, name, originals)
    for item in args.weights:
        name, sep, path = item.partition("=")
        if not sep or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            parser.error("--weights expects NAME=FILE with an alphanumeric, dash or underscore name")
        payload = read(Path(path))
        spec = payload["spec"] if "spec" in payload else payload
        write(args.out / name / "artifact.json", {"spec": spec})
        summary["new"][name] = evaluate(args, "weights", spec, name, originals)
    write(args.out / "comparison.json", summary)
    for name, scenes in {**summary["reference"], **summary["new"]}.items():
        for scene, result in scenes.items():
            print(f"{name}/{scene}: {result['distance_m']:+.3f} m, {result['alternating_steps']} steps, "
                  f"{result['elapsed_s']:.2f} s, success={result['success']}, "
                  f"torque RMSE={result['teacher_torque_error']['rmse']:.5f}")


if __name__ == "__main__":
    main()
