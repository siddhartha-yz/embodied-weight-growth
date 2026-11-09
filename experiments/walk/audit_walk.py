"""Audit walking attempts; optionally verify replay and test frozen controller versions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


ARMS = ("weights", "code")
SCENES = ("train", "transfer")
MOTION = ("drive", "run")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_phase(path):
    log = path / "events.jsonl"
    if not log.is_file() or not (path / "initial.json").is_file():
        return None
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    final = path / "final.json"
    return read(path / "initial.json"), rows, read(path / "protocol.json"), read(final) if final.is_file() else None


def point(snapshot):
    state, walk = snapshot["state"], snapshot["walk"]
    return {
        "x_m": state["x"], "z_m": state["z"], "pitch_rad": state["pitch"],
        "vx_m_s": state["vx"], "contact": state["contact"],
        "elapsed_s": walk["elapsed_s"], "distance_m": walk["distance_m"],
        "peak_x_m": walk["peak_x_m"], "first_success_s": walk["first_success_s"],
        "alternating_steps": walk["alternating_steps"], "fall_reason": walk["fall_reason"],
        "success": snapshot["success"],
    }


def outcome(snapshot):
    return {**point(snapshot), "touchdowns": snapshot["walk"]["touchdowns"],
            "error": snapshot.get("error"),
            "ticks_used": snapshot["meters"]["ticks_used"]}


def budget(snapshot, protocol):
    meters = snapshot["meters"]
    return {"max_decisions": protocol.get("max_decisions"),
            "max_moves": meters["moves_used"] + meters["moves_left"],
            "max_ticks": meters["ticks_used"] + meters["ticks_left"],
            "max_ticks_per_move": meters["max_ticks_per_move"]}


def spec_changes(before, after):
    if isinstance(before, str) or isinstance(after, str):
        return {"source_text_changed": before != after}

    def coords(spec):
        values = {}
        shape = []
        for i, layer in enumerate(spec["layers"]):
            shape.append(len(layer["rows"]))
            for j, row in enumerate(layer["rows"]):
                values[i, j, "bias"] = row["bias"]
                for item in row["terms"]:
                    values[i, j, item["feature"]] = item["coef"]
        return shape, values

    a, p = coords(before)
    b, q = coords(after)
    if a != b:
        return {"before_layer_widths": a, "after_layer_widths": b}
    delta = [abs(p.get(k, 0) - q.get(k, 0)) for k in p.keys() | q.keys()]
    return {"changed_coefficients": sum(d != 0 for d in delta), "max_abs_change": max(delta, default=0)}


def installs(rows):
    return [{"version": row["result"]["policy_version"], "turn": row["turn"], "spec": row["request"]["spec"]}
            for row in rows if row["request"]["op"] == "install" and row["result"]["ok"]]


def motion_evidence(snapshot):
    contacts = {"none": 0, "right_only": 0, "left_only": 0, "both": 0}
    saturated = 0
    for sample in snapshot.get("trace", [])[1:]:
        right, left = sample["contact"]
        contacts[("none", "right_only", "left_only", "both")[int(right) + 2 * int(left)]] += 1
        saturated += any(abs(u) >= .999999 for u in sample["u"])
    return {"contact_ticks": contacts, "ticks_with_saturated_motor": saturated}


def summarize(initial, rows, protocol, final):
    prior, segments, changes, stretches = initial, [], [], []
    installed = None
    current = None
    first_success = 0 if initial["success"] else None
    for row in rows:
        turn, request, now = row["turn"], row["request"], row["result"]
        op = request["op"]
        a, b = point(prior), point(now)
        record = {"turn": turn, "attempt": now["attempt"], "op": op, "note": row["note"],
                  "ok": now["ok"], "policy_version": now["policy_version"], "after": b}
        if not now["ok"]:
            record["error"] = now["error"]
        if op in MOTION:
            record.update(before=a, requested_ticks=request["ticks"],
                          actual_ticks=now["meters"]["ticks_used"] - prior["meters"]["ticks_used"],
                          new_touchdowns=now["walk"]["touchdowns"][len(prior["walk"]["touchdowns"]):],
                          **motion_evidence(now))
            if op == "run" and record["actual_ticks"] > 0:
                key = (now["attempt"], now["policy_version"])
                if current is None or (current["attempt"], current["version"]) != key:
                    current = {"attempt": key[0], "version": key[1], "start_turn": turn,
                               "from_reset": a["elapsed_s"] == 0, "start": a}
                    stretches.append(current)
                c = current["start"]
                current.update(end_turn=turn, end=b,
                               gained_alternating_steps=b["alternating_steps"] - c["alternating_steps"],
                               gained_distance_m=b["x_m"] - c["x_m"],
                               controlled_s=b["elapsed_s"] - c["elapsed_s"])
            if op == "drive":
                current = None
        elif op == "install" and now["ok"]:
            if installed is not None:
                changes.append({"turn": turn, "version": now["policy_version"],
                                "previous_fall": a["fall_reason"],
                                **spec_changes(installed, request["spec"])})
            installed, current = request["spec"], None
        elif op == "reset" and now["ok"]:
            current = None
        if first_success is None and (now["success"] or b["first_success_s"] is not None):
            first_success = turn
        segments.append(record)
        prior = now

    if final is not None and (len(rows) != final["decisions"] or final["state"] != prior):
        raise ValueError("final.json disagrees with the last recorded decision")
    return {"completed": final is not None, "stop": final["stop"] if final is not None else None,
            "model": protocol.get("model"), "seed": protocol["seed"], "budget": budget(initial, protocol),
            "start": point(initial), "final": outcome(prior), "first_success_turn": first_success,
            "decisions": len(rows), "motion_segments": prior["meters"]["moves_used"],
            "falls": [s["turn"] for s in segments if s.get("before", {}).get("fall_reason") is None
                      and s["op"] in MOTION and s["after"]["fall_reason"] is not None],
            "installed_versions": [{"turn": a["turn"], "version": a["version"]} for a in installs(rows)],
            "revisions": changes, "controller_stretches": stretches, "turns": segments}


def differing_path(a, b, path="$"):
    """First divergent field, with narrow tolerance for MuJoCo floating-point output."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return path
        for k, v in a.items():
            if (p := differing_path(v, b[k], f"{path}.{k}")) is not None:
                return p
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return path
        for i, (x, y) in enumerate(zip(a, b)):
            if (p := differing_path(x, y, f"{path}[{i}]")) is not None:
                return p
        return None
    if type(a) in (float, int) and type(b) in (float, int):
        return None if math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-9) else path
    return None if type(a) is type(b) and a == b else path


def replay(args, arm, scene, seed, requests, spec=None):
    flags = ["--arm", arm, "--scene", scene, "--seed", str(seed)]
    if spec is not None:
        flags.append("--frozen-spec-stdin")
    if args.ssh_host:
        remote = [args.python, "-B", "-u", args.remote_server, *flags]
        if args.remote_pythonpath:
            remote = ["env", f"PYTHONPATH={args.remote_pythonpath}", *remote]
        cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "--",
               args.ssh_host, shlex.join(remote)]
    else:
        cmd = [args.python, "-B", "-u", str(Path(__file__).with_name("walk_server.py").resolve()), *flags]
    messages = requests if spec is None else [spec, *requests]
    payload = "".join(json.dumps(msg, allow_nan=False) + "\n" for msg in messages)
    p = subprocess.run(cmd, input=payload, text=True, encoding="utf-8",
                       stdout=subprocess.PIPE, check=True, timeout=120)
    states = [json.loads(line) for line in p.stdout.splitlines()]
    if len(states) != len(requests) + 1:
        raise ValueError(f"{arm}/{scene}: replay returned {len(states)} snapshots for {len(requests)} requests")
    return states


def verify_train(args, arm, data):
    initial, rows, protocol, final = data
    if final is None:
        return {"status": "skipped", "reason": "phase still in progress"}
    requests = [row["request"] for row in rows]
    states = replay(args, arm, "train", protocol["seed"], requests)
    for t, got, expected in zip([0, *[row["turn"] for row in rows]], states,
                                [initial, *[row["result"] for row in rows]]):
        if (path := differing_path(got, expected)) is not None:
            return {"status": "mismatch", "turn": t, "path": path}
    for artifact in installs(rows):
        path = args.run / arm / "train" / f"policy-v{artifact['version']:02}.json"
        if not path.is_file() or read(path) != artifact:
            return {"status": "mismatch", "artifact_file": path.name}
    return {"status": "matched", "snapshots": len(states), "ticks": states[-1]["meters"]["ticks_used"]}


def frozen_comparison(args, arm, data, summary):
    initial, rows, protocol, final = data
    if final is None:
        return {"status": "skipped", "reason": "phase still in progress"}
    modules = installs(rows)
    if not modules:
        return {"status": "skipped", "reason": "no accepted controller to evaluate"}
    v1 = modules[0]
    # Rank the improvement observed while the candidate was actually in control;
    # do not credit a version with footsteps accumulated before its installation.
    revised = {a["version"]: a for a in modules[1:] if a["spec"] != v1["spec"]}
    ranks = [s for s in summary["controller_stretches"] if s["version"] in revised]
    best = max(ranks, key=lambda s: (
        s["end"]["success"] and not s["start"]["success"],
        s["gained_alternating_steps"], s["end"]["fall_reason"] is None,
        s["gained_distance_m"], s["controlled_s"],
    ), default=None)
    selected = [v1]
    if best is not None:
        selected.append(revised[best["version"]])
    ticks = initial["meters"]["max_ticks_per_move"]
    trials = {}
    for scene in SCENES:
        starts = []
        trials[scene] = []
        for artifact in selected:
            request = {"op": "run", "ticks": ticks}
            start, end = replay(args, arm, scene, protocol["seed"], [request], artifact["spec"])
            if not start["policy_frozen"]:
                raise ValueError(f"{arm}/{scene}: invalid frozen trial for v{artifact['version']}")
            if starts and differing_path(start, starts[0]) is not None:
                raise ValueError(f"{arm}/{scene}: controller versions started in different states")
            starts.append(start)
            path = args.run / arm / "frozen" / scene / f"v{artifact['version']:02}.json"
            write(path, {"arm": arm, "scene": scene, "seed": protocol["seed"], "artifact": artifact,
                         "request": request, "initial": start, "result": end})
            trials[scene].append({"version": artifact["version"], "install_turn": artifact["turn"],
                                  "record": str(path.relative_to(args.run)), **outcome(end),
                                  **motion_evidence(end)})
    return {"status": "compared" if len(selected) == 2 else "one_version",
            "selection": "first accepted controller versus strongest distinct revised controller observed in training",
            "rank_order": ["success", "additional qualified alternating steps", "not fallen",
                           "distance gained", "duration under module"],
            "selected_training_stretch": best if len(selected) == 2 else None,
            "requested_ticks_per_trial": ticks, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-replay", action="store_true", help="replay completed online training runs")
    parser.add_argument("--ablate", action="store_true", help="test frozen versions at nominal and altered initial pitch")
    parser.add_argument("--ssh-host")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--remote-server")
    parser.add_argument("--remote-pythonpath")
    args = parser.parse_args()
    if args.ssh_host and not args.remote_server:
        parser.error("--ssh-host requires --remote-server")
    phases = {a: {s: d for s in SCENES if (d := load_phase(args.run / a / s)) is not None} for a in ARMS}
    if not any(phases.values()):
        parser.error("no recorded phases found under run/{weights,code}/{train,transfer}")
    report = {"interpretation": "A single paired pilot is not a population comparison. Online controller stretches can begin in different states; frozen trials use the same initial state and duration, with no Astra calls.",
              "arms": {a: {s: summarize(*d) for s, d in p.items()} for a, p in phases.items()}}
    train = {a: p["train"] for a, p in phases.items() if "train" in p}
    if len(train) == 2:
        a, b = (train[arm] for arm in ARMS)
        report["paired_train_design"] = {
            "same_initial_state": a[0]["state"] == b[0]["state"],
            "same_scene": a[0]["scene"] == b[0]["scene"],
            "same_budget": budget(a[0], a[2]) == budget(b[0], b[2]),
            "same_model_and_seed": (a[2].get("model"), a[2]["seed"]) == (b[2].get("model"), b[2]["seed"]),
        }
    if args.verify_replay:
        report["physical_replay"] = {a: verify_train(args, a, d) for a, d in train.items()}
    if args.ablate:
        report["frozen"] = {a: frozen_comparison(args, a, d, report["arms"][a]["train"]) for a, d in train.items()}
    if args.out:
        write(args.out, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if any(x["status"] == "mismatch" for x in report.get("physical_replay", {}).values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
