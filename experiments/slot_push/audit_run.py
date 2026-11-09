"""Summarize recorded slot pushes; optionally replay transfer with first vs last module."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


ARMS = ("weights", "code")
PHASES = ("train", "transfer")
MOTION = ("drive", "use")
TRANSFER_OPS = ("use", "reset", "finish")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def events(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def yaw(state):
    c, s = state["block_angle"]
    return math.degrees(math.atan2(s, c))


def pose(state):
    return {"block_mm": [round(x * 1000, 2) for x in state["block"]],
            "yaw_deg": round(yaw(state), 2), "block_speed_mm_s": round(math.hypot(*state["block_v"]) * 1000, 2),
            "tip_mm": [round(x * 1000, 2) for x in state["tip"]]}


def budget(state):
    m = state["meters"]
    return {"moves": m["moves_used"] + m["moves_left"], "ticks": m["ticks_used"] + m["ticks_left"],
            "ticks_per_move": m["max_ticks_per_move"]}


def motion_evidence(after):
    c = after["contact_summary"]
    return {
        "tip_block_ticks": c["tip_block_ticks"], "block_wall_ticks": c["block_wall_ticks"],
        "tip_wall_ticks": c["tip_wall_ticks"], "displacement_mm": round(c["block_displacement_m"] * 1000, 2),
        "rotation_deg": round(c["block_rotation_deg"], 2), "stall": c["stall"],
    }


def installed(events_):
    return [{"version": row["result"]["policy_version"], "turn": row["turn"], "spec": row["request"]["spec"]}
            for row in events_ if row["request"]["op"] == "install" and row["result"]["ok"]]


def change(before, after):
    if type(after) is str:
        return {"source_text_changed": before != after}
    a, b = before["layers"], after["layers"]
    shapes_a = [(len(x["weight"][0]), len(x["bias"])) for x in a]
    shapes_b = [(len(x["weight"][0]), len(x["bias"])) for x in b]
    if shapes_a != shapes_b:
        return {"before_shapes": shapes_a, "after_shapes": shapes_b}
    diffs = [abs(p-q) for layer_a, layer_b in zip(a, b)
             for row_a, row_b in zip([*layer_a["weight"], layer_a["bias"]], [*layer_b["weight"], layer_b["bias"]])
             for p, q in zip(row_a, row_b)]
    return {"changed_coefficients": sum(d != 0 for d in diffs), "max_abs_change": max(diffs)}


def phase_summary(initial, rows, protocol, final=None):
    steps = []
    install_turns = []
    before = initial
    prior_module = None
    last_stall = None
    revisions = []
    for row in rows:
        after, request = row["result"], row["request"]
        turn, op = row["turn"], request["op"]
        step = {
            "turn": turn, "attempt": after["attempt"], "op": op, "ok": after["ok"],
            "note": row["note"], "policy_version": after["policy_version"], "success": after["success"],
        }
        if not after["ok"]:
            step["error"] = after["error"]
        if op in MOTION:
            step.update({"request": request, "before": pose(before["state"]), "after": pose(after["state"]),
                         "ticks_executed": after["meters"]["ticks_used"] - before["meters"]["ticks_used"]})
            step["contact"] = motion_evidence(after)
            if step["contact"]["stall"]:
                last_stall = turn
        elif op == "reset" and after["ok"]:
            last_stall = None
        elif op == "install" and after["ok"]:
            install_turns.append(turn)
            if prior_module is not None:
                delta = change(prior_module, request["spec"])
                revisions.append({"turn": turn, "version": after["policy_version"],
                                  "last_stall_this_attempt": last_stall, "artifact_change": delta})
                step["artifact_change"] = delta
            prior_module = request["spec"]
        steps.append(step)
        before = after
    successes = [s["turn"] for s in steps if s["success"]]
    stalls = [s["turn"] for s in steps if s.get("contact", {}).get("stall")]
    motions = [s for s in steps if s["op"] in MOTION]
    if final is not None and (len(rows) != final["decisions"] or not matching(final["state"], before)):
        raise ValueError(f"{protocol['arm']}/{protocol['phase']}: final snapshot differs from the last recorded turn")
    return {
        "context": protocol["context"], "source_policy_version": protocol["source_policy_version"],
        "seed": protocol["seed"], "model": protocol["model"], "budget": {
            "decisions": protocol["max_decisions"], **budget(initial)},
        "start": pose(initial["state"]), "end": pose(before["state"]),
        "complete": final is not None, "stop": final["stop"] if final else None,
        "decisions": len(steps), "moves": before["meters"]["moves_used"], "ticks": before["meters"]["ticks_used"],
        "first_success_turn": 0 if initial["success"] else successes[0] if successes else None,
        "final_success": before["success"], "resets": sum(s["op"] == "reset" and s["ok"] for s in steps),
        "contact_turns": [s["turn"] for s in motions if s["contact"]["tip_block_ticks"]],
        "rail_contact_turns": [s["turn"] for s in motions if s["contact"]["block_wall_ticks"]],
        "stall_turns": stalls, "accepted_install_turns": install_turns, "revisions": revisions, "steps": steps,
    }


def load_phase(path):
    init_file = path / "initial.json"
    if not init_file.exists() or not (path / "events.jsonl").exists():
        return None
    initial, rows, protocol = read(init_file), events(path), read(path / "protocol.json")
    final_file = path / "final.json"
    final = read(final_file) if final_file.exists() else None
    return initial, rows, protocol, final


def paired_design(phases):
    design = {}
    for phase in PHASES:
        if all(phase in phases[a] for a in ARMS):
            a, b = (phases[arm][phase] for arm in ARMS)
            design[phase] = {
                "same_start_state": a[0]["state"] == b[0]["state"], "same_geometry": a[0]["scene"] == b[0]["scene"],
                "same_budget": budget(a[0]) == budget(b[0]) and a[2]["max_decisions"] == b[2]["max_decisions"],
                "same_model_and_seed": (a[2]["model"], a[2]["seed"]) == (b[2]["model"], b[2]["seed"]),
            }
    for arm in ARMS:
        if all(p in phases[arm] for p in PHASES):
            train, transfer = (phases[arm][phase] for phase in PHASES)
            obs = lambda initial: {k: v for k, v in initial["state"].items() if k != "block_angle"}
            geom = lambda initial: {k: v for k, v in initial["scene"].items() if k != "name"}
            mods = installed(train[1])
            design[arm] = {
                "same_start_except_yaw": obs(train[0]) == obs(transfer[0]) and
                    train[0]["state"]["block_angle"] != transfer[0]["state"]["block_angle"],
                "same_geometry": geom(train[0]) == geom(transfer[0]),
                "same_seed": train[2]["seed"] == transfer[2]["seed"],
                "context_recorded_as_retained": transfer[2]["context"] == "retained",
                "no_fresh_install_or_raw_drive": transfer[0]["policy_frozen"] and
                    all(row["request"]["op"] in TRANSFER_OPS for row in transfer[1]),
                "source_version_is_last_train_install": bool(mods) and
                    mods[-1]["version"] == transfer[2]["source_policy_version"],
            }
    return design


def matching(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(matching(v, b[k]) for k, v in a.items())
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(matching(x, y) for x, y in zip(a, b))
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-9)
    return type(a) is type(b) and a == b


def simulator_command(args, arm, phase, seed, frozen):
    flags = ["--arm", arm, "--scene", phase, "--seed", str(seed)]
    if frozen:
        flags.append("--frozen-spec-stdin")
    if args.ssh_host:
        remote = [args.python, "-B", "-u", args.remote_server, *flags]
        if args.remote_pythonpath:
            remote = ["env", f"PYTHONPATH={args.remote_pythonpath}", *remote]
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "--", args.ssh_host, shlex.join(remote)]
    return [args.python, "-B", "-u", str(Path(__file__).with_name("slot_server.py").resolve()), *flags]


def replay(args, arm, phase, seed, spec, requests):
    messages = list(requests) if spec is None else [spec, *requests]
    payload = "".join(json.dumps(x, allow_nan=False) + "\n" for x in messages)
    p = subprocess.run(simulator_command(args, arm, phase, seed, spec is not None), input=payload,
                       text=True, encoding="utf-8", stdout=subprocess.PIPE, check=True, timeout=120)
    states = [json.loads(line) for line in p.stdout.splitlines()]
    if len(states) != len(requests) + 1:
        raise ValueError(f"{arm}/{phase}: simulator returned {len(states)} snapshots for {len(requests)} requests")
    return states


def verify_phase(args, arm, phase, data, train):
    if data is None or data[3] is None:
        return {"status": "skipped", "reason": "recorded phase incomplete"}
    initial, rows, protocol, _ = data
    artifact = None
    if phase == "transfer":
        mods = installed(train[1]) if train else []
        if not mods or mods[-1]["version"] != protocol["source_policy_version"]:
            return {"status": "invalid", "reason": "the last training artifact does not match the transfer version"}
        artifact = mods[-1]
    states = replay(args, arm, phase, protocol["seed"], artifact["spec"] if artifact else None,
                    [row["request"] for row in rows])
    for turn, got, expected in zip([0, *[r["turn"] for r in rows]], states, [initial, *[r["result"] for r in rows]]):
        if not matching(got, expected):
            return {"status": "invalid", "reason": f"snapshot differs at turn {turn}"}
    if phase == "train":
        mods = installed(rows)
        artifact = mods[-1] if mods else None
    if artifact:
        path = args.run / arm / phase / "motor_artifact.json"
        if not path.exists() or read(path) != artifact:
            return {"status": "invalid", "reason": "saved artifact differs from the recorded install"}
    return {"status": "matched", "snapshots": len(states), "ticks": states[-1]["meters"]["ticks_used"],
            "artifact": artifact["version"] if artifact else None}


def replay_outcome(states, rows):
    first_success = next((row["turn"] for state, row in zip(states[1:], rows) if state["success"]), None)
    return {"first_success_turn": 0 if states[0]["success"] else first_success,
            "final_success": states[-1]["success"], "end": pose(states[-1]["state"]),
            "moves": states[-1]["meters"]["moves_used"], "ticks": states[-1]["meters"]["ticks_used"]}


def ablation(args, arm, train, transfer, verified=None):
    if train is None or transfer is None or transfer[3] is None:
        return {"status": "skipped", "reason": "both recorded phases must be complete"}
    mods = installed(train[1])
    if len(mods) < 2:
        return {"status": "skipped", "reason": "no accepted revision to compare"}
    first, last = mods[0], mods[-1]
    if first["spec"] == last["spec"]:
        return {"status": "skipped", "reason": "first and final artifacts are identical"}
    initial, rows, protocol, _ = transfer
    if protocol["source_policy_version"] != last["version"]:
        return {"status": "invalid", "reason": "recorded transfer did not use the final training version"}
    requests = [row["request"] for row in rows]
    if not initial["policy_frozen"] or any(req["op"] not in TRANSFER_OPS for req in requests):
        return {"status": "invalid", "reason": "recorded transfer allowed a controller bypass or edit"}
    verified = verified or verify_phase(args, arm, "transfer", transfer, train)
    if verified["status"] != "matched":
        return {"status": "invalid", "reason": f"final-artifact replay: {verified['reason']}"}
    final_states = [initial, *[r["result"] for r in rows]]
    first_states = replay(args, arm, "transfer", protocol["seed"], first["spec"], requests)
    if not matching(first_states[0], final_states[0]):
        return {"status": "invalid", "reason": "the two replays did not start at the same state"}
    divergence = next((row["turn"] for row, a, b in zip(rows, first_states[1:], final_states[1:])
                       if not matching(a, b)), None)
    return {
        "status": "replayed",
        "interpretation": "same transfer requests, no model calls; does not test how Astra would replan with the first version",
        "first_version": first["version"], "final_version": last["version"],
        "final_replay_matched_recording": True, "first_divergent_turn": divergence,
        "first": replay_outcome(first_states, rows), "final": replay_outcome(final_states, rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, help="write JSON here rather than stdout")
    parser.add_argument("--verify-replay", action="store_true", help="recompute all completed phases from recorded commands")
    parser.add_argument("--ablate-transfer", action="store_true")
    parser.add_argument("--ssh-host")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--remote-server")
    parser.add_argument("--remote-pythonpath")
    args = parser.parse_args()
    if args.ssh_host and not args.remote_server:
        parser.error("--ssh-host requires --remote-server")
    phases = {arm: {p: loaded for p in PHASES if (loaded := load_phase(args.run / arm / p)) is not None}
              for arm in ARMS}
    if not any(phases.values()):
        parser.error("no recorded phases found under run/{weights,code}/{train,transfer}")
    report = {
        "transfer_scope": "Reuse with retained context tests the frozen module plus Astra's planning, not the module alone.",
        "design": paired_design(phases),
        "arms": {arm: {p: phase_summary(*data) for p, data in arm_phases.items()}
                 for arm, arm_phases in phases.items()},
    }
    if args.verify_replay:
        report["physical_replay"] = {
            arm: {p: verify_phase(args, arm, p, data, arm_phases.get("train")) for p, data in arm_phases.items()}
            for arm, arm_phases in phases.items()}
    if args.ablate_transfer:
        report["fixed_action_transfer_ablation"] = {
            arm: ablation(args, arm, arm_phases.get("train"), arm_phases.get("transfer"),
                          report.get("physical_replay", {}).get(arm, {}).get("transfer"))
            for arm, arm_phases in phases.items()}
    result = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
