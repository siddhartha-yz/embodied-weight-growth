"""Paired real-time and zero-simulation-latency replay of the same Astra patch."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
import urllib.request

import numpy as np

from actor_source import load
from compare import finite_json, window
from edits import Controller, json_payload, sparse_edit
from live_rollout import TriggerConfig, live_rollout
from push_env import Scenario, describe, reusable_native_env

CASES = (1, 8, 9)
ALLOWED_TENSORS = {"policy.l3.bias", "policy.l3.weight"}
MAX_EDITS = 24
MAX_DELTA = 0.35


def write(path, obj):
    path.write_text(json.dumps(finite_json(obj), ensure_ascii=False, allow_nan=False, indent=2) + "\n")


@dataclass
class Packet:
    response: dict
    params: dict | None
    roundtrip_s: float | None
    sent_wall_s: float | None = None
    ready_wall_s: float | None = None
    error: str | None = None

    def record(self):
        return {"response": self.response, "roundtrip_s": self.roundtrip_s,
                "sent_wall_s": self.sent_wall_s, "ready_wall_s": self.ready_wall_s,
                "error": self.error, "valid_patch": self.params is not None}


def bounded_patch(text, actor):
    payload = json_payload(text)
    if not isinstance(payload, dict) or set(payload) != {"edits"} or not isinstance(payload["edits"], list):
        raise ValueError("Patch must be an object with an edits list")
    edits = payload["edits"]
    if not 1 <= len(edits) <= MAX_EDITS:
        raise ValueError(f"Provide 1 to {MAX_EDITS} changes")
    for row in edits:
        if (not isinstance(row, list) or len(row) != 3 or not isinstance(row[0], str)
                or row[0] not in ALLOWED_TENSORS or type(row[2]) not in (int, float)
                or not -MAX_DELTA <= row[2] <= MAX_DELTA):
            raise ValueError("Patch outside approved final-layer tensors or delta limit")
    return sparse_edit(text, actor.params)


def event_obs(event):
    obs = np.asarray(event["obs"], dtype=np.float64)
    if obs.shape != (51,):
        raise ValueError("Event must include the current 51-value observation")
    return obs


def prior_obs(event):
    points = []
    for item in event.get("recent", event.get("history", [])):
        if isinstance(item, dict):
            item = item.get("obs", item.get("observation", []))
        x = np.asarray(item, dtype=np.float64)
        if x.shape == (51,):
            points.append(x)
    return points


def prompt(actor, io, event):
    now = event_obs(event)
    previous = prior_obs(event)
    samples = previous[-12::3] + [now]
    commands, hidden = actor.forward(np.stack(samples))
    change = np.abs(hidden[-1] - hidden[0])
    top = np.argsort(change)[-8:][::-1].tolist()
    w = actor.params["policy.l3.weight"]
    b = actor.params["policy.l3.bias"]
    public = {
        "state_history": np.stack(samples).round(4).tolist(),
        "base_targets": commands[-1].round(4).tolist(),
        "reference_lateral_velocity": event["reference_lateral_velocity_mps"],
        "channels": io["channels"],
        "current_hidden_columns": [{"j": j, "h": hidden[:, j].round(4).tolist(),
                                    "weights_by_actuator": w[:, j].round(4).tolist()} for j in top],
        "output_bias": b.round(4).tolist(),
    }
    return (
        "H1 is currently walking under a pretrained DR.Q actor. Proprioception just detected a sudden lateral "
        "disturbance. The simulator does not pause; help the next available actor tick recover direction and "
        "normal walking speed without falling. You get only this actual observed state/history, not the disturbance "
        "force, timing schedule, future state or any action teacher. The frozen encoder feeds a 512-unit ReLU "
        "last hidden layer; the output is tanh(W*h+b) for 19 normalized position targets at 25 Hz. Native state "
        "is qpos[0:26] (pelvis xyz, quaternion wxyz, joint angles), qvel[0:25] (root xyz velocity, root angular "
        "rates xyz, joint velocities).\n"
        "Return patch with payload JSON {\"edits\":[[\"policy.l3.bias\",[output_i],0.03],"
        "[\"policy.l3.weight\",[output_i,hidden_j],-0.02]]} or pass with empty payload. "
        "At most 24 nonzero additive deltas, each within [-0.35,0.35]. The patch is relative to original "
        "weights, remains installed once applied, and is not code or a residual controller. Use a small "
        "well-directed correction; finish quickly. If no justified change, pass.\n"
        + json.dumps(public, separators=(",", ":"), allow_nan=False)
    )


def request(actor, url, token, thread, message):
    payload = json.dumps({"thread": thread, "phase": "playground", "prompt": message, "images": []}).encode()
    req = urllib.request.Request(url, data=payload, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"})
    sent = time.monotonic()
    action = {}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=90) as response:
            reply = json.load(response)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "Astra relay rejected turn"))
        action = reply["action"]
        if action["op"] == "patch":
            params = bounded_patch(action["payload"], actor)
        elif action["op"] == "pass" and not action["payload"]:
            params = None
        else:
            raise ValueError("Unexpected Astra operation")
        done = time.monotonic()
        return Packet(action, params, done - sent, sent, done)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        done = time.monotonic()
        return Packet(action, None, done - sent, sent, done, f"{type(exc).__name__}: {exc}")


def summarize(result, scenario, ref):
    m = result["metrics"]
    if ref is not None:
        post = window(result["trace"], scenario)
        matched = {"met_tolerances": False}
        if post is not None:
            d = post - ref
            yaw = float(np.degrees(np.arctan2(np.sin(d[2]), np.cos(d[2]))))
            matched.update(vx_error_mps=float(d[0]), vy_error_mps=float(d[1]), yaw_error_deg=yaw,
                           met_tolerances=bool(abs(d[0]) <= .2 and abs(d[1]) <= .15 and abs(yaw) <= 10))
        m["matched_recovery"] = matched
        m["recovered_and_finished"] = bool(m["completed_horizon"] and not m["physically_down"]
                                            and matched["met_tolerances"])
    result["trace"] = result["trace"][::5]
    return result


def step_of(event):
    return int(event["trigger_step"])


def run_case(args, actor, io, scenario, name, root, *, video=False):
    out = root / name
    out.mkdir()
    trigger = TriggerConfig()
    ref = None
    if scenario.push_step is not None:
        clean = replace(scenario, push_step=None, lateral_impulse_ns=0.0)
        nominal = live_rollout(Controller(actor), clean, trace_every=1)
        ref = window(nominal["trace"], scenario)
        if ref is None:
            raise RuntimeError("Same-seed original clean reference does not reach the test window")
    ctrl = Controller(actor)
    events, applied = [], []
    token = args.token_file.read_text().strip()

    def install(packet, event):
        if packet.params is not None:
            ctrl.params = packet.params
            applied.append({"trigger_step": step_of(event), "controller_tick": ctrl.count,
                            "simulation_s": round(.02 * ctrl.count, 4),
                            "wall_s": time.monotonic()})
            return True
        return False

    with ThreadPoolExecutor(max_workers=1) as pool:
        def on_event(event):
            message = prompt(actor, io, event)
            f = pool.submit(request, actor, args.brain, token,
                            f"{args.run.name}-h1-live-{name}", message)
            events.append((event, f))
            return f

        r = live_rollout(ctrl, scenario, on_event=on_event, on_ready=install,
                         mode="async", realtime=True, video=out / "live.mp4" if video else None,
                         trace_every=1, trigger=trigger)
        r = summarize(r, scenario, ref)
        # Receive late replies for exact-patch replay only, never apply them to a finished trial.
        packets = [(e, f.result()) for e, f in events]

    report = {"scenario": r["scenario"], "live": r, "actual_patch_applies": applied,
              "requests": [{"event": e, **packet.record()} for e, packet in packets]}
    write(out / "live.json", report)

    if packets and packets[0][1].params is not None:
        event0, packet = packets[0]
        replay_ctrl = Controller(actor)
        replay_events, replay_applied = [], []

        def replay_event(event):
            if step_of(event) != step_of(event0) or not np.array_equal(event_obs(event), event_obs(event0)):
                raise RuntimeError("Replay did not reach the identical pre-edit native state")
            replay_events.append(event)
            return packet

        def replay_install(reply, event):
            replay_ctrl.params = reply.params
            replay_applied.append({"trigger_step": step_of(event), "controller_tick": replay_ctrl.count,
                                   "simulation_s": round(.02 * replay_ctrl.count, 4)})
            return True

        rr = live_rollout(replay_ctrl, scenario, on_event=replay_event, on_ready=replay_install,
                          mode="pause", realtime=False, video=out / "immediate.mp4" if video else None,
                          trace_every=1, trigger=trigger)
        if not replay_events:
            raise RuntimeError("Replay did not reproduce the trigger")
        report["immediate"] = summarize(rr, scenario, ref)
        report["immediate_patch_applies"] = replay_applied
        report["exact_trigger_observation_replay"] = True
    write(out / "paired.json", report)
    print(name, "trigger", [step_of(e) for e, _ in packets], "latency",
          [p.roundtrip_s for _, p in packets], "live", applied, "immediate",
          report.get("immediate_patch_applies"), flush=True)
    return {"scenario": report["scenario"], "requests": report["requests"],
            "actual_patch_applies": applied, "live_metrics": r["metrics"],
            "live_timing": r["live"],
            "immediate_metrics": report["immediate"]["metrics"] if "immediate" in report else None,
            "immediate_patch_applies": report.get("immediate_patch_applies", []),
            "exact_trigger_observation_replay": report.get("exact_trigger_observation_replay", False)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actor", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--brain", default="http://127.0.0.1:44193/turn")
    p.add_argument("--cases", nargs="+", type=int, choices=CASES, default=CASES)
    p.add_argument("--skip-clean", action="store_true")
    args = p.parse_args()
    args.run.mkdir()
    actor = load(args.actor)
    io = describe()
    plan = json.loads(args.plan.read_text())
    previous = json.loads(args.previous.read_text())
    records = {}
    for idx in args.cases:
        name = f"case-{idx:02d}"
        s = Scenario(**plan["holdout"][idx - 1])
        result = run_case(args, actor, io, s, name, args.run, video=(idx == 1))
        result["prior"] = {arm: previous[arm]["push"][idx - 1]["metrics"]
                           for arm in ("base", "wsp", "code")}
        records[name] = result
        write(args.run / "summary.json", records)
    if not args.skip_clean:
        clean = Scenario(**plan["nominal_holdout"][0])
        records["clean-01"] = run_case(args, actor, io, clean, "clean-01", args.run)
        write(args.run / "summary.json", records)


if __name__ == "__main__":
    with reusable_native_env():
        main()
