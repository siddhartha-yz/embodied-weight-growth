"""Astra writes once between two pushes while native H1 keeps walking."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np

from actor_source import load
from continuous_env import Pulse, Scenario, reusable_native_env, rollout
from edits import json_payload, sparse_edit
from push_env import DT, describe

HORIZON = 1570  # 31.4 s; the fixed actor rarely remains upright past ~30–35 s
PUSH_STEPS = (110, 1400)  # 2.20 s and 28.00 s
PUSH_TICKS = 5
IMPULSE_NS = 104.5
REQUEST_STEP = PUSH_STEPS[0] + PUSH_TICKS + 31  # 2.92 s: first even actor tick after 0.6 s feedback
ALLOWED_TENSORS = {"policy.l3.weight", "policy.l3.bias"}
CASE_ORDER = (11, 2, 3, 5, 8, 12, 9, 4, 6, 7, 10, 1)


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def wrap(x):
    return (x + 180.) % 360. - 180.


def window(trace, tick):
    if len(trace) <= tick or tick < 14:
        return None
    rows = np.asarray([r["obs"] for r in trace[tick-14:tick+1]])
    if len(rows) != 15:
        return None
    w, x, y, z = rows[:, 3:7].T
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2*(y*y+z*z))
    return [float(rows[:, 26].mean()), float(rows[:, 27].mean()),
            math.degrees(math.atan2(float(np.sin(yaw).mean()), float(np.cos(yaw).mean())))]


def down_step(episode):
    """First confirmed pelvis-down segment, using the previous 0.60m/0.20s rule."""
    n = 0
    trace = episode["trace"]
    for row in trace:
        n = n + 1 if row["obs"][2] < .6 else 0
        if n >= 10:
            return row["step"] - 9
    if trace[-1]["obs"][2] < .6 or episode["metrics"].get("fell", False):
        return trace[-1]["step"]
    return None


def pulse_meta(p):
    return {**asdict(p), "start_s": p.push_step * DT, "end_s": (p.push_step + PUSH_TICKS) * DT,
            "target_s": (p.push_step + PUSH_TICKS + 100) * DT}


def recover(episode, pulse, target):
    point = pulse.push_step + PUSH_TICKS + 100
    after = window(episode["trace"], point)
    down = down_step(episode)
    r = {"target_s": point*DT, "reference": target, "observed": after,
         "recovered_and_stayed_up": False}
    if after is not None:
        vx, vy, yaw = after
        dvx, dvy, dyaw = vx-target[0], vy-target[1], wrap(yaw-target[2])
        r.update(vx_error_mps=dvx, vy_error_mps=dvy, yaw_error_deg=dyaw,
                 recovered_and_stayed_up=bool(down is None and
                    episode["trace"][-1]["step"] == HORIZON and
                    abs(dvx) <= .2 and abs(dvy) <= .15 and abs(dyaw) <= 10))
    return r


@dataclass
class Packet:
    action: dict
    params: dict | None
    sent_wall_s: float
    ready_wall_s: float
    error: str | None = None

    def record(self):
        return {"response": self.action, "roundtrip_s": self.ready_wall_s-self.sent_wall_s,
                "sent_wall_s": self.sent_wall_s, "ready_wall_s": self.ready_wall_s,
                "error": self.error, "valid_patch": self.params is not None}


def patch_params(text, current):
    payload = json_payload(text)
    if not isinstance(payload, dict) or set(payload) != {"edits"} or not isinstance(payload["edits"], list):
        raise ValueError("Need an edits list")
    if not 1 <= len(payload["edits"]) <= 24:
        raise ValueError("Need 1 to 24 head coefficient edits")
    for e in payload["edits"]:
        if (not isinstance(e, list) or len(e) != 3 or not isinstance(e[0], str)
                or e[0] not in ALLOWED_TENSORS or type(e[2]) not in (int, float)
                or not -0.35 <= e[2] <= 0.35):
            raise ValueError("Head parameter delta exceeds allowed tensor/range")
    return sparse_edit(text, current)


def relay_request(url, token, thread, prompt, params):
    data = json.dumps({"thread": thread, "phase": "playground", "prompt": prompt, "images": []}).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"})
    sent = time.monotonic()
    action = {}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=95) as resp:
            reply = json.load(resp)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "Relay rejected the turn"))
        action = reply["action"]
        if action["op"] == "patch":
            edited = patch_params(action["payload"], params)
        elif action["op"] == "pass" and not action["payload"]:
            edited = None
        else:
            raise ValueError("Unexpected relay action")
        return Packet(action, edited, sent, time.monotonic())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return Packet(action, None, sent, time.monotonic(), f"{type(exc).__name__}: {exc}")


class Policy:
    def __init__(self, actor, params, *, submit=None, clean=None, channels=None):
        self.actor, self.params = actor, params
        self.submit, self.clean, self.channels = submit, clean, channels
        self.history = []
        self.held = None
        self.future = None
        self.packet = None
        self.install = None

    def __call__(self, obs, step):
        if self.submit is not None:
            if len(self.history) != step:
                raise AssertionError("Unexpected physical clock discontinuity")
            self.history.append(obs.copy())
        if step % 2:
            return self.held
        if self.future is not None and self.packet is None and self.future.done():
            self.packet = self.future.result()
            if self.packet.params is not None:
                pre, _ = self.actor.forward(obs, self.params)
                self.params = self.packet.params
                post, _ = self.actor.forward(obs, self.params)
                scales = [(c["max_rad"]-c["min_rad"])/2 for c in self.channels]
                self.install = {"step": step, "simulation_s": step*DT, "wall_s": time.monotonic(),
                                "same_state_action_delta_l2_rad": float(np.linalg.norm((post-pre)*scales)),
                                "same_state_action_delta_linf_rad": float(np.max(np.abs((post-pre)*scales)))}
                self.held = post
                return self.held
        if self.submit is not None and step == REQUEST_STEP:
            self.future = self.submit(self.message())
        self.held, _ = self.actor.forward(obs, self.params)
        return self.held

    def message(self):
        pulse = PUSH_STEPS[0]
        steps = [pulse-10, pulse, *np.linspace(pulse + PUSH_TICKS, REQUEST_STEP-1, 6, dtype=int).tolist()]
        pushed = np.asarray([self.history[t] for t in steps])
        normal = np.asarray([self.clean["trace"][t]["obs"] for t in steps])
        nominal_a, h0 = self.actor.forward(normal, self.params)
        pushed_a, h1 = self.actor.forward(pushed, self.params)
        h_change = np.max(np.abs(h1[2:]-h0[2:]), axis=0)
        top = np.argsort(h_change)[-10:][::-1].tolist()
        w = self.params["policy.l3.weight"]
        b = self.params["policy.l3.bias"]
        public = {"seconds_relative_to_first_push": [(t-pulse)*DT for t in steps],
                  "first_push_observed_qpos_qvel": pushed.round(4).tolist(),
                  "same_seed_no_push_qpos_qvel": normal.round(4).tolist(),
                  "actor_targets_on_push": pushed_a.round(4).tolist(),
                  "actor_targets_on_nominal": nominal_a.round(4).tolist(),
                  "output_channels": self.channels, "output_bias": b.round(4).tolist(),
                  "changed_final_units": [{"index": j, "nominal": h0[:,j].round(4).tolist(),
                      "pushed": h1[:,j].round(4).tolist(), "current_w_by_output": w[:,j].round(4).tolist()}
                      for j in top]}
        return (
            "You are adjusting a currently walking pretrained H1 actor after its first lateral disturbance. "
            "Physics continues while you respond; apply a modest parameter change that preserves nominal gait "
            "and improves response if a similar lateral disturbance occurs again. You see only actual first-push "
            "proprioception and a matched no-push trajectory, not any future force or schedule. "
            "Observations are qpos 26 (pelvis xyz, quaternion wxyz, joint angles) then qvel 25 (root linear xyz, "
            "angular xyz, joint velocities). Actor is 25 Hz with fixed encoder/final ReLU units and "
            "normalized actuator target tanh(W*h+b); physical ranges in output_channels. Current model already "
            "includes the frozen torso patch; deltas apply to these CURRENT weights. "
            "Submit patch payload JSON {\"edits\":[[\"policy.l3.weight\",[output_i,hidden_j],0.05],"
            "[\"policy.l3.bias\",[output_i],-0.02]]}, max 24 nonzero deltas within +/-0.35, or pass with empty payload. "
            "Parameters are installed immediately at response and stay in use; not a transient external controller. "
            "Prefer a small activation-specific correction with little nominal effect, no code or BP.\n"
            + json.dumps(public, separators=(",", ":"), allow_nan=False)
        )


def finished(r):
    return r["trace"][-1]["step"] == HORIZON and down_step(r) is None


def baseline_roll(actor, params, seed, pulses):
    return rollout(Policy(actor, params), Scenario(seed=seed,horizon_steps=HORIZON,pulses=tuple(pulses)))


def choose_case(args, actor, params, pulses):
    seeds = json.loads((args.prior / "calibrated/plan.json").read_text())["holdout"]
    trials = []
    alternate = None
    for case in CASE_ORDER:
        seed = seeds[case-1]["seed"]
        clean = baseline_roll(actor, params, seed, ())
        first = baseline_roll(actor, params, seed, pulses[:1])
        if not (finished(clean) and finished(first)):
            trials.append({"case":case,"seed":seed,"eligible":False,
                           "clean_down":down_step(clean),"first_only_down":down_step(first)})
            print("calibration reject",case,trials[-1],flush=True)
            continue
        fixed = baseline_roll(actor, params, seed, pulses)
        second_target = window(first["trace"], pulses[1].push_step+PUSH_TICKS+100)
        first_target = window(clean["trace"], pulses[0].push_step+PUSH_TICKS+100)
        result = recover(fixed,pulses[1],second_target)
        rec1 = recover(first,pulses[0],first_target)
        trials.append({"case":case,"seed":seed,"eligible":True,
                       "fixed_down":down_step(fixed),"first":rec1,"second":result})
        print("calibration candidate",case,"down",down_step(fixed),"second",result,flush=True)
        candidate = (case,seed,clean,first,fixed,first_target,second_target)
        # Prefer a fixed arm that is still upright at the judging window and misses recovery,
        # so this is not just a floor-vs-walking comparison. Only fixed outputs inform selection.
        if result["observed"] is not None and (down_step(fixed) is None or
                down_step(fixed) > pulses[1].push_step+PUSH_TICKS+100):
            if alternate is None:
                alternate = candidate
            if not result["recovered_and_stayed_up"]:
                return candidate,trials
    if alternate is not None:
        return alternate,trials
    raise RuntimeError(f"No suitable original-patch continuous episode: {trials}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--brain", default="http://127.0.0.1:44194/turn")
    args = p.parse_args()
    args.run.mkdir()
    actor = load(args.prior / "drq_checkpoint/actor_bundle.npz")
    params = sparse_edit((args.prior / "run/wsp/selected.json").read_text(),actor.params)
    pulses = tuple(Pulse(step,IMPULSE_NS) for step in PUSH_STEPS)
    (case,seed,clean,first,fixed,target1,target2), tried = choose_case(args,actor,params,pulses)
    write(args.run/"calibration.json", {"selected_case":case,"seed":seed,"fixed_only":tried,
                                        "pulses":[pulse_meta(x) for x in pulses]})
    print("Selected fixed-only calibration case",case,"starting real-time run",flush=True)
    token = args.token_file.read_text().strip()
    channels = describe()["channels"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        def submit(message):
            print("Astra request at simulation",REQUEST_STEP*DT,flush=True)
            return pool.submit(relay_request,args.brain,token,f"{args.run.name}-h1-two-push",message,params)
        ctl = Policy(actor,params,submit=submit,clean=clean,channels=channels)
        live = rollout(ctl,Scenario(seed=seed,horizon_steps=HORIZON,pulses=pulses),realtime=True)
        packet = ctl.future.result() if ctl.future is not None else None

    # Identical native state and actions through the install observation. Comparing
    # only those fields keeps real wall time separate from simulation equivalence.
    n = ctl.install["step"]+1 if ctl.install else len(live["trace"])
    def physics(r):
        return [(x["step"],x["obs"],x["action"]) for x in r["trace"][:n]]
    if physics(live) != physics(fixed) or len(fixed["trace"]) < n:
        raise AssertionError("Fixed and live diverged before the actual weight installation")
    if not ctl.install and len(live["trace"]) != len(fixed["trace"]):
        raise AssertionError("No installed patch, but episode length differed")
    patch = {"request_step": REQUEST_STEP if packet else None,
             "install_step": ctl.install["step"] if ctl.install else None,
             "installation": ctl.install, "installed_before_second_push": bool(
                 ctl.install and ctl.install["step"] < pulses[1].push_step),
             **(packet.record() if packet else {})}
    if packet and packet.params is not None:
        edits = json_payload(packet.action["payload"])["edits"]
        patch["edits_count"] = len(edits)
        patch["edits"] = edits
    arms = {"clean":clean,"first_only":first,"fixed":fixed,"live":live}
    for key,r in arms.items():
        r["metrics"].update({"physical_down_step":down_step(r),"completed_without_down":finished(r)})
        if key in ("fixed","live"):
            r["recovery"] = [recover(r,p,t) for p,t in zip(pulses,(target1,target2),strict=True)]
            pulse2 = r["trace"][pulses[1].push_step]["obs"] if len(r["trace"]) > pulses[1].push_step else None
            r["pre_second_push"] = {"vx_mps":pulse2[26],"vy_mps":pulse2[27],"pelvis_z_m":pulse2[2]} if pulse2 else None
    report = {"scenario":{"seed":seed,"horizon_steps":HORIZON,"source_seed_case":case},
              "pulses":[pulse_meta(p) for p in pulses],"patch":patch,
              "prefix_identical_through_install_observation":True,
              "second_reference":"same frozen checkpoint, same seed, first push only",
              "arms":arms}
    write(args.run/"two-push.json",report)
    compact = {**report,"arms":{k:{kk:vv for kk,vv in v.items() if kk!="trace"} for k,v in arms.items()}}
    write(args.run/"summary.json",compact)
    print(json.dumps({"patch":patch,"fixed_second":fixed["recovery"][1],"live_second":live["recovery"][1]},
                     ensure_ascii=False),flush=True)


if __name__ == "__main__":
    with reusable_native_env():
        main()
