"""Matched H1 recovery pilot: untouched base, sparse WSP, and code residual."""
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import json
from pathlib import Path
import signal
import time
import urllib.request

import numpy as np

from actor_source import load
from edits import Controller, json_payload, residual_code, sparse_edit
from push_env import Scenario, describe, reusable_native_env, rollout


def write(path, x):
    path.write_text(json.dumps(x, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def jwrite(path, x):
    with path.open("a") as f:
        f.write(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n")


def finite_json(x):
    if isinstance(x, dict):
        return {k: finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in x]
    if isinstance(x, np.generic):
        return finite_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def window(trace, scenario):
    tick = scenario.push_step + 5 + 100
    x = [row["obs"] for row in trace if tick - 14 <= row["step"] <= tick]
    if len(x) != 15:
        return None
    x = np.asarray(x)
    w, qx, qy, qz = x[:, 3:7].T
    yaw = np.arctan2(2 * (w*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return np.array([x[:, 26].mean(), x[:, 27].mean(),
                     np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean())])


def sim(ctrl, scenario, *, capture=False, video=None, trace_every=0, reference=None):
    ctrl.reset()
    ctrl.collect = capture
    result = rollout(ctrl, scenario, video=video, trace_every=1 if reference is not None else trace_every)
    ctrl.collect = False
    if reference is not None:
        m = result["metrics"]
        actual = window(result["trace"], scenario)
        ref = {"comparison": "same-seed unpushed original actor at the same two-second window",
               "met_tolerances": False}
        if actual is not None:
            d = actual - reference
            dyaw = float(np.degrees(np.arctan2(np.sin(d[2]), np.cos(d[2]))))
            ref.update(vx_error_mps=float(d[0]), vy_error_mps=float(d[1]), yaw_error_deg=dyaw,
                       nominal_vx_mps=float(reference[0]),
                       met_tolerances=bool(abs(d[0]) <= .2 and abs(d[1]) <= .15 and abs(dyaw) <= 10))
        m["matched_recovery"] = ref
        m["recovered_and_finished"] = bool(m["completed_horizon"] and not m["physically_down"]
                                            and ref["met_tolerances"])
        result["trace"] = result["trace"][::trace_every] if trace_every else []
    return finite_json(result)


def references(actor, scenarios):
    ctrl = Controller(actor)
    items = {}
    for scenario in scenarios:
        nominal = replace(scenario, push_step=None, lateral_impulse_ns=0.0)
        result = sim(ctrl, nominal, trace_every=1)
        ref = window(result["trace"], scenario)
        if ref is None:
            raise RuntimeError("Original unpushed actor did not reach the matched recovery window")
        items[scenario] = ref
    return items


def compact(result, *, private=False):
    out = {"metrics": result["metrics"]}
    if private:
        out["scenario"] = result["scenario"]
        out["audit"] = result["audit"]
    return out


def stats_rows(rows):
    metrics = [r["metrics"] for r in rows]
    return {
        "recover": sum(bool(x["recovered_and_finished"]) and x["completed_horizon"] for x in metrics),
        "upright": sum(x["completed_horizon"] and not x["physically_down"] for x in metrics),
        "mean_return": float(np.mean([x["return"] for x in metrics])),
        "mean_vx_mps": float(np.mean([x["mean_vx_mps"] for x in metrics])),
        "n": len(metrics),
    }


def diagnose(logs, scenario, hidden_indexes=None, *, match_step=None):
    if not logs:
        return {}
    push = scenario.push_step if match_step is None else match_step
    n = len(logs)
    pivot = min(max(5, push or 50), n - 1)
    before = np.asarray([v[3] for v in logs[max(0, pivot - 20):pivot + 1]])
    after = np.asarray([v[3] for v in logs[pivot:min(pivot + 30, n)]])
    if hidden_indexes is None:
        change = np.abs(after.mean(axis=0) - before.mean(axis=0))
        hidden_indexes = np.argsort(change)[-10:][::-1].tolist()
    points = sorted(set([min(x, n - 1) for x in [0, max(0, pivot - 1), pivot + 5,
                                                  pivot + 15, pivot + 30, pivot + 75]] + [n - 1]))
    return {"hidden_indexes": hidden_indexes, "points": [
        {"step": t, "qpos": logs[t][1][:26].round(4).tolist(),
         "qvel": logs[t][1][26:].round(4).tolist(),
         "action": logs[t][2].round(4).tolist(),
         "hidden": logs[t][3][hidden_indexes].round(4).tolist()} for t in points]}


def paired_diagnostic(actor, scenario, reference):
    original = Controller(actor)
    pushed = sim(original, scenario, capture=True, reference=reference)
    push_log = original.logs
    clean_case = replace(scenario, push_step=None, lateral_impulse_ns=0.0)
    clean = sim(original, clean_case, capture=True)
    clean_log = original.logs
    start = scenario.push_step + 5
    stop = min(start + 30, len(push_log), len(clean_log))
    if start < stop:
        delta = np.stack([push_log[i][3] - clean_log[i][3] for i in range(start, stop)])
        hidden = np.argsort(np.mean(delta * delta, axis=0))[-10:][::-1].tolist()
    else:
        hidden = None
    return {"push": {**compact(pushed), "activation": diagnose(push_log, scenario, hidden)},
            "matched_clean": {**compact(clean), "activation": diagnose(
                clean_log, clean_case, hidden, match_step=scenario.push_step)}}


class Relay:
    def __init__(self, url, token):
        self.url, self.token = url, token
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def ask(self, thread, prompt, images=()):
        payload = {"thread": thread, "phase": "playground", "prompt": prompt,
                   "images": [{"name": p.name,
                               "base64": base64.b64encode(p.read_bytes()).decode("ascii")} for p in images]}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.token}"})
        with self.opener.open(req, timeout=330) as resp:
            result = json.load(resp)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Relay failed"))
        return result["action"]


def arm_spec(arm):
    if arm == "wsp":
        return ('You may change only existing numeric parameters; no residual code, optimization or new network. '
                'submit payload is a complete patch relative to the ORIGINAL checkpoint, as '
                '{"edits":[["tensor_name",[row,column],0.01],["bias_name",[index],-0.02]]}. '
                'At most 192 nonzero additive changes across at most 2 tensors; each delta in [-5,5]. '
                'Use the actual exact tensor names/orientation in the actor manifest.')
    return ('The ORIGINAL trained actor is frozen. You may submit Python implementing '
            'def residual(obs, base_action, hidden, memory): returning 19 finite normalized corrections. '
            'Host adds it to base_action and clips to [-1,1]. Runs on the native trained actor 25 Hz cadence; each command holds for two 50 Hz physics controls. '
            'obs is flat [26 qpos,25 qvel]; hidden is the trained actor final hidden vector; memory is an '
            'initially empty dict retained within a rollout. Only math/numpy imports and plain functions; '
            'no tools, classes, globals, file I/O or extra sensors. Use a state-dependent low-level correction.')


def actor_spec(actor):
    io = describe()
    return {"params": {k: list(a.shape) for k, a in actor.params.items()},
            "actor": getattr(actor, "description", ""), "channels": io["channels"],
            "hidden_size": len(actor.forward(np.zeros(51))[1]),
            "can_mute_final_hidden": bool(getattr(actor, "can_mute", False))}


def candidate(arm, actor, payload):
    if arm == "wsp":
        return Controller(actor, params=sparse_edit(payload, actor.params))
    return Controller(actor, residual=residual_code(payload))


def probe(payload, actor, baseline, current, scenarios, refs):
    p = json_payload(payload)
    if not isinstance(p, dict):
        raise ValueError("Probe payload must be a JSON object")
    kind = p.get("kind", "rollout")
    if kind == "weights":
        name = p.get("tensor")
        x = actor.params.get(name)
        if x is None:
            raise ValueError("Unknown tensor")
        axis, index = p.get("axis", 0), p.get("index", 0)
        if (type(axis) is not int or not 0 <= axis < x.ndim
                or type(index) is not int or not 0 <= index < x.shape[axis]):
            raise ValueError("Axis or index outside tensor")
        values = np.asarray(np.take(x, index, axis=axis)).ravel()
        if len(values) > 300:
            raise ValueError("Requested slice exceeds 300 values")
        return {"tensor": name, "axis": axis, "index": index, "values": values.round(7).tolist()}
    if kind != "rollout":
        raise ValueError("Probe kind must be rollout or weights")
    slot = p.get("scenario", 0)
    if type(slot) is not int or not 0 <= slot < len(scenarios):
        raise ValueError("Invalid public scenario number")
    scenario = scenarios[slot]
    choose = p.get("policy", "base")
    if choose not in {"base", "best"}:
        raise ValueError("Probe policy must be base or best")
    ctrl = baseline if choose == "base" else current
    mute = p.get("mute", [])
    if mute:
        n = len(actor.forward(np.zeros(51))[1])
        if (choose != "base" or not getattr(actor, "can_mute", False) or not isinstance(mute, list)
                or len(mute) > 8 or any(type(i) is not int or not 0 <= i < n for i in mute)):
            raise ValueError("Mute up to 8 named final-hidden indexes on the base, if supported")
        ctrl = Controller(actor, mute=mute)
    hidden = p.get("hidden", None)
    if hidden is not None:
        n = len(actor.forward(np.zeros(51))[1])
        if (not isinstance(hidden, list) or len(hidden) > 12
                or any(type(i) is not int or not 0 <= i < n for i in hidden)):
            raise ValueError("Inspect up to 12 valid hidden indexes")
    r = sim(ctrl, scenario, capture=True, reference=refs.get(scenario))
    return {**compact(r), "activation": diagnose(ctrl.logs, scenario, hidden)}


def deadline(_signum, _frame):
    raise TimeoutError("Candidate/probe exceeded 60 seconds on the lab simulator")


def edit_arm(arm, actor, relay, run, train, clean, base_train, base_clean, spec, diagnostic, refs, turns):
    out = run / arm
    out.mkdir()
    baseline = Controller(actor)
    best = baseline
    best_payload = None
    before = stats_rows(base_train)
    best_key = (before["recover"], before["upright"], before["mean_return"])
    report = None
    initial = (
        "H1 already walks normally on flat ground. Its problem is recovering from an unpredictable lateral "
        "100ms torso push while maintaining the original walking gait otherwise. Input qpos is pelvis xyz, "
        "quaternion wxyz, then 19 joint positions. qvel is root xyz velocity, angular velocity, then "
        "19 joint velocities. Push timing and force are NOT controller inputs. Native step is 0.02 seconds; the trained actor and both repairs update every 0.04 seconds. "
        "Practice and frozen testing are separate. You must repair the same base policy, not build a new gait. "
        "Main outcome is recovered_and_finished (returns to the same-seed unpushed original gait speed/heading at 2 s and completes 20 s); nominal must stay upright with at least 85% of its original return. "
        + arm_spec(arm) +
        "\n\nSame probes for both arms. Use probe JSON "
        '{"kind":"rollout","scenario":0,"policy":"base","hidden":[0,1],"mute":[]} to see trajectory '
        "and activations or mute a few final hidden units (optional keys). Use "
        '{"kind":"weights","tensor":"NAME","axis":0,"index":0} to read a slice. '
        "Start by examining evidence rather than replacing a stable controller. Each candidate runs the same "
        "practice pushes and one normal episode. Candidates are complete replacements for earlier candidate artifacts. "
        "The host keeps the best practice candidate subject to nominal preservation. Finish when appropriate.\n"
        + json.dumps({"actor_manifest": spec,
                      "public_scenarios": [{"index": i, "kind": "push" if x.push_step is not None else "clean"}
                                           for i, x in enumerate(train + clean)],
                      "base_on_practice": base_train, "base_on_clean": base_clean,
                      "paired_activation_preview": diagnostic}, ensure_ascii=False)
    )
    for turn in range(1, turns + 1):
        prompt = (initial + "\n" if turn == 1 else "") + f"Turn {turn}/{turns}. "
        if report is not None:
            prompt += "Previous result: " + json.dumps(report, ensure_ascii=False)
        started = time.monotonic()
        action = relay.ask(f"{run.name}-h1-push-{arm}", prompt)
        jwrite(out / "turns.jsonl", {"turn": turn, **action, "seconds": time.monotonic() - started})
        op = action["op"]
        if op == "finish":
            break
        signal.setitimer(signal.ITIMER_REAL, 60)
        try:
            if op == "probe":
                report = probe(action["payload"], actor, baseline, best, train + clean, refs)
            else:
                new = candidate(arm, actor, action["payload"])
                seen = [compact(sim(new, s, reference=refs[s])) for s in train]
                normal = [compact(sim(new, s)) for s in clean]
                scores = stats_rows(seen)
                keep = all(a["metrics"]["completed_horizon"] and not a["metrics"]["physically_down"]
                           and a["metrics"]["return"] >= 0.85 * b["metrics"]["return"]
                           for a, b in zip(normal, base_clean, strict=True))
                key = scores["recover"], scores["upright"], scores["mean_return"]
                accepted = keep and key > best_key
                if accepted:
                    best_key, best, best_payload = key, new, action["payload"]
                    (out / ("selected.json" if arm == "wsp" else "selected.py")).write_text(best_payload + "\n")
                report = {"practice": seen, "clean": normal, "clean_guard_passed": keep,
                          "selected_as_current_best": accepted}
        except (ValueError, TypeError, SyntaxError, KeyError, IndexError, NameError, AttributeError,
                RuntimeError, ArithmeticError, TimeoutError) as exc:
            report = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        jwrite(out / "feedback.jsonl", {"turn": turn, "op": op, "result": report})
    write(out / "selection.json", {"selected_edit": best_payload is not None, "practice_key": best_key})
    return best


def train_eval(args, actor, run, plan):
    sc = {k: [Scenario(**s) for s in items] for k, items in plan.items()}
    train, train_clean = sc["train"], sc["nominal_train"]
    base = Controller(actor)
    refs = references(actor, train)
    base_train = [compact(sim(base, s, reference=refs[s])) for s in train]
    base_clean = [compact(sim(base, s)) for s in train_clean]
    write(run / "practice-base.json", {"push": base_train, "clean": base_clean})
    spec = actor_spec(actor)
    write(run / "actor-manifest.json", spec)
    relay = Relay(args.brain, args.token_file.read_text().strip())
    index = next((i for i, r in enumerate(base_train) if r["metrics"]["physically_down"]), 0)
    diagnostic = {"practice_slot": index, **paired_diagnostic(actor, train[index], refs[train[index]])}
    write(run / "shared-diagnostic.json", diagnostic)
    controllers = {"base": base}
    signal.signal(signal.SIGALRM, deadline)
    for arm in ("wsp", "code"):
        controllers[arm] = edit_arm(arm, actor, relay, run, train, train_clean,
                                    base_train, base_clean, spec, diagnostic, refs, args.turns)
    results = {}
    heldout = sc["holdout"]
    heldout_refs = references(actor, heldout)
    for name, ctrl in controllers.items():
        push = [sim(ctrl, s, video=(run / f"{name}-push.mp4") if i == 0 else None, trace_every=5, reference=heldout_refs[s])
                for i, s in enumerate(heldout)]
        clean = [sim(ctrl, s) for s in sc["nominal_holdout"]]
        results[name] = {"push": push, "clean": clean, "scores": stats_rows(push)}
        write(run / "heldout.json", results)
    print(json.dumps({k: x["scores"] for k, x in results.items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actor", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--brain", default="http://127.0.0.1:44192/turn")
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--turns", type=int, default=12)
    args = p.parse_args()
    args.run.mkdir()
    plan = json.loads(args.plan.read_text())
    actor = load(args.actor)
    train_eval(args, actor, args.run, plan)


if __name__ == "__main__":
    with reusable_native_env():
        main()
