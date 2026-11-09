"""Run one GPT-6 Astra closed-loop walking session against a MuJoCo biped."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

from openai_codex import ApprovalMode, Codex, CodexConfig, LocalImageInput, Sandbox, TextInput

from walk_view import render_view

MODEL = "gpt-6-astra"
DECISIONS = 36
PASSIVE_ITEMS = {"userMessage", "reasoning", "agentMessage", "plan"}
BASE = """You are the brain of a simulated planar biped. Make it walk forward using its own legs.
The body balances freely in the vertical plane under gravity. You have six normalized motor inputs [-1,1] in this order:
right hip, right knee, right ankle, left hip, left knee, left ankle. No gait, stabilizer, pose controller, route or joint targets are provided.
At each physical control tick (0.02 s) your installed module receives actual sensors and returns those six motor commands.
Positive x is forward, z is height. Observations are meters, radians, seconds and derived velocities. Contact flags are right then left.
Positive torso pitch tilts forward. Hip and knee angle limits are approximately [-2.62,0] rad; ankles have limits ±0.785 rad. A normalized motor command of ±1 requests ±100 N·m at that joint.
The goal is forward walking, not merely falling forward or sliding: clear at least 1 meter, make at least four alternating forward touchdowns, and remain upright after at least 2 seconds. Nominal forward speed is 0.5 m/s.
To count as a step, the swinging foot must leave the ground for at least 0.06 s, advance at least 4.5 cm before landing, and have the opposite foot support it at some point in the swing.
Operations:
- drive: repeat a raw six-number u for ticks. Useful for brief probes; an unsupported body can fall.
- install: replace YOUR controller artifact; no motion. A fresh simulator has no installed controller.
- run: execute YOUR controller for ticks. The simulator stops early on a fall. You get the motion trace after the action.
- reset: restart from the same initial body and pose, keeping your installed controller, spent budget and the conversation.
- finish: end the pilot when you have demonstrated walking or cannot progress within budget.
Budget: 36 decisions, 24 motion commands, 1600 control ticks total, at most 250 ticks per motion. The simulation stays paused between decisions.
Use short diagnostic runs as needed. Read the actual pose, speed and contacts after each run. When you observe a reproducible limitation, revise the module and test again; a stable first module can be kept.
Do not call any Codex tools. Respond with exactly one JSON action. In note, say what you observed and what this next action tests.
Unused fields are u=[], layers=[], source="", ticks=0. Do not invent successful steps absent the observations.
"""
WEIGHTS = """Submit explicit numeric weights, not source. A sparse layer is {"rows":[{"terms":[{"feature":index,"coef":number},...],"bias":number},...]}; absent terms are zero.
Use one linear layer, or up to two hidden layers (tanh after each, width 1..24) and a linear output layer with exactly six rows. In later layers the feature indices index the preceding layer's output, starting at zero.
Input order:
0:constant one, 1:z, 2:pitch, 3:vx, 4:vz, 5:pitch_v,
6..11:q0..q5, 12..17:dq0..dq5, 18:right contact, 19:left contact,
20:sin(2πt), 21:cos(2πt), 22:t, 23:target_v, 24:x.
The clock is a raw 1 Hz time feature, not a step planner. q and dq follow the six motor indices. Outputs are clipped to [-1,1]. Use source="".
"""
CODE = """Submit restricted Python, exactly def policy(obs, goal): returning six numbers.
obs scalar names are z,pitch,vx,vz,pitch_v,q0..q5,dq0..dq5,contact0,contact1,clock_sin,clock_cos,t,target_v,x. Also obs["q"], obs["dq"], obs["contact"] are the corresponding arrays; goal=[0.5].
clock_sin and clock_cos are sin(2πt) and cos(2πt), not a gait planner. Outputs are clipped to [-1,1].
Use ordinary assignments, comparisons, arithmetic, indexing, if/else, and return. Functions: abs,min,max,sin,cos,tan,sqrt,atan2,hypot,exp,log,tanh. No imports, loops, object attributes, persistent globals or tools. Use layers=[].
"""


def obj(fields):
    return {"type": "object", "properties": fields, "required": list(fields),
            "additionalProperties": False}


def arr(items):
    return {"type": "array", "items": items}


NUM = {"type": "number"}
SCHEMA = obj({
    "op": {"type": "string", "enum": ["drive", "install", "run", "reset", "finish"]},
    "note": {"type": "string"}, "u": arr(NUM), "ticks": {"type": "integer"},
    "layers": arr(obj({"rows": arr(obj({"terms": arr(obj({"feature": {"type": "integer"}, "coef": NUM})),
                                      "bias": NUM}))})),
    "source": {"type": "string"},
})


def command_of(reply, arm):
    op = reply["op"]
    if op == "install":
        return {"op": op, "spec": {"layers": reply["layers"]} if arm == "weights" else reply["source"]}
    keys = {"drive": ("u", "ticks"), "run": ("ticks",)}.get(op, ())
    return {"op": op, **{key: reply[key] for key in keys}}


def compact(value):
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def packet(result):
    data = {k: result[k] for k in (
        "ok", "error", "state", "scene", "meters", "attempt", "policy_version", "success",
        "walk", "op_summary", "policy_frozen", "dt",
    ) if k in result}
    if "state" in data:
        data["state"] = {k: v for k, v in data["state"].items() if k != "pose"}
    samples = result.get("samples", [])
    if samples:
        end = len(samples) - 1
        indexes = sorted({0, end, max(0,end-1), max(0,end-2), *(i * end // 5 for i in range(1,5))})
        sample_keys = ("tick", "t", "x", "z", "pitch", "vx", "q", "dq", "contact", "u")
        data["samples"] = [{k: samples[i][k] for k in sample_keys if k in samples[i]} for i in indexes]
    return compact(data)


def save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def send(sim, payload):
    sim.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
    sim.stdin.flush()


def receive(sim):
    line = sim.stdout.readline()
    if not line:
        raise RuntimeError(f"simulator did not return JSON (exit={sim.poll()})")
    return json.loads(line)


def simulator_command(args):
    flags = ["--arm", args.arm, "--scene", "train", "--seed", str(args.seed)]
    if not args.ssh_host:
        return [args.python, "-B", "-u", str(Path(__file__).with_name("walk_server.py").resolve()), *flags]
    remote = [args.python, "-B", "-u", args.remote_server, *flags]
    if args.remote_pythonpath:
        remote = ["env", f"PYTHONPATH={args.remote_pythonpath}", *remote]
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "--", args.ssh_host, shlex.join(remote)]


def train(args, brain):
    path = args.out / "train"
    path.mkdir()
    views = path / "views"
    views.mkdir()
    save(path / "protocol.json", {
        "phase": "train", "arm": args.arm, "seed": args.seed, "model": MODEL, "max_decisions": DECISIONS,
        "context": "retained_through_resets", "start_utc": datetime.now(timezone.utc).isoformat(),
    })
    with subprocess.Popen(
        simulator_command(args), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
    ) as sim:
        state = receive(sim)
        save(path / "initial.json", state)
        print(f"[{args.arm}/train] start", flush=True)
        installed = None
        stop = "decision_budget"
        op = None
        with (path / "events.jsonl").open("x", encoding="utf-8") as log:
            for i in range(1, DECISIONS + 1):
                img = views / f"{i:02}.png"
                render_view(state, img)
                info = packet(state)
                lead = BASE + (WEIGHTS if args.arm == "weights" else CODE) if i == 1 else (
                    f"Decision {i}/{DECISIONS}. Here is the result of your last action."
                )
                prompt = lead + "\nSimulator observation: " + json.dumps(info, ensure_ascii=False) + "\nChoose one next action."
                save(views / f"{i:02}.json", {"prompt": prompt, "observation": info})
                t0 = time.monotonic()
                deliberate = i == 1 or state["policy_version"] == 0 or op == "reset" or (
                    op == "run" and bool(state["walk"]["fall_reason"])
                )
                effort = "medium" if deliberate else "low"
                result = brain.run([TextInput(prompt), LocalImageInput(str(img.resolve()))],
                                   effort=effort, output_schema=SCHEMA)
                kinds = [getattr(getattr(it, "root", it), "type", type(it).__name__) for it in result.items]
                if set(kinds) - PASSIVE_ITEMS or not result.final_response:
                    raise RuntimeError(f"unexpected tool or missing response at decision {i}: {kinds}")
                reply = json.loads(result.final_response)
                command = command_of(reply, args.arm)
                op = command["op"]
                send(sim, command)
                state = receive(sim)
                row = {"turn": i, "note": reply["note"], "request": command, "result": state,
                       "sdk": {"model": MODEL, "effort": effort, "items": kinds,
                               "seconds": round(time.monotonic() - t0, 2),
                               "usage": result.usage.model_dump() if result.usage else None}}
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                if op == "install" and state["ok"]:
                    installed = {"version": state["policy_version"], "turn": i, "spec": command["spec"]}
                    save(path / f"policy-v{installed['version']:02}.json", installed)
                walk = state["walk"]
                print(f"[{args.arm}/train {i:02}] {op} ok={state['ok']} v={state['policy_version']} "
                      f"x={walk['distance_m']:+.2f}m steps={walk['alternating_steps']} "
                      f"fall={walk['fall_reason'] or '-'} success={state['success']} "
                      f"{row['sdk']['seconds']:.1f}s {reply['note'][:200]}", flush=True)
                if state.get("finished"):
                    stop = "agent_finished"
                    break
                if state["meters"]["moves_left"] == 0 or state["meters"]["ticks_left"] == 0:
                    stop = "motion_budget"
                    break
        save(path / "final.json", {"state": state, "decisions": i, "stop": stop})
        render_view(state, path / "final.png")
        if installed is not None:
            save(path / "motor_artifact.json", installed)
        sim.stdin.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("weights", "code"), required=True)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ssh-host")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--remote-server")
    parser.add_argument("--remote-pythonpath")
    args = parser.parse_args()
    if args.ssh_host and not args.remote_server:
        parser.error("--ssh-host requires --remote-server")
    args.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="astra-walk-") as tmp, Codex(CodexConfig(
        codex_bin=os.environ.get("CODEX_BIN") or None, cwd=tmp,
        config_overrides=("features.shell_tool=false", "features.multi_agent=false",
                          "features.apps=false", "features.plugins=false", 'web_search="disabled"', "mcp_servers={}"),
    )) as codex:
        brain = codex.thread_start(
            cwd=tmp, model=MODEL, sandbox=Sandbox.read_only, ephemeral=True, approval_mode=ApprovalMode.deny_all,
            base_instructions="You are a simulated robot brain. Use only supplied observations and the strict JSON interface. No tools.",
        )
        train(args, brain)


if __name__ == "__main__":
    main()
