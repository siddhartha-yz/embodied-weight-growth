"""Run Astra in a closed loop with a MuJoCo arm, then reuse the saved controller."""
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

from slot_view import render_view

MODEL = "gpt-6-astra"
DECISIONS = 45
PASSIVE_ITEMS = {"userMessage", "reasoning", "agentMessage", "plan"}
BASE = """You are the brain of a simulated two-joint planar robot arm with a round pusher.
Goal: rotate and push the free rectangular block fully into the narrow slot, with its long side along the slot and no motion at the end.
The target and rails are visible in the image and geometry. No route, push primitive, or controller is provided.
Only two anonymous normalized motor inputs u in [-0.8,0.8] are available. A tick is 0.02 seconds. The motors and body do not change.
The JSON shows coordinates in meters (x right, y up), angles in radians, velocity per second.
block_angle is [cos(yaw), sin(yaw)]; contact is [pusher touching block, block touching rail].
The purple rectangle is the actual block and the orange dot is your pusher; rails and target are drawn at their true locations. Consult both the image and numbers.
Make one decision at a time. Every action returns actual feedback; use it to decide the next action, not a prewritten path.
An installed motor module receives live observations and a tip goal in meters at every tick; the host does not steer or stop at the goal.
Do not encode a route or replay past actions in the module. It should be a reusable goal-responsive motor skill.
Operations:
- drive: send two raw motor inputs for ticks.
- install: install or revise YOUR motor module. No motion.
- use: run YOUR installed module for ticks with your chosen [goal_x,goal_y] tip goal.
- reset: restore the initial robot and block pose, keeping your module and spent budget.
- finish: end the phase. The snapshot's success flag is the geometric/settled test; use feedback before declaring success.
Motion budget per phase: 36 commands, 2400 total ticks, at most 100 per command. Decision budget: 45.
Watch whether each push actually translates or rotates the block, or stalls against a rail. Correct a bad approach, and revise the controller if its motor behavior is the problem.
Do not call any Codex tools. Return exactly one JSON decision. In 'note', first name the specific feedback that matters, then why you chose this next action. Keep it brief.
Unused fields: u/goal/layers=[], source="", ticks=0.
"""
WEIGHTS = """You must directly supply the numeric weights, not code. The installed module reads exactly these 23 inputs:
[1,q0,q1,dq0,dq1,tip_x,tip_y,tip_vx,tip_vy,goal_x-tip_x,goal_y-tip_y,prev_u0,prev_u1,prev2_u0,prev2_u1,block_x,block_y,block_cos,block_sin,block_vx,block_vy,tip_block_contact,block_wall_contact]
Supply layers as one 23->2 linear layer or 23->H->2 (tanh after hidden layer, H=1..16). Each layer is {"weight":[rows],"bias":[...]}. Motor outputs are clipped to [-0.8,0.8]. Use source="".
"""
CODE = """You must supply restricted Python source, defining exactly def policy(obs, goal): returning [motor0,motor1].
All obs fields are 2-vectors: q,dq,tip,tip_v,prev_u,prev2_u,block,block_angle,block_v,contact. goal=[x,y] in meters. Motor outputs are clipped to [-0.8,0.8].
Allowed: ordinary assignments, arithmetic, comparisons, if/else, indexing, return, and abs,min,max,sin,cos,sqrt,atan2,tanh. No imports, loops, attribute access, globals, or tool calls. Use layers=[].
"""
TRANSFER = """Transfer phase: same body, same task and same budget, but the block starts at another yaw.
You keep the conversation and your most recently accepted motor module is installed and frozen.
Choose the approach and tip goals from the new observations; do not copy the old trajectory blindly.
Only use, reset, or finish are available. No raw drive or controller revision in this phase.
"""


def obj(fields):
    return {"type": "object", "properties": fields, "required": list(fields),
            "additionalProperties": False}


def arr(items):
    return {"type": "array", "items": items}


def schema(phase):
    nums = arr({"type": "number"})
    return obj({
        "op": {"type": "string", "enum": ["use", "reset", "finish"] if phase == "transfer"
               else ["drive", "install", "use", "reset", "finish"]},
        "note": {"type": "string"}, "u": nums, "goal": nums, "ticks": {"type": "integer"},
        "layers": arr(obj({"weight": arr(nums), "bias": nums})), "source": {"type": "string"},
    })


def command_of(reply, arm):
    op = reply["op"]
    if op == "install":
        return {"op": op, "spec": {"layers": reply["layers"]} if arm == "weights" else reply["source"]}
    keys = {"drive": ("u", "ticks"), "use": ("goal", "ticks")}.get(op, ())
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
        "ok", "error", "state", "scene", "meters", "attempt", "policy_version", "success", "finished",
        "policy_frozen", "contact_summary", "dt",
    ) if k in result}
    samples = result.get("samples", [])
    if samples:
        ix = sorted({0, (len(samples)-1)//3, 2*(len(samples)-1)//3, len(samples)-1})
        sample_keys = ("tick", "tip", "block", "block_angle", "contact", "u")
        data["samples"] = [{k: samples[i][k] for k in sample_keys if k in samples[i]} for i in ix]
    return compact(data)


def save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def send(sim, payload):
    sim.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
    sim.stdin.flush()


def receive(sim):
    line = sim.stdout.readline()
    if not line:
        raise RuntimeError(f"simulator did not return a JSON snapshot (exit={sim.poll()})")
    return json.loads(line)


def simulator_command(args, phase, frozen=False):
    flags = ["--arm", args.arm, "--scene", phase, "--seed", str(args.seed)]
    if frozen:
        flags.append("--frozen-spec-stdin")
    if not args.ssh_host:
        return [args.python, "-B", "-u", str(Path(__file__).with_name("slot_server.py").resolve()), *flags]
    remote = [args.python, "-B", "-u", args.remote_server, *flags]
    if args.remote_pythonpath:
        remote = ["env", f"PYTHONPATH={args.remote_pythonpath}", *remote]
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "--", args.ssh_host, shlex.join(remote)]


def run_phase(args, phase, brain, installed=None):
    path = args.out / phase
    path.mkdir(parents=True)
    views = path / "views"
    views.mkdir()
    frozen = installed is not None
    protocol = {"phase": phase, "arm": args.arm, "seed": args.seed, "model": MODEL,
                "max_decisions": DECISIONS, "context": "retained" if frozen else "new",
                "source_policy_version": installed["version"] if frozen else None,
                "start_utc": datetime.now(timezone.utc).isoformat()}
    save(path / "protocol.json", protocol)
    with subprocess.Popen(
        simulator_command(args, phase, frozen), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, encoding="utf-8",
    ) as sim:
        if frozen:
            send(sim, installed["spec"])
        state = receive(sim)
        save(path / "initial.json", state)
        print(f"[{args.arm}/{phase}] start", flush=True)
        stop = "decision_budget"
        action_schema = schema(phase)
        with (path / "events.jsonl").open("x", encoding="utf-8") as log:
            for i in range(1, DECISIONS+1):
                img = views / f"{i:02}.png"
                render_view(state, img)
                info = packet(state)
                if i == 1:
                    lead = TRANSFER if frozen else BASE + (WEIGHTS if args.arm == "weights" else CODE)
                else:
                    lead = f"Decision {i}/{DECISIONS}. Here is the result of your last action."
                prompt = lead + "\nSimulator observation: " + json.dumps(info, ensure_ascii=False) + "\nChoose one next action."
                save(views / f"{i:02}.json", {"prompt": prompt, "observation": info})
                t0 = time.monotonic()
                # Give both representations the same reasoning budget at the same stages.
                effort = "medium" if i == 1 or state.get("policy_version", 0) == 0 or state.get("contact_summary", {}).get("stall") else "low"
                result = brain.run([TextInput(prompt), LocalImageInput(str(img.resolve()))],
                                   effort=effort, output_schema=action_schema)
                items = [getattr(it, "root", it) for it in result.items]
                kinds = [getattr(it, "type", type(it).__name__) for it in items]
                if set(kinds) - PASSIVE_ITEMS or not result.final_response:
                    raise RuntimeError(f"unexpected tool or missing response at decision {i}: {kinds}")
                reply = json.loads(result.final_response)
                command = command_of(reply, args.arm)
                send(sim, command)
                state = receive(sim)
                row = {"turn": i, "note": reply["note"], "request": command, "result": state,
                       "sdk": {"model": MODEL, "effort": effort, "items": kinds,
                               "seconds": round(time.monotonic() - t0, 2),
                               "usage": result.usage.model_dump() if result.usage else None}}
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                if command["op"] == "install" and state["ok"]:
                    installed = {"version": state["policy_version"], "turn": i, "spec": command["spec"]}
                    save(path / f"policy-v{installed['version']:02}.json", installed)
                print(f"[{args.arm}/{phase} {i:02}] {command['op']} "
                      f"ok={state['ok']} success={state['success']} v={state['policy_version']} "
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
    return installed


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
    with tempfile.TemporaryDirectory(prefix="astra-slot-") as tmp, Codex(CodexConfig(
        codex_bin=os.environ.get("CODEX_BIN") or None, cwd=tmp,
        config_overrides=("features.shell_tool=false", "features.multi_agent=false",
                          "features.apps=false", "features.plugins=false", 'web_search="disabled"', "mcp_servers={}"),
    )) as codex:
        brain = codex.thread_start(
            cwd=tmp, model=MODEL, sandbox=Sandbox.read_only, ephemeral=True, approval_mode=ApprovalMode.deny_all,
            base_instructions="You are a simulated robot brain. Use only supplied observations and the strict JSON interface. No tools.",
        )
        installed = run_phase(args, "train", brain)
        if installed is not None:
            run_phase(args, "transfer", brain, installed)
        else:
            save(args.out / "transfer_skipped.json", {"reason": "no successfully installed training artifact"})


if __name__ == "__main__":
    main()
