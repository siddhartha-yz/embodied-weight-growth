"""Run a two-turn Astra experiment using the official Python Codex SDK.

MuJoCo runs behind a private, finite-budget SSH process. The SDK receives only
probe observations; its only accepted outputs are probe commands and weights.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile

from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox


MODEL = "gpt-6-astra"
PASSIVE_ITEMS = {"userMessage", "reasoning", "agentMessage", "plan"}
INIT = [
    {"start": [0.45, -0.85], "segments": [[0.5, 0, 5], [-0.5, 0, 7], [0, 0, 5], [0.5, 0, 7]]},
    {"start": [0.45, -0.85], "segments": [[0, 0.5, 5], [0, -0.5, 7], [0, 0, 5], [0, 0.5, 7]]},
]
NUMBER_VEC = {"type": "array", "items": {"type": "number"}}


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def array(item):
    return {"type": "array", "items": item}


PROBE_SCHEMA = obj({"hypothesis": {"type": "string"},
                    "probes": array(obj({"start": NUMBER_VEC, "segments": array(NUMBER_VEC)}))})
PATCH_SCHEMA = obj({
    "diagnosis": {"type": "string"},
    "adapter": obj({"mix": array(NUMBER_VEC), "bias": NUMBER_VEC,
                    "state": array(NUMBER_VEC), "history": array(NUMBER_VEC),
                    "history2": array(NUMBER_VEC), "deadzone": NUMBER_VEC}),
})
DIMS = {"mix": (2, 2), "bias": (2,), "state": (2, 4),
        "history": (2, 2), "history2": (2, 2), "deadzone": (2,)}


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def episode_text(episodes, first=0):
    parts = []
    for i, ep in enumerate(episodes, first):
        # A row describes (state before action, action, resulting acceleration).
        obs = ep["initial"]
        rows = []
        for step in ep["steps"]:
            accel = [(x - y) / .02 for x, y in zip(step["dq"], obs["dq"])]
            rows.append(" ".join(f"{x:.3f}" for x in [*obs["q"], *obs["dq"], *step["u"], *accel]))
            obs = step
        parts.append(f"Probe {i + 1}:\n" + "\n".join(rows))
    return "\n".join(parts)


def turn(thread, prompt, schema, output, stage):
    result = thread.run(prompt, effort="medium", output_schema=schema)
    # openai-codex models item variants as a Pydantic RootModel.
    variants = [getattr(item, "root", item) for item in result.items]
    kinds = [getattr(item, "type", type(item).__name__) for item in variants]
    audit = {
        "stage": stage, "model_requested": MODEL,
        "status": str(result.status), "error": str(result.error) if result.error else None,
        "item_types": kinds, "response": result.final_response,
        "usage": result.usage.model_dump() if result.usage else None,
        "duration_ms": result.duration_ms,
    }
    save(output / (stage + ".json"), audit)
    if not result.final_response:
        raise RuntimeError(f"Astra produced no response: {audit['status']} {audit['error']}")
    if any(k not in PASSIVE_ITEMS for k in kinds):
        raise RuntimeError(f"unexpected SDK tool item(s) in {stage}: {kinds}")
    return json.loads(result.final_response)


def request_probe(sim, payload):
    sim.stdin.write(json.dumps(payload) + "\n")
    sim.stdin.flush()
    line = sim.stdout.readline()
    if not line:
        raise RuntimeError("lab simulator closed unexpectedly")
    reply = json.loads(line)
    if "error" in reply:
        raise ValueError(f"lab simulator rejected a probe: {reply['error']}")
    return reply["episode"]


def vector(values, count):
    if not isinstance(values, list) or len(values) != count or any(
        type(x) not in {float, int} or not math.isfinite(x) for x in values
    ):
        raise ValueError(f"expected {count} finite values")


def validate_patch(adapter):
    if set(adapter) != set(DIMS):
        raise ValueError("adapter fields differ from specification")
    for name, shape in DIMS.items():
        value = adapter[name]
        if len(shape) == 1:
            vector(value, shape[0])
        else:
            if not isinstance(value, list) or len(value) != shape[0]:
                raise ValueError(f"wrong rows for {name}")
            for row in value:
                vector(row, shape[1])
    if not all(0 <= x <= 0.8 for x in adapter["deadzone"]):
        raise ValueError("deadzone outside allowed range")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=("linear", "backlash"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume-probe", type=Path, help="reuse a completed probe design with identical initial measurements")
    p.add_argument("--ssh-host")
    p.add_argument("--remote-python", default="python3")
    p.add_argument("--remote-server")
    args = p.parse_args()
    if bool(args.ssh_host) != bool(args.remote_server):
        p.error("--ssh-host and --remote-server must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    probe_args = ["--condition", args.condition, "--seed", str(args.seed)]
    if args.ssh_host:
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "--",
                   args.ssh_host, shlex.join([args.remote_python, "-B", "-u", args.remote_server, *probe_args])]
    else:
        command = [sys.executable, "-B", "-u", str(Path(__file__).with_name("probe_server.py")), *probe_args]
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
    ) as sim, tempfile.TemporaryDirectory(prefix="astra-body-") as isolation:
        hello = json.loads(sim.stdout.readline())
        if hello["dt"] != .02:
            raise ValueError("probe cadence changed")
        records = {"episodes": [request_probe(sim, x) for x in INIT]}
        save(args.output / "records.json", records)
        conf = CodexConfig(
            codex_bin=os.environ.get("CODEX_BIN") or None,
            cwd=isolation,
            config_overrides=(
                "features.shell_tool=false", "features.multi_agent=false",
                "features.apps=false", "features.plugins=false",
                "web_search=\"disabled\"", "mcp_servers={}",
            ),
        )
        with Codex(conf) as codex:
            brain = codex.thread_start(
                cwd=isolation, model=MODEL, sandbox=Sandbox.read_only,
                approval_mode=ApprovalMode.deny_all, ephemeral=True,
                base_instructions="You are the high-level brain of a simulated robot with a newly attached body. "
                "Use only the supplied observations. Return the specified JSON; do not call tools. "
                "The host runs any approved motor probes and returns measurements.",
            )
            probe_prompt = """You are calibrating a horizontal two-joint arm with two anonymous normalized motor inputs.
You get joint encoders q (rad), dq (rad/s) and tip position (metres), not the motor-to-joint mapping.
Later, a fixed local trajectory tracker will use YOUR forward model of joint acceleration for tasks you never see.
No tasks are given during exploration. A control step is 0.02 s. A reset returns to the requested joint angles and
zero velocity; it preserves the body but clears transient motion history. Two initial test pulses are below.
Columns: q0 q1 dq0 dq1 u0 u1 qdd0 qdd1. qdd=(dq_next-dq)/0.02.
Design 2–4 informative follow-up probe episodes, not task attempts. Each has start [q0,q1] with |q|<=1.65,
and segments of [u0,u1,ticks]; |u|<=0.8, ticks integer 1..16, <=48 total ticks per episode.
Your end product after probes will be weights for a small forward dynamics module. Aim to identify both channels'
joint effects, history-dependence if any, and passive velocity response. Provide a concise, testable hypothesis.

""" + episode_text(records["episodes"])
            handoff = ""
            if args.resume_probe:
                previous = json.loads((args.resume_probe / "probe_request.json").read_text())
                previous_records = json.loads((args.resume_probe / "records.json").read_text())
                if (previous["model_requested"] != MODEL or previous["status"] != "TurnStatus.completed"
                    or not previous["response"] or set(previous["item_types"]) - PASSIVE_ITEMS
                    or previous_records["episodes"][:len(INIT)] != records["episodes"]):
                    raise ValueError("prior probe design does not match the current body and audit")
                save(args.output / "probe_request.json", previous)
                plan = json.loads(previous["response"])
                handoff = probe_prompt + "\nYour completed probe design: " + json.dumps(plan) + "\n"
            else:
                print(f"start {MODEL} probe design on lab server", flush=True)
                plan = turn(brain, probe_prompt, PROBE_SCHEMA, args.output, "probe_request")
            if not 2 <= len(plan["probes"]) <= 4:
                raise ValueError("Astra must request 2–4 episodes")
            for req in plan["probes"]:
                # JSON number decoding may yield 4.0; preserve strict integer ticks on host.
                for seg in req["segments"]:
                    if type(seg[2]) is float and seg[2].is_integer():
                        seg[2] = int(seg[2])
                records["episodes"].append(request_probe(sim, req))
                save(args.output / "records.json", records)
            print(f"completed {len(records['episodes'])} probes; requesting direct weight write", flush=True)
            out = turn(brain, handoff + """Your follow-up observations appear below. Columns: q0 q1 dq0 dq1 u0 u1 qdd0 qdd1.
Produce the final numeric weights for a shared forward dynamics model. Do not return code or a task-specific controller.
Exact equation and shapes:
qdd = mix[2,2] @ soft_threshold(u, deadzone[2]) + bias[2]
      + state[2,4] @ [q0,q1,dq0,dq1] + history[2,2] @ u_previous
      + history2[2,2] @ u_two_steps_ago
where soft_threshold(u,d)=sign(u)*max(abs(u)-d,0) elementwise; previous controls zero at reset.
The tracker numerically inverts mix and caps controls to [-1,1]. mix must be invertible and not ill-conditioned.
Estimate sensible control/damping weights from the traces; use 0 for terms with no evidence. Do not claim
that one/two-step history exactly represents arbitrarily long transmission memory. Include a concise diagnosis.

""" + episode_text(records["episodes"][2:], 2), PATCH_SCHEMA, args.output, "weight_response")
        validate_patch(out["adapter"])
        save(args.output / "adapter.json", out["adapter"])
        sim.stdin.close()
    print(f"adapter frozen: {args.output / 'adapter.json'}", flush=True)


if __name__ == "__main__":
    main()
