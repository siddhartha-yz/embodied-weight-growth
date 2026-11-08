"""Two Astra agents learn to operate the same pen arm through Codex SDK."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

from openai_codex import ApprovalMode, Codex, CodexConfig, LocalImageInput, Sandbox, TextInput
from PIL import Image, ImageDraw

MODEL = "gpt-6-astra"
PASSIVE_ITEMS = {"userMessage", "reasoning", "agentMessage", "plan"}
DECISIONS = 24
FEATURES = (
    "[1,q0,q1,dq0,dq1,x,y,vx,vy,goal_x-x,goal_y-y,"
    "prev_u0,prev_u1,prev2_u0,prev2_u1]"
)

BASE = """You are the brain of a simulated two-joint arm holding a pen. Your goal is to write a recognizable uppercase W on a blank sheet.
Nobody gives you a W reference path or a controller. Work out both the letter and the unfamiliar motor response yourself.
The paper is blank at the start. Use a visible size (roughly several cm across); no exact size or position is required.
The two anonymous input channels accept normalized motor commands in [-0.8,0.8]. A tick is 0.02 s.
The horizontal and vertical coordinates in the attached view increase to the right and up. Use its meter grid.
You can test controls with the pen up, learn a motor skill, then use that skill to draw. Observe after each action and revise as needed.
Plan on only up to 18 motion requests / 1000 physics ticks total; allow time for enough drawing moves. You keep the same conversation.
Goal inputs for your learned motor skill are your own chosen pen coordinates, in meters. The host does not steer to the goal.
A good module keeps observing the current body at every tick. It should respond to a goal you specify, not contain the W or replay a script.
Operations:
- drive: repeat raw u for ticks, pen always up. Use this to discover how your body responds.
- install: replace the skill artifact. No motion.
- use: run YOUR installed artifact at each tick for a single absolute goal. Pen down if pen=true, otherwise move without ink.
  It runs for exactly ticks (1..100), with no automatic target stop. Keep pen=true across moves for one continuous stroke.
- reset: clear ink and return to the initial posture, keeping your skill and the spent budget.
- finish: stop once the actual ink looks like a W or you cannot improve within the budget.
Do not call any Codex tools. Return one JSON action with a short note naming what you observed and what you are trying next.
For unused fields, return empty arrays/strings, ticks=0 and pen=false. You can refer to the earlier observations in this thread.
"""

WEIGHTS = """Your arm writes WEIGHTS, not Python. There is no initialized controller.
An installed module reads these 15 inputs in exactly this order:
{features}
where q is joint angle in rad, dq rad/s, x,y meters, vx,vy m/s; the last four entries are the last two motor commands.
Return layers: [{{"weight": [[...], [...]], "bias": [b0,b1]}}] for a direct linear map 15->2, or
two such layers 15->H->2 with tanh after the first, H between 1 and 16. Outputs are clipped to [-0.8,0.8].
You must choose all numeric weights and biases. In particular you can combine position/velocity feedback and use your probes to reason about signs and cross-coupling.
Use the raw two-channel probes first if the mapping is unclear. A small goal-responsive module is sufficient; no W waypoints in the weights.
Use source="" for every request.
"""

CODE = """Your arm writes CODE. No controller is provided. Install Python source defining exactly:
def policy(obs, goal):
    # Return [motor0,motor1].
obs fields q,dq,tip,tip_v,prev_u,prev2_u are lists of two numbers. Units: rad, rad/s, m, m/s, normalized inputs.
goal is [x,y] in metres. Outputs are clipped to [-0.8,0.8].
Use normal assignments to names, simple arithmetic, comparisons, if/else, indexing, and return a two-number list.
No imports, loops, attribute access, objects, persistent globals or other tools. Functions allowed: abs,min,max,sin,cos,sqrt,atan2,tanh.
Write your own goal-responsive mapping and correct it using observations. No W waypoints in your installed code.
Use layers=[] for every request.
"""


def obj(fields):
    return {"type": "object", "properties": fields, "required": list(fields),
            "additionalProperties": False}


def arr(item):
    return {"type": "array", "items": item}


NUMBERS = arr({"type": "number"})
SCHEMA = obj({
    "op": {"type": "string", "enum": ["drive", "install", "use", "reset", "finish"]},
    "note": {"type": "string"},
    "u": NUMBERS, "goal": NUMBERS, "ticks": {"type": "integer"}, "pen": {"type": "boolean"},
    "layers": arr(obj({"weight": arr(NUMBERS), "bias": NUMBERS})),
    "source": {"type": "string"},
})


def request_of(reply, arm):
    op = reply["op"]
    if op == "install":
        spec = {"layers": reply["layers"]} if arm == "weights" else reply["source"]
        return {"op": op, "spec": spec}
    if op == "drive":
        return {"op": op, "u": reply["u"], "ticks": reply["ticks"]}
    if op == "use":
        return {k: reply[k] for k in ("op", "goal", "ticks", "pen")}
    return {"op": op}


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def receive(sim):
    line = sim.stdout.readline()
    if not line:
        raise RuntimeError(f"simulation ended unexpectedly (exit={sim.poll()})")
    return json.loads(line)


def round_tree(obj):
    if isinstance(obj, float):
        return round(obj, 5)
    if isinstance(obj, dict):
        return {k: round_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_tree(v) for v in obj]
    return obj


def feedback(res):
    data = {k: res[k] for k in (
        "ok", "error", "state", "samples", "attempt", "meters", "policy_version", "pen", "dt", "board",
    ) if k in res}
    trace = res.get("trace", [])
    if trace:
        data["move_trace_sample_m"] = trace[::max(1, len(trace)//8)] + [trace[-1]]
    data["ink_points_per_stroke"] = [len(s) for s in res.get("strokes", [])]
    return json.dumps(round_tree(data), ensure_ascii=False, separators=(",", ":"))


def render_view(res, path):
    """Actual observations only; no target letter or waypoints."""
    canvas = Image.new("RGB", (400, 470), "#fffdf8")
    d = ImageDraw.Draw(canvas)
    board = res.get("board", {"x": [.09, .245], "y": [-.10, .105]})
    x0, x1 = board["x"]
    y0, y1 = board["y"]
    left, top, right, bottom = 48, 38, 360, 444
    def pix(p):
        return (left + (p[0]-x0)/(x1-x0)*(right-left),
                bottom - (p[1]-y0)/(y1-y0)*(bottom-top))
    for i in range(round(100*x0)+1, int(100*x1)+1):
        if i % 2 == 0:
            x, _ = pix((i/100, 0))
            d.line([(x, top), (x, bottom)], fill="#e0e7e9")
            d.text((x-5, bottom+5), str(i), fill="#455766")
    for i in range(round(100*y0), int(100*y1)+1, 2):
        _, y = pix((0, i/100))
        d.line([(left, y), (right, y)], fill="#e0e7e9")
        d.text((left-26, y-6), str(i), fill="#455766")
    d.rectangle((left, top, right, bottom), outline="#9aaab0")
    d.text((left, 12), "Paper (axes: centimetres); red dot: pen tip", fill="#1a3242")
    for stroke in res.get("strokes", []):
        if len(stroke)>1:
            d.line([pix(p) for p in stroke], fill="#153d66", width=4, joint="curve")
    trace = res.get("trace", [])
    if not res.get("pen", False) and len(trace)>1:
        d.line([pix(p) for p in trace], fill="#95afb0", width=2)
    x,y = pix(res["state"]["tip"])
    d.ellipse((x-4,y-4,x+4,y+4),fill="#c24e3c")
    canvas.save(path)


def session(args, path, sim, brain, scratch):
    state = receive(sim)
    initial = state
    save(path/"initial.json", state)
    instruction = BASE + (WEIGHTS.format(features=FEATURES) if args.arm=="weights" else CODE)
    with (path/"events.jsonl").open("x", encoding="utf-8") as log:
        for i in range(1, DECISIONS+1):
            view = scratch/f"view-{i:02d}.png"
            render_view(state, view)
            info = ("\nInitial state: " if i==1 else "Latest simulator feedback: ") + feedback(state)
            if i==1:
                prompt = instruction + info
            else:
                prompt = f"Decision {i}/{DECISIONS}. " + info + "\nChoose exactly one next action."
            effort = "medium" if state.get("policy_version", 0)==0 else "low"
            t0 = time.monotonic()
            result = brain.run([TextInput(prompt), LocalImageInput(str(view))],
                               effort=effort, output_schema=SCHEMA)
            items = [getattr(it, "root", it) for it in result.items]
            kinds = [getattr(it, "type", type(it).__name__) for it in items]
            if set(kinds)-PASSIVE_ITEMS or not result.final_response:
                raise RuntimeError(f"unexpected tool or missing response at decision {i}: {kinds}")
            reply = json.loads(result.final_response)
            command = request_of(reply, args.arm)
            sim.stdin.write(json.dumps(command, allow_nan=False)+"\n")
            sim.stdin.flush()
            state = receive(sim)
            row = {"turn":i, "note":reply["note"], "request":command, "result":state,
                   "sdk":{"model":MODEL,"effort":effort,"items":kinds,
                          "seconds":round(time.monotonic()-t0,2),
                          "usage":result.usage.model_dump() if result.usage else None}}
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            log.flush()
            print(f"[{args.arm} {i:02}] {command['op']:<8} "
                  f"ok={state['ok']} attempt={state.get('attempt')} "
                  f"ink={sum(map(len,state.get('strokes',[])))} "
                  f"{row['sdk']['seconds']:.1f}s: {reply['note'][:160]}",
                  flush=True)
            if state.get("finished"):
                return initial, state, i, "agent_finished"
            if state.get("meters",{}).get("moves_left") == 0 or state.get("meters",{}).get("ticks_left") == 0:
                return initial, state, i, "motion_budget"
    return initial, state, DECISIONS, "decision_budget"


def simulator_command(args):
    flags = ["--arm", args.arm, "--condition", args.condition, "--seed", str(args.seed)]
    if args.ssh_host:
        remote = shlex.join([args.remote_python, "-B", "-u", args.remote_server, *flags])
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                "--", args.ssh_host, remote]
    server = Path(__file__).resolve().with_name("w_server.py")
    return [sys.executable, "-B", "-u", str(server), *flags]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=("weights","code"))
    p.add_argument("--condition", default="linear", choices=("linear","backlash"))
    p.add_argument("--seed", type=int, default=271)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ssh-host", help="run the simulator on an existing SSH host instead of locally")
    p.add_argument("--remote-python", help="absolute path to a Python interpreter on the SSH host")
    p.add_argument("--remote-server", help="absolute path to this repository's w_server.py on the SSH host")
    args=p.parse_args()
    if args.ssh_host and not all(x and x.startswith("/") for x in
                                 (args.remote_python, args.remote_server)):
        p.error("--ssh-host requires absolute --remote-python and --remote-server paths")
    if not args.ssh_host and (args.remote_python or args.remote_server):
        p.error("--remote-python and --remote-server require --ssh-host")
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out/"protocol.json", {"arm":args.arm, "condition":args.condition,
                                   "seed":args.seed, "out":str(args.out),
                                   "model":MODEL, "max_decisions":DECISIONS})
    with tempfile.TemporaryDirectory(prefix="astra-w-") as tmp, subprocess.Popen(
        simulator_command(args),
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="utf-8",
    ) as sim:
        config=CodexConfig(
            codex_bin=os.environ.get("CODEX_BIN") or None, cwd=tmp,
            config_overrides=("features.shell_tool=false","features.multi_agent=false",
                              "features.apps=false","features.plugins=false",
                              'web_search="disabled"',"mcp_servers={}"),
        )
        with Codex(config) as codex:
            brain=codex.thread_start(
                cwd=tmp, model=MODEL, sandbox=Sandbox.read_only, ephemeral=True,
                approval_mode=ApprovalMode.deny_all,
                base_instructions="You are a simulated robot brain. Use only supplied observations and the strict JSON interface. No tools.",
            )
            initial,state,n,stop=session(args,args.out,sim,brain,Path(tmp))
        save(args.out/"final.json", {"state":state,"decisions":n,"stop":stop})
        render_view(state,args.out/"final.png")
        sim.stdin.close()


if __name__ == "__main__":
    main()
