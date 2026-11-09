"""Weight-only Codex SDK transport for the author H1 workspace session."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import time
import traceback
import urllib.request

import numpy as np

from orchestrator.session_client import request


def jwrite(path, obj):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def public_state(folder):
    data = json.loads((folder / "observation.json").read_text())
    with np.load(folder / "arrays.npz", allow_pickle=False) as arrays:
        def decode(item):
            if isinstance(item, dict):
                if "npz_key" in item:
                    a = arrays[item["npz_key"]]
                    return a.tolist() if a.size < 250 else item
                return {k: decode(v) for k, v in item.items()}
            if isinstance(item, list):
                return [decode(v) for v in item]
            return item
        return {k: decode(data[k]) for k in ("instruction", "state", "action", "additional_info") if k in data}


def main(args):
    if args.phase != "playground":
        raise ValueError("This from-zero pilot runs Playground only")
    root = args.episode
    workspace = root / "workspace"
    audit = root / "sdk-turns.jsonl"
    token = args.token_file.read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def native(op, **kw):
        return request({"op": op, **kw}, socket_path=str(args.socket))

    def call_brain(prompt, images):
        data = json.dumps({"thread": args.thread, "phase": args.phase, "prompt": prompt, "images": images}).encode()
        req = urllib.request.Request(args.brain, data=data, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, method="POST")
        with opener.open(req, timeout=330) as response:
            reply = json.load(response)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "SDK relay rejected turn"))
        return reply["action"]

    initial = (
        "You are the fresh weight-space policy designer for official HumanoidBench H1, "
        "not a coding agent. Your weights run at 50 Hz. "
        "You will see the same type of physical state, camera, and prior trial feedback each turn. "
        "No controller is installed initially. A rejected submission may consume a slot but no physics. "
        "Numeric rewards are private; the goal is forward walking and stable posture. "
        "The public state contains qpos (26 values), qvel (25), the 19 native actuator channels "
        "in output order, and pelvis displacement from reset. The initial root quaternion is wxyz. "
        "qvel starts with 3 translational and 3 angular velocities, followed by joint velocities. "
        "Each actuator channel gives its joint name, indices and physical radians for normalized -1 and 1. "
        "External camera label cam_head is legacy; it is not head-mounted.\n\n"
        + (workspace / "instruction.md").read_text()
        + "\n\nWEIGHT FORMAT:\n" + (workspace / "policy_guide.md").read_text()
    )
    recent = None
    for turn in range(1, args.turns + 1):
        status = native("status")
        if status["closed"]:
            return
        folder = workspace / status["observation"]
        state = public_state(folder)
        prompt = (initial + "\n\n" if turn == 1 else "") + (
            f"Phase: {args.phase}. SDK turn {turn}/{args.turns}.\n"
            f"Current session budgets and success flag: {json.dumps(status)}\n"
            f"Current public observation: {json.dumps(state)}\n"
        )
        if recent is not None:
            prompt += "Previous execution result:\n" + json.dumps(recent, ensure_ascii=False)
        if turn == args.turns or status.get("success") or status["executions_remaining"] == 0:
            prompt += "\nReturn finish and summarize which numeric policy worked best and what it actually did."
        image = folder / "current_cam_head.png"
        images = [{"name": "cam_head", "base64": base64.b64encode(image.read_bytes()).decode("ascii")}]
        started = time.monotonic()
        action = call_brain(prompt, images)
        summary = {"turn": turn, "seconds": round(time.monotonic() - started, 2), "op": action["op"],
                   "note": action["note"], "steps": action["steps"], "reset_before": action["reset_before"],
                   "policy_chars": len(action["policy"])}
        jwrite(audit, summary)
        if action["op"] == "finish" or turn == args.turns or status["executions_remaining"] == 0:
            (workspace / "lessons" / "wsp-result.txt").write_text(action["note"] + "\n")
            native("finish")
            return
        plan = json.dumps({k: action[k] for k in ("op", "steps", "policy", "reset_before")},
                          ensure_ascii=True, separators=(",", ":"))
        outcome = native("exec", files=[{"path": "submission/weights.json", "code": plan}])
        recent = {k: outcome.get(k) for k in ("ok", "success", "error", "stdout", "stderr",
                                              "native_steps_used", "native_steps_remaining", "observation")}
        raw_feedback = recent["stdout"]
        if raw_feedback:
            feedback = json.loads(raw_feedback)
            trace = feedback["trace"]
            if len(trace) > 12:
                feedback["trace"] = trace[::(len(trace) + 9) // 10] + [trace[-1]]
            recent["stdout"] = feedback
            if outcome["ok"] and action["op"] == "install":
                (workspace / "skills" / "latest-weights.json").write_text(action["policy"] + "\n")
        jwrite(audit, {"turn": turn, "feedback": recent})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--phase", choices=("playground", "test"), required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=25)
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        (args.episode / "sdk-error.log").write_text(traceback.format_exc())
        raise
