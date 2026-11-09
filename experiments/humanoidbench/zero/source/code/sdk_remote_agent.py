"""Narrow Codex SDK transport for an upstream EmbodiedEvo workspace session."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
import traceback
import urllib.request

import numpy as np

from orchestrator.session_client import request, source_snapshot


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


def safe_path(workspace, value, *, include=False):
    name = PurePosixPath(value)
    if (name.is_absolute() or len(name.parts) < 2 or len(name.parts) > 5
            or ".." in name.parts or any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", p) for p in name.parts)
            or name.parts[0] not in ({"skills"} if include else {"skills", "lessons"})
            or name.suffix not in ({".py"} if include else {".py", ".md", ".txt"})):
        raise ValueError(f"invalid knowledge path: {value}")
    path = workspace / name
    path.resolve().relative_to(workspace / name.parts[0])
    if path.is_symlink():
        raise ValueError("symlink is not a knowledge file")
    return path


def bundle(workspace):
    parts = []
    for area in ("skills", "lessons"):
        for path in sorted((workspace / area).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md", ".txt"}:
                parts.append(f"--- {path.relative_to(workspace)} ---\n{path.read_text(encoding='utf-8')}")
    return "\n".join(parts)


def main(args):
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
        with opener.open(req, timeout=330) as result:
            reply = json.load(result)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "SDK relay rejected turn"))
        return reply["action"]

    first = True
    recent = None
    status = native("status")
    initial = (
        "You are the Codex SDK coding agent for an upstream EmbodiedEvo session. The host follows the "
        "author protocol, but the workspace bridge submits your JSON actions instead of giving you a shell. "
        "You can see the task, current camera frames and public robot state; there are no private object poses. "
        "Use action op=exec for the next Python segment, with files for new reusable skills or lessons and "
        "include for ordered skills/*.py to prepend. Your Python namespace persists across submissions. "
        "Use op=reset only in Playground; op=finish may save final skills/lessons in Playground. "
        "Do not submit many blind steps: start with probing and feedback, check for motion, and use bounded control loops. "
        "When the native task succeeds, or when you are blocked/out of budget, finish; in Playground save tested "
        "reusable code and evidence-based lessons first, without hardcoded target positions or a replay of this scene. "
        "In Test, use frozen knowledge where it helps and name what you reused; do not modify or regenerate the frozen files. "
        "A single external tracking camera is provided each turn (legacy label cam_head; it is not head-mounted). "
        "Use only public observations and documented primitives.\n\n"
        + (workspace / "instruction.md").read_text()
        + "\n\nPRIMITIVE MANUAL:\n" + (workspace / "primitives/primitives.md").read_text()
    )
    if args.phase == "test":
        inherited = bundle(workspace)
        initial += "\n\nFROZEN KNOWLEDGE (READ ONLY):\n" + (inherited or "(none)")
    for turn in range(1, args.turns + 1):
        status = native("status")
        if status["closed"]:
            return
        folder = workspace / status["observation"]
        state = public_state(folder)
        prompt = (initial + "\n\n" if first else "") + (
            f"Phase: {args.phase}. SDK turn {turn}/{args.turns}.\n"
            f"Current session budgets and success flag: {json.dumps(status)}\n"
            f"Current public observation (state, command and instruction): {json.dumps(state)}\n"
        )
        if recent is not None:
            prompt += "Previous execution result:\n" + json.dumps(recent, ensure_ascii=False)
        if turn == args.turns or status.get("success") or status.get("executions_remaining", 1) == 0:
            prompt += "\nThis is your final turn. Return finish and, in Playground, save useful knowledge now."
        images = [
            {"name": camera, "base64": base64.b64encode((folder / f"current_{camera}.png").read_bytes()).decode("ascii")}
            for camera in ("cam_head",) if (folder / f"current_{camera}.png").is_file()
        ]
        prompt += "\nAttached images in order: " + ", ".join(x["name"] for x in images)
        started = time.monotonic()
        action = call_brain(prompt, images)
        first = False
        op = action["op"]
        summary = {"turn": turn, "seconds": round(time.monotonic() - started, 2),
                   "op": op, "note": action.get("note", ""), "include": action.get("include", []),
                   "files": [f["path"] for f in action.get("files", [])]}
        jwrite(audit, summary)
        try:
            writes = action.get("files", [])
            if args.phase == "test" and writes:
                raise ValueError("Test cannot modify the inherited skills or lessons")
            prepared = [(safe_path(workspace, f["path"]), f["content"]) for f in writes]
            if sum(len(content.encode()) for _, content in prepared) > 45_000:
                raise ValueError("knowledge write exceeds 45 kB")
            for path, content in prepared:
                if path.suffix == ".py":
                    compile(content, str(path.relative_to(workspace)), "exec")
            for path, content in prepared:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            if op == "finish":
                native("finish")
                return
            if turn == args.turns:
                native("finish")
                return
            if op == "reset":
                if args.phase != "playground":
                    raise ValueError("reset is forbidden in Test")
                outcome = native("reset")
            elif op == "exec":
                include = action.get("include", [])
                for name in include:
                    safe_path(workspace, name, include=True)
                program = action["program"]
                if not program.isascii() or len(program) > 45_000:
                    raise ValueError("submitted program must be ASCII and at most 45 kB")
                entry = "submission/solution.py"
                (workspace / entry).write_text(program)
                files = source_snapshot(workspace, entry, include)
                outcome = native("exec", files=files)
            else:
                raise ValueError(f"unsupported SDK operation {op}")
            recent = {k: outcome.get(k) for k in ("ok", "success", "error", "stdout", "stderr",
                                                   "native_steps_used", "native_steps_remaining", "observation")}
            for key in ("stdout", "stderr"):
                if isinstance(recent.get(key), str):
                    recent[key] = recent[key][-5000:]
            jwrite(audit, {"turn": turn, "feedback": recent})
        except (FileNotFoundError, ValueError, SyntaxError) as exc:
            recent = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            jwrite(audit, {"turn": turn, "feedback": recent})
    native("finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--phase", choices=("playground", "test"), required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=40)
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        (args.episode / "sdk-error.log").write_text(traceback.format_exc())
        raise
