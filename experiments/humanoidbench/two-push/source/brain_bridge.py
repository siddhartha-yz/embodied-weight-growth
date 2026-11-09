#!/usr/bin/env python3
"""Loopback Codex SDK relay for in-episode HumanoidBench weight patches."""
from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import signal
import sys
import tempfile
import time

MODEL = "gpt-6-astra"
DEFAULT_EFFORT = "medium"
CODEX_BIN = "/usr/lib/chatgpt/resources/codex"
SDK_DIR = Path.home() / "文档/ChatGPT/奇怪的idea/work/body-growth-pilot/.runtime/sdk"
PASSIVE_ITEMS = {"userMessage", "reasoning", "agentMessage", "plan"}
MAX_REQUEST = 12 * 1024 * 1024
BASE = (
    "You are the weight-space programming agent for one live HumanoidBench episode. "
    "At each event use only the supplied observations and the episode's prior feedback. "
    "Do not call tools, access files, run commands, write controller code, or train weights. "
    "Return only the structured action with op (patch or pass), payload (string), and "
    "a brief note (string). For patch, encode a sparse weight patch as JSON text, "
    "following the host's format and its reference weights. For pass, return an empty "
    "payload. The host alone applies accepted weight patches and advances the episode."
)

def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


STR = {"type": "string"}
SCHEMA = obj({
    "op": {"type": "string", "enum": ["patch", "pass"]},
    "payload": STR,
    "note": STR,
})

def no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key: {key}")
        out[key] = value
    return out


def reject_constant(value):
    raise ValueError(f"not a JSON constant: {value}")


def decode_action(raw):
    d = json.loads(raw, object_pairs_hook=no_duplicate_keys)
    if not isinstance(d, dict) or d.keys() != SCHEMA["properties"].keys():
        raise ValueError("invalid action fields")
    if (d["op"] not in ("patch", "pass")
            or not isinstance(d["payload"], str) or not isinstance(d["note"], str)):
        raise ValueError("invalid operation, payload string, or note string")
    if d["op"] == "pass":
        if d["payload"] != "":
            raise ValueError("pass requires an empty payload")
    else:
        payload = json.loads(d["payload"], object_pairs_hook=no_duplicate_keys,
                             parse_constant=reject_constant)
        if not isinstance(payload, (dict, list)):
            raise ValueError("patch requires a JSON object or array payload")
    return d

def decode_request(body):
    d = json.loads(body, object_pairs_hook=no_duplicate_keys)
    if not isinstance(d, dict) or not {"thread", "prompt"} <= d.keys() <= {"thread", "phase", "prompt", "images"}:
        raise ValueError("expected thread and prompt, with optional images and phase")
    thread, phase, prompt, images = d["thread"], d.get("phase"), d["prompt"], d.get("images", [])
    if not isinstance(thread, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", thread):
        raise ValueError("invalid thread label")
    if phase is not None and phase not in ("playground", "test"):
        raise ValueError("invalid phase")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 240_000:
        raise ValueError("invalid prompt")
    if not isinstance(images, list) or len(images) > 6:
        raise ValueError("invalid image list")
    decoded = []
    for image in images:
        if not isinstance(image, dict) or image.keys() != {"name", "base64"}:
            raise ValueError("invalid image fields")
        name = image["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name) or not isinstance(image["base64"], str):
            raise ValueError("invalid image name or encoding")
        try:
            pixels = base64.b64decode(image["base64"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid base64 image") from exc
        if pixels.startswith(b"\x89PNG\r\n\x1a\n"):
            suffix = ".png"
        elif pixels.startswith(b"\xff\xd8\xff"):
            suffix = ".jpg"
        else:
            raise ValueError("images must be PNG or JPEG")
        decoded.append((name, pixels, suffix))
    return thread, phase, prompt, decoded


class Relay:
    def __init__(self, sdk, codex, workspace, audit, effort=DEFAULT_EFFORT):
        self.sdk = sdk
        self.codex = codex
        self.workspace = workspace
        self.audit = audit
        self.effort = effort
        self.threads = {}

    def __call__(self, request):
        label, phase, prompt, images = request
        key = label, phase
        entry = {"thread": label, "phase": phase, "model": MODEL, "effort": self.effort,
                 "started_utc": datetime.now(timezone.utc).isoformat(), "images": [i[0] for i in images]}
        started = time.monotonic()
        try:
            if key not in self.threads:
                self.threads[key] = self.codex.thread_start(
                    cwd=str(self.workspace), model=MODEL,
                    sandbox=self.sdk.Sandbox.read_only,
                    approval_mode=self.sdk.ApprovalMode.deny_all,
                    ephemeral=True, base_instructions=BASE,
                )
            thread = self.threads[key]
            entry["sdk_thread_id"] = thread.id
            with tempfile.TemporaryDirectory(prefix="frames-", dir=self.workspace) as folder:
                content = [self.sdk.TextInput(prompt)]
                for i, (name, pixels, suffix) in enumerate(images):
                    path = Path(folder) / f"{i}{suffix}"
                    path.write_bytes(pixels)
                    content.extend((self.sdk.TextInput(f"Camera: {name}"), self.sdk.LocalImageInput(str(path))))
                result = thread.run(content, effort=self.effort, output_schema=SCHEMA)
            kinds = [getattr(getattr(item, "root", item), "type", None) for item in result.items]
            status = getattr(result.status, "value", str(result.status))
            entry.update(sdk_turn_id=result.id, status=status, duration_ms=result.duration_ms,
                         item_types=kinds, usage=result.usage.model_dump(mode="json") if result.usage else None)
            if status != "completed" or not result.final_response:
                raise RuntimeError(f"Codex turn did not complete: {status}: {result.error}")
            if any(kind not in PASSIVE_ITEMS for kind in kinds):
                raise RuntimeError(f"Codex returned non-passive item types: {kinds}")
            action = decode_action(result.final_response)
            entry.update(op=action["op"], note=action["note"],
                         payload_chars=len(action["payload"]))
            return action
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry["wall_s"] = round(time.monotonic() - started, 3)
            self.audit.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.audit.flush()


def handler_for(relay, token):
    expected = "Bearer " + token

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/health":
                self.reply(HTTPStatus.OK, {"ok": True})
            else:
                self.reply(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown route"})

        def do_POST(self):
            if self.path != "/turn":
                self.reply(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown route"})
                return
            received = self.headers.get("Authorization", "")
            if not secrets.compare_digest(received.encode(), expected.encode()):
                self.reply(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "invalid authorization"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST:
                    raise ValueError("invalid request size")
                request = decode_request(self.rfile.read(length))
            except (ValueError, UnicodeError) as exc:
                self.reply(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            try:
                action = relay(request)
            except Exception as exc:
                # The authorized lab caller needs the actual SDK error to stop safely.
                self.reply(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                return
            self.reply(HTTPStatus.OK, {"ok": True, "action": action})

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--effort", choices=("low", "medium", "high"), default=DEFAULT_EFFORT)
    args = p.parse_args()
    token = args.token_file.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[!-~]{32,}", token):
        p.error("token must contain at least 32 printable ASCII characters, no spaces")
    sys.path.insert(0, str(SDK_DIR.resolve(strict=True)))
    import openai_codex as sdk

    def stop(*_args):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    with tempfile.TemporaryDirectory(prefix="humanoidbench-live-wsp-codex-") as work:
        conf = sdk.CodexConfig(
            codex_bin=CODEX_BIN, cwd=work,
            config_overrides=(
                "features.shell_tool=false", "features.multi_agent=false",
                "features.apps=false", "features.plugins=false", "features.memories=false",
                'web_search="disabled"', "mcp_servers={}", "project_doc_max_bytes=0",
            ),
        )
        fd = os.open(args.audit, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as audit, sdk.Codex(conf) as codex:
            relay = Relay(sdk, codex, Path(work), audit, effort=args.effort)
            with HTTPServer(("127.0.0.1", args.port), handler_for(relay, token)) as httpd:
                print(f"ready: http://127.0.0.1:{httpd.server_port}/turn", flush=True)
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    pass


if __name__ == "__main__":
    main()
