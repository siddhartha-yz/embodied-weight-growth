"""Execute an Astra-authored Python decision tool in a networkless local sandbox."""

from __future__ import annotations

import ast
import json
import subprocess


CHILD = """
import json, math, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (200 * 1024 * 1024, 200 * 1024 * 1024))
source, observation = json.load(sys.stdin)
namespace = {"math": math}
exec(compile(source, "<Astra skill>", "exec"), namespace)
print(json.dumps(namespace["decide"](observation), allow_nan=False))
"""
BWRAP = (
    "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
    "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
    "--symlink", "usr/lib64", "/lib64", "--tmpfs", "/tmp", "--chdir", "/tmp", "--clearenv",
    "/usr/bin/python3", "-I", "-c", CHILD,
)
OPS = {"move_tcp", "grip", "joint", "pose_preview", "done", "fail_reobserve"}


def validate_source(source):
    tree = ast.parse(source, mode="exec")
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decide"]
    if len(source) > 20_000 or len(definitions) != 1 or len(definitions[0].args.args) != 1:
        raise ValueError("the skill must define one decide(observation) function")
    return source


def call_skill(source, observation):
    try:
        result = subprocess.run(
            BWRAP, input=json.dumps([source, observation], allow_nan=False), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("skill exceeded the sandbox time budget") from error
    if result.returncode:
        raise ValueError(f"skill did not produce a command: {result.stderr[-1400:]}")
    reply = json.loads(result.stdout)
    if not isinstance(reply, dict) or reply.get("op") not in OPS:
        raise ValueError("skill must return a dict with a supported op")
    state = reply.get("state", {})
    if not isinstance(state, dict) or len(json.dumps(state)) > 8192:
        raise ValueError("skill state must be a small dictionary")
    return reply
