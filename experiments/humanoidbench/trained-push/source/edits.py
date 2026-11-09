"""Apply bounded direct edits or a frozen-actor Python action residual."""
import ast
import json
import math

import numpy as np

from simulators.interactive import NUMPY, SAFE_BUILTINS, validate_snippet

MAX_EDITS = 192
MAX_TENSORS = 2


def _unique(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"Duplicate key: {k!r}")
        d[k] = v
    return d


def json_payload(text):
    def fail(token):
        raise ValueError(f"Invalid numeric constant: {token}")
    return json.loads(text, object_pairs_hook=_unique, parse_constant=fail)


def sparse_edit(text, params):
    obj = json_payload(text)
    if not isinstance(obj, dict) or obj.keys() != {"edits"} or not isinstance(obj["edits"], list):
        raise ValueError("Expected JSON object with an edits list")
    edits = obj["edits"]
    if not 1 <= len(edits) <= MAX_EDITS:
        raise ValueError(f"Expected 1 to {MAX_EDITS} sparse scalar edits")
    updates, seen = {}, set()
    for row in edits:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Each edit is [tensor name, index list, additive delta]")
        name, indexes, delta = row
        if not isinstance(name, str) or name not in params:
            raise ValueError(f"Unknown tensor: {name!r}")
        x = params[name]
        if (not isinstance(indexes, list) or len(indexes) != x.ndim
                or any(type(i) is not int or not 0 <= i < n for i, n in zip(indexes, x.shape))):
            raise ValueError(f"Index {indexes!r} does not fit {name} shape {x.shape}")
        if type(delta) not in (int, float) or not -5 <= delta <= 5 or delta == 0:
            raise ValueError("Deltas must be finite and nonzero in [-5, 5]")
        loc = tuple(indexes)
        if (name, loc) in seen:
            raise ValueError("Duplicate tensor coordinate")
        seen.add((name, loc))
        if name not in updates:
            if len(updates) >= MAX_TENSORS:
                raise ValueError(f"An edit can touch at most {MAX_TENSORS} named tensors")
            updates[name] = x.copy()
        updates[name][loc] += delta
    return {**params, **updates}


def residual_code(source):
    if len(source) > 12000 or not source.isascii():
        raise ValueError("Residual source must be ASCII, at most 12000 characters")
    validate_snippet(source)
    module = ast.parse(source)
    for node in module.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            raise ValueError("Only imports and function definitions are allowed at top level")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.returns
                or node.args.defaults or any(x is not None for x in node.args.kw_defaults)
                or any(a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs))
                or node.args.vararg is not None or node.args.kwarg is not None):
            raise ValueError("Define plain functions, no decorators, defaults or annotations")
    globs = {"__builtins__": dict(SAFE_BUILTINS), "np": NUMPY, "math": math}
    exec(compile(module, "<action-residual>", "exec"), globs)
    fn = globs.get("residual")
    if not callable(fn):
        raise ValueError("Define residual(obs, base_action, hidden, memory)")
    return fn


class Controller:
    """One base actor or one complete candidate; reset only residual memory."""

    def __init__(self, actor, *, params=None, residual=None, mute=()):
        self.actor = actor
        self.params = params
        self.residual = residual
        self.mute = tuple(mute)
        self.reset()

    def reset(self):
        self.memory = {}
        self.count = 0
        self.logs = []
        self.collect = False
        self._held = None

    def __call__(self, obs):
        if self.count % 2:
            action, hidden = self._held
        else:
            action, hidden = self.compute(obs)
            self._held = action, hidden
        if self.collect:
            self.logs.append((self.count, np.asarray(obs).copy(), action.copy(), hidden.copy()))
        self.count += 1
        return action

    def compute(self, obs):
        if self.mute:
            action, hidden = self.actor.forward(obs, self.params, mute=self.mute)
        else:
            action, hidden = self.actor.forward(obs, self.params)
        action = np.asarray(action, dtype=np.float64)
        h = np.asarray(hidden)
        if action.shape != (19,) or not np.isfinite(action).all() or h.ndim != 1 or not np.isfinite(h).all():
            raise ValueError("Actor did not produce finite action19/final hidden")
        if self.residual:
            delta = np.asarray(self.residual(np.asarray(obs).copy(), action.copy(), h.copy(),
                                              self.memory), dtype=np.float64)
            if delta.shape != (19,) or not np.isfinite(delta).all():
                raise ValueError("residual() must return 19 finite action corrections")
            action = np.clip(action + delta, -1, 1)
        return action, h
