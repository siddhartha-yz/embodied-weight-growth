"""Compile a submitted motor module; share the W pilot's restricted interpreter."""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "w_writer"))
from policy import (FEATURES as W_FEATURES, MAX_HIDDEN, OBS_FIELDS as W_OBS_FIELDS,
                    PolicyError, _output, _parse, _statements, _vector)

OBS_FIELDS = (*W_OBS_FIELDS, "block", "block_angle", "block_v", "contact")
FEATURES = (*W_FEATURES, "block_x", "block_y", "block_cos", "block_sin",
            "block_vx", "block_vy", "tip_block_contact", "block_wall_contact")


def _inputs(obs, goal):
    return {key: _vector(obs[key], 2, key) for key in OBS_FIELDS}, _vector(goal, 2, "goal")


def _compile_weights(spec):
    if not isinstance(spec, dict) or set(spec) != {"layers"}:
        raise PolicyError("weights must be an object with only a 'layers' field")
    layers = spec["layers"]
    if not isinstance(layers, list) or len(layers) not in (1, 2):
        raise PolicyError("supply one output layer, or one hidden layer and one output layer")
    compiled = []
    n_in = len(FEATURES)
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {"weight", "bias"}:
            raise PolicyError(f"layer {i} needs 'weight' and 'bias'")
        rows = layer["weight"]
        final = i == len(layers) - 1
        if not isinstance(rows, list) or (len(rows) != 2 if final else not 1 <= len(rows) <= MAX_HIDDEN):
            raise PolicyError(f"layer {i} requires " + ("2 output rows" if final else f"1–{MAX_HIDDEN} hidden rows"))
        compiled.append(([_vector(row, n_in, f"layer {i} weight row") for row in rows],
                         _vector(layer["bias"], len(rows), f"layer {i} bias")))
        n_in = len(rows)

    def run(obs, goal):
        obs, goal = _inputs(obs, goal)
        x, y = obs["tip"]
        z = [1.0, *obs["q"], *obs["dq"], x, y, *obs["tip_v"],
             goal[0]-x, goal[1]-y, *obs["prev_u"], *obs["prev2_u"],
             *obs["block"], *obs["block_angle"], *obs["block_v"], *obs["contact"]]
        for i, (rows, bias) in enumerate(compiled):
            z = _vector([math.fsum(w*a for w, a in zip(row, z)) + b
                         for row, b in zip(rows, bias)], len(rows), f"layer {i} output")
            if i < len(compiled)-1:
                z = [math.tanh(v) for v in z]
        return _output(z)

    return run


def _compile_code(source):
    body = _parse(source)

    def run(obs, goal):
        obs, goal = _inputs(obs, goal)
        returned, value = _statements(body, {"obs": obs, "goal": goal})
        if not returned:
            raise PolicyError("the executed policy branch did not return")
        return _output(value)

    return run


def compile_policy(arm, spec):
    if arm == "weights":
        return _compile_weights(spec)
    if arm == "code":
        return _compile_code(spec)
    raise PolicyError("arm must be 'weights' or 'code'")
