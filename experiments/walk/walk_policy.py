"""Compile direct numeric or restricted Python policies for the same walker."""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "w_writer"))
from policy import PolicyError, _number, _parse, _statements, _vector

JOINT_ORDER = ("right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle")
FEATURES = (
    "1", "z", "pitch", "vx", "vz", "pitch_v",
    *(f"q{i}" for i in range(6)), *(f"dq{i}" for i in range(6)),
    "contact0", "contact1", "clock_sin", "clock_cos", "t", "target_v", "x",
)
MAX_HIDDEN = 24
N_MOTORS = len(JOINT_ORDER)


def inputs(state, target_v):
    return {
        "1": 1.0, "z": state["z"], "pitch": state["pitch"], "vx": state["vx"],
        "vz": state["vz"], "pitch_v": state["pitch_v"],
        **{f"q{i}": q for i, q in enumerate(state["q"])},
        **{f"dq{i}": v for i, v in enumerate(state["dq"])},
        "contact0": state["contact"][0], "contact1": state["contact"][1],
        "clock_sin": state["clock_sin"], "clock_cos": state["clock_cos"],
        "t": state["t"], "target_v": target_v, "x": state["x"],
    }


def output(value):
    return [max(-1.0, min(1.0, u)) for u in _vector(value, N_MOTORS, "motor output")]


def compile_weights(spec):
    if not isinstance(spec, dict) or set(spec) != {"layers"}:
        raise PolicyError("weights must be an object with one 'layers' field")
    layers = spec["layers"]
    if not isinstance(layers, list) or not 1 <= len(layers) <= 3:
        raise PolicyError("weights need one output layer and at most two hidden layers")

    compiled = []
    n_in = len(FEATURES)
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {"rows"} or not isinstance(layer["rows"], list):
            raise PolicyError(f"layer {i} needs a list of rows")
        rows = layer["rows"]
        final = i == len(layers) - 1
        if final and len(rows) != N_MOTORS:
            raise PolicyError(f"output layer needs {N_MOTORS} motor rows")
        if not final and not 1 <= len(rows) <= MAX_HIDDEN:
            raise PolicyError(f"hidden layer needs 1..{MAX_HIDDEN} rows")
        resolved = []
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {"terms", "bias"}
                    or not isinstance(row["terms"], list)):
                raise PolicyError("each row needs 'terms' and 'bias'")
            terms = {}
            for term in row["terms"]:
                if not isinstance(term, dict) or set(term) != {"feature", "coef"}:
                    raise PolicyError("each term needs 'feature' and 'coef'")
                feature = term["feature"]
                if type(feature) is not int or not 0 <= feature < n_in or feature in terms:
                    raise PolicyError(f"row feature indexes must be unique integers in [0, {n_in})")
                terms[feature] = _number(term["coef"], "coefficient")
            resolved.append((tuple(terms.items()), _number(row["bias"], "bias")))
        compiled.append(resolved)
        n_in = len(rows)

    def run(state, target_v):
        obs = inputs(state, target_v)
        z = [obs[name] for name in FEATURES]
        for i, rows in enumerate(compiled):
            z = [
                _number(math.fsum(z[j] * coefficient for j, coefficient in terms) + bias,
                        "layer output")
                for terms, bias in rows
            ]
            if i < len(compiled) - 1:
                z = [math.tanh(v) for v in z]
        return output(z)

    return run


def compile_code(source):
    body = _parse(source)

    def run(state, target_v):
        obs = inputs(state, target_v)
        obs.update({k: list(state[k]) for k in ("q", "dq", "contact")})
        returned, value = _statements(body, {"obs": obs, "goal": [target_v]})
        if not returned:
            raise PolicyError("the executed policy branch did not return")
        return output(value)

    return run


def compile_policy(arm, spec):
    if arm == "weights":
        return compile_weights(spec)
    if arm == "code":
        return compile_code(spec)
    raise PolicyError("arm must be 'weights' or 'code'")
