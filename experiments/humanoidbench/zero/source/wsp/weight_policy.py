"""Fixed numerical forward pass for the untrained HumanoidBench H1 weight arm."""

from dataclasses import dataclass
import json
import math
import re

import numpy as np


FORMAT = "h1-wsp-v1"
NQ, NV, NU = 26, 25, 19
INPUT_NAMES = (
    *(f"qpos_{i}" for i in range(NQ)),
    *(f"qvel_{i}" for i in range(NV)),
    *(f"delta_{i}" for i in range(NQ)),
    "t",
)
ACTIVATIONS = ("tanh", "sin", "relu")
MAX_HIDDEN = 128
MAX_PARAMETERS = 4096
MAX_MAGNITUDE = 10_000
MAX_JSON_BYTES = 128 * 1024
HIDDEN_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}")


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate key: {name!r}")
        result[name] = value
    return result


def _nonfinite(token):
    raise ValueError(f"Nonfinite JSON number: {token}")


def _fields(value, label, *, required=(), optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = set(required) - value.keys()
    unknown = value.keys() - (set(required) | set(optional))
    if missing or unknown:
        raise ValueError(f"{label}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}")
    return value


def _number(value, label):
    if (type(value) not in (int, float) or not -MAX_MAGNITUDE <= value <= MAX_MAGNITUDE):
        raise ValueError(f"{label} must be a finite number in [-{MAX_MAGNITUDE}, {MAX_MAGNITUDE}]")
    return float(value)


def _vector(value, size, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {size} finite numbers")
    return result


@dataclass(frozen=True)
class WeightPolicy:
    """Sparse JSON is compiled into immutable affine layers, without executing it."""

    _hidden_w: np.ndarray
    _hidden_b: np.ndarray
    _activation_indices: tuple
    _output_w: np.ndarray
    _output_b: np.ndarray

    @classmethod
    def from_json(cls, text, channels):
        if not isinstance(text, str) or not text.isascii() or len(text) > MAX_JSON_BYTES:
            raise ValueError(f"Weights must be ASCII JSON, at most {MAX_JSON_BYTES} bytes")
        try:
            payload = json.loads(text, object_pairs_hook=_object, parse_constant=_nonfinite)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(f"Invalid policy JSON: {exc}") from exc
        _fields(payload, "policy", required=("format", "outputs"), optional=("hidden",))
        if payload["format"] != FORMAT:
            raise ValueError(f"format must be {FORMAT!r}")
        hidden = payload.get("hidden", {})
        outputs = payload["outputs"]
        if not isinstance(hidden, dict) or len(hidden) > MAX_HIDDEN:
            raise ValueError(f"hidden must be an object with at most {MAX_HIDDEN} units")
        if not isinstance(outputs, dict):
            raise ValueError("outputs must be an object keyed by native actuator name")
        names = [c["joint"] for c in channels]
        if len(names) != NU or len(set(names)) != NU:
            raise ValueError("Expected 19 uniquely named native actuator channels")

        inputs = {name: i for i, name in enumerate(INPUT_NAMES)}
        nh = len(hidden)
        ni = len(inputs)
        hw, hb = np.zeros((nh, ni)), np.zeros(nh)
        ow, ob = np.zeros((NU, ni + nh)), np.zeros(NU)
        activation_rows = {activation: [] for activation in ACTIVATIONS}
        feature_indices = dict(inputs)
        nparam = 0

        def read_affine(spec, label, row, matrix, biases, features):
            nonlocal nparam
            weights = spec.get("weights", {})
            if not isinstance(weights, dict):
                raise ValueError(f"{label}.weights must be an object keyed by feature name")
            nparam += len(weights) + ("bias" in spec)
            if nparam > MAX_PARAMETERS:
                raise ValueError(f"More than {MAX_PARAMETERS} numeric parameters")
            biases[row] = _number(spec.get("bias", 0), f"{label}.bias")
            for name, value in weights.items():
                if name not in features:
                    raise ValueError(f"Unknown feature {name!r} in {label}")
                matrix[row, features[name]] = _number(value, f"{label}.weights.{name}")

        for row, (name, spec) in enumerate(hidden.items()):
            if not HIDDEN_NAME.fullmatch(name) or name in inputs:
                raise ValueError(f"Invalid hidden unit name: {name!r}")
            label = f"hidden.{name}"
            _fields(spec, label, required=("activation",), optional=("bias", "weights"))
            activation = spec["activation"]
            if activation not in ACTIVATIONS:
                raise ValueError(f"{label}.activation must be one of {ACTIVATIONS}")
            activation_rows[activation].append(row)
            read_affine(spec, label, row, hw, hb, inputs)
            feature_indices[name] = ni + row

        output_rows = {name: i for i, name in enumerate(names)}
        for name, spec in outputs.items():
            if name not in output_rows:
                raise ValueError(f"Unknown actuator: {name!r}")
            label = f"outputs.{name}"
            _fields(spec, label, optional=("bias", "weights"))
            read_affine(spec, label, output_rows[name], ow, ob, feature_indices)

        activation_indices = tuple(np.array(activation_rows[name], dtype=np.intp) for name in ACTIVATIONS)
        for array in (hw, hb, ow, ob, *activation_indices):
            array.flags.writeable = False
        return cls(hw, hb, activation_indices, ow, ob)

    def action(self, qpos, qvel, initial_qpos, t):
        qpos = _vector(qpos, NQ, "qpos")
        qvel = _vector(qvel, NV, "qvel")
        initial = _vector(initial_qpos, NQ, "initial_qpos")
        t = float(t)
        if not math.isfinite(t) or t < 0:
            raise ValueError("t must be finite nonnegative simulated seconds")
        try:
            with np.errstate(over="raise", invalid="raise"):
                base = np.concatenate((qpos, qvel, qpos - initial, [t]))
                hidden = self._hidden_w @ base + self._hidden_b
                tanh, sin, relu = self._activation_indices
                hidden[tanh] = np.tanh(hidden[tanh])
                hidden[sin] = np.sin(hidden[sin])
                hidden[relu] = np.maximum(hidden[relu], 0.0)
                output = self._output_w @ np.concatenate((base, hidden)) + self._output_b
        except FloatingPointError as exc:
            raise ValueError("Nonfinite policy forward pass") from exc
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite policy forward pass")
        return np.clip(output, -1.0, 1.0)
