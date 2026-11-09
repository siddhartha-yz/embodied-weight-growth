"""Execute an installed, numeric-only, hard-gated action network.

A stage selects a small feed-forward network. Its four output logits select
an operation; the other outputs specify a position, a jaw command, the
next stage, and a world-Z yaw. The caller advances the gate with measured post-action
sensors, so success/failure transitions are weights rather than host rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real


FEATURES = (
    "state",
    "source_x", "source_y", "source_z",
    "dest_x", "dest_y", "dest_z",
    "tcp_x", "tcp_y", "tcp_z",
    "jaw", "ok", "source_yaw",
)
OPS = ("move_tcp", "grip", "done", "fail_reobserve")
OUTPUTS = (*OPS, "target_x", "target_y", "target_z", "target_jaw", "next_state", "target_yaw")
MAX_STAGES = 16
MAX_LAYERS = 3
MAX_HIDDEN = 16


class PolicyError(ValueError):
    """The weights or their input/output violate the policy contract."""


def _finite(value: object, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise PolicyError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise PolicyError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise PolicyError(f"{label} must be a finite number")
    return result


@dataclass(frozen=True)
class Decision:
    op: str
    xyz: tuple[float, float, float]
    jaw: float
    logits: tuple[float, float, float, float]
    yaw: float


@dataclass(frozen=True)
class _Layer:
    activation: str
    bias: tuple[float, ...]
    rows: tuple[tuple[tuple[int, float], ...], ...]

    def __call__(self, values: tuple[float, ...]) -> tuple[float, ...]:
        try:
            output = tuple(
                _finite(math.fsum([bias, *(coef * values[col] for col, coef in terms)]), "layer output")
                for bias, terms in zip(self.bias, self.rows)
            )
        except (OverflowError, ValueError) as exc:
            raise PolicyError("non-finite layer output") from exc
        if self.activation == "tanh":
            return tuple(math.tanh(value) for value in output)
        return output


@dataclass(frozen=True)
class Policy:
    _stages: tuple[tuple[_Layer, ...], ...]

    @property
    def stages(self) -> int:
        return len(self._stages)

    def outputs(self, state: int, sensors: Mapping[str, object]) -> tuple[float, ...]:
        if type(state) is not int or not 0 <= state < self.stages:
            raise PolicyError(f"state must be a stage index in [0, {self.stages})")
        if not isinstance(sensors, Mapping):
            raise PolicyError("sensors must map the named features to numbers")
        missing = [feature for feature in FEATURES[1:] if feature not in sensors]
        if missing:
            raise PolicyError(f"missing sensors: {', '.join(missing)}")
        ok = sensors["ok"]
        ok = float(ok) if isinstance(ok, bool) else _finite(ok, "sensor ok")
        if ok not in (0.0, 1.0):
            raise PolicyError("sensor ok must be 0 or 1")
        values = (
            float(state),
            *(ok if name == "ok" else _finite(sensors[name], f"sensor {name}") for name in FEATURES[1:]),
        )
        for layer in self._stages[state]:
            values = layer(values)
        return values

    def decide(self, state: int, sensors: Mapping[str, object]) -> Decision:
        values = self.outputs(state, sensors)
        logits = values[0], values[1], values[2], values[3]
        op = OPS[max(range(len(OPS)), key=logits.__getitem__)]
        return Decision(op, (values[4], values[5], values[6]), values[7], logits, values[9])

    def advance(self, state: int, sensors: Mapping[str, object]) -> int:
        """Use the same stage's state output on the new physical feedback."""
        next_state = round(self.outputs(state, sensors)[8])
        if not 0 <= next_state < self.stages:
            raise PolicyError(f"next_state rounded to {next_state}, outside [0, {self.stages})")
        return next_state


def _load_layer(spec: object, n_in: int, *, final: bool, label: str) -> _Layer:
    if not isinstance(spec, dict) or set(spec) != {"activation", "bias", "weights"}:
        raise PolicyError(f"{label} must contain activation, bias, and weights")
    activation, bias, weights = spec["activation"], spec["bias"], spec["weights"]
    if activation not in ("identity", "tanh"):
        raise PolicyError(f"{label} activation must be identity or tanh")
    if not isinstance(bias, list):
        raise PolicyError(f"{label} bias must be a list")
    n_out = len(bias)
    if final:
        if n_out != len(OUTPUTS) or activation != "identity":
            raise PolicyError(f"{label} must have {len(OUTPUTS)} outputs with identity activation")
    elif not 1 <= n_out <= MAX_HIDDEN:
        raise PolicyError(f"{label} hidden width must be between 1 and {MAX_HIDDEN}")
    if not isinstance(weights, list):
        raise PolicyError(f"{label} weights must be a list of [row, column, coefficient] triples")
    rows: list[dict[int, float]] = [{} for _ in range(n_out)]
    for term in weights:
        if not isinstance(term, list) or len(term) != 3:
            raise PolicyError(f"{label} weight must be a [row, column, coefficient] triple")
        row, col, coef = term
        if type(row) is not int or not 0 <= row < n_out:
            raise PolicyError(f"{label} weight row must be an integer in [0, {n_out})")
        if type(col) is not int or not 0 <= col < n_in:
            raise PolicyError(f"{label} weight column must be an integer in [0, {n_in})")
        if col in rows[row]:
            raise PolicyError(f"{label} has duplicate weight [{row}, {col}]")
        rows[row][col] = _finite(coef, f"{label} coefficient")
    return _Layer(
        activation,
        tuple(_finite(value, f"{label} bias") for value in bias),
        tuple(tuple(row.items()) for row in rows),
    )


def load_policy(spec: object) -> Policy:
    """Validate and compile an immutable policy from sparse numeric JSON."""
    if not isinstance(spec, dict) or set(spec) != {"stages"}:
        raise PolicyError("policy must be an object with a stages field")
    stages = spec["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= MAX_STAGES:
        raise PolicyError(f"policy must contain between 1 and {MAX_STAGES} stages")
    compiled = []
    for stage, entry in enumerate(stages):
        if not isinstance(entry, dict) or set(entry) != {"layers"}:
            raise PolicyError(f"stage {stage} must contain layers")
        layers = entry["layers"]
        if not isinstance(layers, list) or not 1 <= len(layers) <= MAX_LAYERS:
            raise PolicyError(f"stage {stage} must contain between 1 and {MAX_LAYERS} layers")
        compiled_layers = []
        n_in = len(FEATURES)
        for i, layer in enumerate(layers):
            result = _load_layer(layer, n_in, final=i == len(layers) - 1, label=f"stage {stage} layer {i}")
            compiled_layers.append(result)
            n_in = len(result.bias)
        compiled.append(tuple(compiled_layers))
    return Policy(tuple(compiled))
