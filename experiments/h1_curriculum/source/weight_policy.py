"""Fixed affine–tanh–affine H1 policy with named, sparse numeric weights."""

from dataclasses import dataclass
import json
from pathlib import Path

FORMAT = "h1-curriculum-v1"
HIDDEN_NAMES = tuple(f"h{i}" for i in range(32))
MAX_MAGNITUDE = 10_000


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(token):
    raise ValueError(f"Invalid JSON number {token!r}")


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_pairs, parse_constant=_invalid_constant)


def empty_checkpoint():
    return {"format": FORMAT, "hidden": {}, "outputs": {}}


def check_names(feature_names, action_names):
    for label, names in (("features", feature_names), ("actions", action_names)):
        if (
            not isinstance(names, (list, tuple))
            or not names
            or not all(isinstance(name, str) and name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError(f"Expected ordered, distinct names for {label}")
        collisions = set(names) & {"bias", *HIDDEN_NAMES}
        if collisions:
            raise ValueError(f"Reserved names used in {label}: {sorted(collisions)}")
    if feature_names.count("target_vx") != 1:
        raise ValueError("Policy must have exactly one continuous target_vx feature")
    if len(action_names) != 19:
        raise ValueError("Expected the 19 native H1 actuators")


def numeric(value, label):
    if type(value) not in (int, float) or not -MAX_MAGNITUDE <= value <= MAX_MAGNITUDE:
        raise ValueError(f"{label} must be a finite number in ±{MAX_MAGNITUDE}")
    return float(value)


def normalize_checkpoint(payload, feature_names, action_names):
    """Validate and canonicalize a checkpoint without importing NumPy."""
    check_names(feature_names, action_names)
    if (
        not isinstance(payload, dict)
        or not {"format", "outputs"} <= set(payload)
        or set(payload) - {"format", "hidden", "outputs"}
        or payload["format"] != FORMAT
    ):
        raise ValueError(f"Expected format {FORMAT!r} and sparse hidden/outputs maps")
    result = empty_checkpoint()
    for layer, row_names, sources in (
        ("hidden", HIDDEN_NAMES, feature_names),
        ("outputs", action_names, (*feature_names, *HIDDEN_NAMES)),
    ):
        rows = payload.get(layer, {})
        if not isinstance(rows, dict):
            raise ValueError(f"{layer} must be an object")
        unknown_rows = set(rows) - set(row_names)
        if unknown_rows:
            raise ValueError(f"Unknown {layer} rows: {sorted(unknown_rows)}")
        valid_sources = set(sources)
        for name, row in rows.items():
            if not isinstance(row, dict) or set(row) - {"bias", "weights"}:
                raise ValueError(f"{layer}.{name} accepts only bias and weights")
            weights = row.get("weights", {})
            if not isinstance(weights, dict) or set(weights) - valid_sources:
                raise ValueError(f"{layer}.{name} has an invalid source map")
            bias = numeric(row.get("bias", 0), f"{layer}.{name}.bias")
            terms = {source: numeric(value, f"{layer}.{name}.{source}") for source, value in weights.items()}
            nonzero = {k: v for k, v in terms.items() if v != 0}
            if bias or nonzero:
                result[layer][name] = {
                    **({"bias": bias} if bias else {}),
                    **({"weights": nonzero} if nonzero else {}),
                }
    return result


@dataclass(frozen=True)
class Policy:
    hidden_weights: object
    hidden_bias: object
    output_weights: object
    output_bias: object

    @classmethod
    def load(cls, path, feature_names, action_names):
        return cls.compile(read_json(path), feature_names, action_names)

    @classmethod
    def compile(cls, payload, feature_names, action_names):
        import numpy as np

        data = normalize_checkpoint(payload, feature_names, action_names)

        def affine(layer, targets, sources):
            indices = {name: i for i, name in enumerate(sources)}
            weight = np.zeros((len(targets), len(sources)), dtype=np.float64)
            bias = np.zeros(len(targets), dtype=np.float64)
            for i, name in enumerate(targets):
                row = data[layer].get(name, {})
                bias[i] = row.get("bias", 0)
                for k, value in row.get("weights", {}).items():
                    weight[i, indices[k]] = value
            weight.flags.writeable = False
            bias.flags.writeable = False
            return weight, bias

        return cls(
            *affine("hidden", HIDDEN_NAMES, feature_names),
            *affine("outputs", action_names, (*feature_names, *HIDDEN_NAMES)),
        )

    def act(self, observation):
        import numpy as np

        x = np.asarray(observation, dtype=np.float64)
        if x.shape != (self.hidden_weights.shape[1],) or not np.isfinite(x).all():
            raise ValueError("Expected one finite observation in the registered feature order")
        h = np.tanh(self.hidden_weights @ x + self.hidden_bias)
        u = self.output_weights @ np.concatenate((x, h)) + self.output_bias
        if not np.isfinite(u).all():
            raise ValueError("Nonfinite action")
        return np.clip(u, -1, 1)
