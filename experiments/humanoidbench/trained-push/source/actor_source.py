"""Deterministic inference/export for DR.Q's official h1-walk-v0 checkpoint.

Source: https://github.com/dmksjfl/DR.Q/blob/cb2b46f679301aca97d381e95cd85bd2950af175/DRQ/models.py
Train wrapper: DRQ/env_preprocessing.py, HBenchPreprocessing: action_repeat=2.
No learned observation normalizer, no exploration noise, and no recurrent state.
The saved bundle contains only tensors needed for deterministic inference.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F

ACTOR_REPEAT = 2
HIDDEN_DIM = 512
OBS_DIM = 51
ACTION_DIM = 19
OUTPUT_WEIGHT = "policy.l3.weight"
OUTPUT_BIAS = "policy.l3.bias"

# Exact widths from the source config. This also catches wrong-body checkpoints.
WIDTHS = {"encoder": (OBS_DIM, 750, 750, 512), "policy": (512, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM)}
SHAPES = {}
for group, widths in WIDTHS.items():
    for layer, (n_in, n_out) in enumerate(zip(widths, widths[1:]), 1):
        SHAPES[f"{group}.l{layer}.weight"] = (n_out, n_in)
        SHAPES[f"{group}.l{layer}.bias"] = (n_out,)


def _check_params(params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(params) != set(SHAPES):
        raise ValueError(f"actor tensors differ: missing={set(SHAPES) - set(params)}, extra={set(params) - set(SHAPES)}")
    clean = {}
    for name, shape in SHAPES.items():
        value = np.asarray(params[name], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} is not a finite tensor of shape {shape}: {value.shape}")
        clean[name] = np.ascontiguousarray(value)
    return clean


class Actor:
    can_mute = True
    description = (
        "DR.Q h1-walk-v0 seed 0: raw 51-float native observation, no obs scaler; "
        "51→750→750→512 LayerNorm+ELU encoder, 512→512→512→19 "
        "LayerNorm+ReLU policy, tanh output to native normalized joint targets. "
        "policy.l3.weight is [actuator, hidden], bias is [actuator]. Native environment 50Hz; "
        "original HBench training wrapper repeats every action for 2 native steps (actor 25Hz)."
    )

    def __init__(self, params: Mapping[str, np.ndarray]):
        self.params = _check_params(params)

    @torch.inference_mode()
    def forward(
        self, obs: np.ndarray, params: Mapping[str, np.ndarray] | None = None,
        mute: tuple[int, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized 19 native targets and the 512-vector feeding the final head.

        Supports a single observation (51,) or batches (N, 51). A caller may replace
        all tensors by supplying a complete param mapping; neither mapping is mutated.
        """
        p = self.params if params is None else params
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim not in (1, 2) or obs.shape[-1] != OBS_DIM:
            raise ValueError(f"expected (...,{OBS_DIM}) native observations, got {obs.shape}")
        single = obs.ndim == 1
        x = torch.from_numpy(np.ascontiguousarray(obs.reshape(-1, OBS_DIM)))

        def linear(value: torch.Tensor, layer: str) -> torch.Tensor:
            weight = torch.from_numpy(np.asarray(p[f"{layer}.weight"], dtype=np.float32))
            bias = torch.from_numpy(np.asarray(p[f"{layer}.bias"], dtype=np.float32))
            return F.linear(value, weight, bias)

        def normalize(value: torch.Tensor) -> torch.Tensor:
            return F.layer_norm(value, (value.shape[-1],))

        for i in range(1, 4):
            x = F.elu(normalize(linear(x, f"encoder.l{i}")))
        for i in range(1, 3):
            x = F.relu(normalize(linear(x, f"policy.l{i}")))
        if mute:
            x[..., list(mute)] = 0.0
        y = torch.tanh(linear(x, "policy.l3"))
        return (y[0].numpy(), x[0].numpy()) if single else (y.numpy(), x.numpy())


def load(path: str | Path) -> Actor:
    """Load the plain numeric NPZ bundle, never a Python-object pickle."""
    with np.load(path, allow_pickle=False) as file:
        return Actor({name: file[name] for name in file.files})


def export(encoder_path: str | Path, policy_path: str | Path, output: str | Path) -> None:
    """Select the inference tensors from the two upstream state dictionaries."""
    encoder = torch.load(encoder_path, map_location="cpu", weights_only=True)
    policy = torch.load(policy_path, map_location="cpu", weights_only=True)
    params = {}
    for name in SHAPES:
        group, leaf = name.split(".", 1)
        tensor = encoder[f"zs_mlp.{leaf}"] if group == "encoder" else policy[name]
        params[name] = tensor.detach().cpu().numpy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **_check_params(params))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export(args.encoder, args.policy, args.out)


if __name__ == "__main__":
    main()
