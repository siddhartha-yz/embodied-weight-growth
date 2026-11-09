"""Save absolute action-head edits in a complete offline SmolVLA checkpoint."""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smol_actor import SmolActor

PREFIX = "model.action_out_proj."
MODEL = "model.safetensors"


def set_absolute(state, edits):
    result = []
    seen = set()
    for edit in edits:
        row, col, value = edit["row"], edit["column"], edit["value"]
        weight = state[PREFIX + "weight"]
        if (
            type(row) is not int or type(col) is not int
            or not 0 <= row < weight.shape[0] or not -1 <= col < weight.shape[1]
            or (row, col) in seen or type(value) not in (float, int)
        ):
            raise ValueError(f"Invalid or repeated absolute edit: {edit}")
        seen.add((row, col))
        tensor = state[PREFIX + ("bias" if col == -1 else "weight")]
        if tensor.dtype != torch.float32:
            raise ValueError(f"Expected FP32 action projection, found {tensor.dtype}")
        k = row if col == -1 else (row, col)
        cast = torch.tensor(value, dtype=tensor.dtype)
        if not bool(cast.isfinite()):
            raise ValueError(f"Non-finite FP32 value: {edit}")
        before = tensor[k].item()
        tensor[k] = cast
        result.append({"row": row, "column": col, "before": before, "saved": tensor[k].item()})
    return result


def equal_bits(tensor, expected):
    return tensor.dtype == expected.dtype and torch.equal(
        tensor.detach().cpu().contiguous().view(torch.int32),
        expected.contiguous().view(torch.int32),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edits", type=Path, required=True,
                        help="JSON with an edits list of unique row, column, value assignments")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "models/smolvla_libero")
    parser.add_argument("--vlm-assets", type=Path, default=ROOT / "models/smolvlm2")
    parser.add_argument("--verify-device", default="cpu")
    args = parser.parse_args()

    base, out = args.base.resolve(), args.out.resolve()
    if out.exists() or out.is_relative_to(base):
        raise FileExistsError(f"Choose a new output directory outside the base checkpoint: {out}")
    edits = json.loads(args.edits.read_text())["edits"]
    out.parent.mkdir(parents=True, exist_ok=True)

    # Publish a complete directory only once the ordinary actor reloads it.
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".materialize-") as temp:
        stage = Path(temp) / "checkpoint"
        shutil.copytree(base, stage, ignore=shutil.ignore_patterns(MODEL, "vlm_assets", "materialization.json"))
        assets = stage / "vlm_assets"
        shutil.copytree(args.vlm_assets, assets)
        source = base / MODEL
        with safe_open(source, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        state = load_file(source, device="cpu")
        result = set_absolute(state, edits)
        expected = {kind: state[PREFIX + kind].clone() for kind in ("weight", "bias")}
        save_file(state, stage / MODEL, metadata=metadata)
        del state

        actor = SmolActor(checkpoint_dir=stage, vlm_assets_dir=assets,
                          device=args.verify_device, n_action_steps=10)
        layer = actor.policy.model.action_out_proj
        for kind, tensor in expected.items():
            if not equal_bits(getattr(layer, kind), tensor):
                raise ValueError(f"Reloaded action projection {kind} differs from the saved model")
        evidence = {"base_checkpoint": str(base), "checkpoint": str(out),
                    "verification_device": args.verify_device,
                    "actor_action_projection_fp32_bit_exact": True, "edits": result}
        (stage / "materialization.json").write_text(json.dumps(evidence, indent=2) + "\n")
        stage.rename(out)
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
