"""Recipient-scene prompt-contrast V activation diagnostic, with no weight edits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from causal_probe import language_value_hook
from paired_env import PairedEnv
from screen import PUBLIC, run_episode, save
from smol_actor import SmolActor


CASES = ((2, "bottle"), (3, "bowl"))


def shared_suffix(intended, alternate, tokenizer):
    """Only shared suffix IDs after the changed noun; never align different nouns by index."""
    a, b = intended["tokens"], alternate["tokens"]
    pairs = []
    for i, j in zip(range(len(a) - 1, -1, -1), range(len(b) - 1, -1, -1)):
        if a[i] != b[j]:
            break
        if a[i] not in tokenizer.all_special_ids and tokenizer.decode([a[i]]).strip():
            pairs.append((i, j))
    pairs.reverse()
    if len(pairs) < 3:
        raise RuntimeError(f"Only {len(pairs)} matching content suffix tokens")
    return pairs


class RecipientContrast:
    def __init__(self, actor, tokenizer, alternate_instruction, layer, gain, cap, n_replans):
        self.actor = actor
        self.tokenizer = tokenizer
        self.alternate = alternate_instruction
        self.layer, self.gain, self.cap, self.n_replans = layer, gain, cap, n_replans
        if actor.policy.config.n_obs_steps != 1:
            raise RuntimeError("Counterfactual queries require a one-observation actor")

    def reset(self, seed):
        self.actor.reset(seed)
        self.step = 0
        self.chunk = None
        self.audit = []

    @torch.inference_mode()
    def act(self, obs, instruction):
        stride = self.actor.policy.config.n_action_steps
        index = self.step % stride
        if index == 0:
            self.replan(obs, instruction, stride)
        self.step += 1
        return self.chunk[index]

    def replan(self, obs, instruction, stride):
        target = self.actor.prepare(obs, instruction)
        noise = self.actor.make_noise()  # consumed once per actual replan
        replan = self.step // stride
        if replan >= self.n_replans:
            self.chunk = self.actor.predict_action_chunk(target, noise=noise)[0, :stride].cpu().numpy()
            return

        other = self.actor.prepare(obs, self.alternate)
        with language_value_hook(self.actor, self.layer) as intended:
            base = self.actor.predict_action_chunk(target, noise=noise.clone())
        with language_value_hook(self.actor, self.layer) as alternate:
            alt = self.actor.predict_action_chunk(other, noise=noise.clone())
        if replan == 0:
            stock = self.actor.predict_action_chunk(target, noise=noise.clone())
            with language_value_hook(self.actor, self.layer, intended):
                sham = self.actor.predict_action_chunk(target, noise=noise.clone())
            if not torch.equal(stock, base) or not torch.equal(base, sham):
                raise RuntimeError("First-replan capture/self-transplant must preserve the stock chunk exactly")

        pairs = shared_suffix(intended, alternate, self.tokenizer)
        rows_a, rows_b = zip(*pairs)
        v = intended["value"][:, list(rows_a), :].float()
        u = alternate["value"][:, list(rows_b), :].float()
        delta = (v - u) * self.gain
        ref_norm = float(torch.linalg.vector_norm(v))
        delta_norm = float(torch.linalg.vector_norm(delta))
        scale = min(1.0, self.cap * ref_norm / max(delta_norm, 1e-8))
        replacement = intended["value"].clone()
        replacement[:, list(rows_a), :] = (v + scale * delta).to(replacement)
        patch = {"tokens": intended["tokens"], "value": replacement}
        with language_value_hook(self.actor, self.layer, patch) as application:
            actions = self.actor.predict_action_chunk(target, noise=noise.clone())
        used_base = base[0, :stride]
        used_alt = alt[0, :stride]
        used_actions = actions[0, :stride]
        changed = replacement[:, list(rows_a), :].float() - v
        self.audit.append({
            "replan": replan, "step": self.step, "layer": self.layer,
            "matched_token_ids": [intended["tokens"][i] for i in rows_a],
            "matched_tokens": self.tokenizer.convert_ids_to_tokens([intended["tokens"][i] for i in rows_a]),
            "intended_token_indices": list(rows_a), "alternate_token_indices": list(rows_b),
            "nominal_gain": self.gain, "effective_gain": self.gain * scale,
            "relative_value_delta_l2": float(torch.linalg.vector_norm(changed)) / max(ref_norm, 1e-8),
            "projection_calls": application["calls"],
            "alternate_action_rms": float((used_alt - used_base).square().mean().sqrt()),
            "patched_action_rms": float((used_actions - used_base).square().mean().sqrt()),
            "first_replan_sham_bitwise_equal": True if replan == 0 else None,
            "first_replan_base_actions": used_base.cpu().tolist() if replan == 0 else None,
        })
        self.chunk = used_actions.cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--render-gpu", type=int, default=3)
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--gain", type=float, default=0.35)
    p.add_argument("--cap", type=float, default=0.20)
    p.add_argument("--replans", type=int, default=16)
    args = p.parse_args()
    if args.layer < 1 or args.layer % 2 == 0 or not 0 < args.cap <= 1 or args.replans <= 0:
        raise ValueError("Use an odd cross-attention layer, finite positive cap and positive replans")
    args.run.mkdir()
    config = {"layer": args.layer, "gain": args.gain, "cap": args.cap, "early_replans": args.replans,
              "cases": CASES, "n_action_steps": 10, "same_recipient_vision_and_state": True,
              "same_noise_for_intended_alternate_and_intervention": True,
              "alignment": "common content suffix after changed noun, exact token IDs; position offsets may differ"}
    save(args.run / "plan.json", config)
    actor = SmolActor(n_action_steps=10)
    tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parent / "models/smolvlm2", local_files_only=True)
    results = []
    with PairedEnv(pair="objects", camera_size=256, render_gpu_device_id=args.render_gpu) as env:
        for init, goal in CASES:
            target = f"init-{init:03d}-{goal}"
            old = args.screen / "swap_xy" / target
            baseline = json.loads((old / "result.json").read_text())
            if not baseline["wrong_object_lifted_first"] or baseline["instruction"] != env.instructions[goal]:
                raise RuntimeError("Require the known wrong-lift case with the exact native policy prompt")
            alternate = "bowl" if goal == "bottle" else "bottle"
            control = RecipientContrast(actor, tokenizer, env.instructions[alternate], args.layer,
                                        args.gain, args.cap, args.replans)
            row, obs, state = run_episode(env, control, goal, init, baseline["seed"], 300, args.run,
                                          video=True, init_source="bowl", layout="swap_xy")
            with np.load(old / "initial-public.npz", allow_pickle=False) as expected:
                if not np.array_equal(state, expected["sim_state"]) or any(not np.array_equal(obs[k], expected[k]) for k in PUBLIC):
                    raise RuntimeError("Recipient's native initial state/camera differs from its recorded baseline")
            baseline_trace = json.loads((old / "trace.json").read_text())
            stock_actions = np.asarray([r["action"] for r in baseline_trace[:10]])
            stock_error = float(np.max(np.abs(stock_actions - np.asarray(control.audit[0]["first_replan_base_actions"]))))
            if stock_error > 1e-6:
                raise RuntimeError(f"First stock chunk differs from recorded baseline: {stock_error}")
            audit = {"baseline": baseline, "candidate": row, "baseline_initial_state_bitwise_equal": True,
                     "baseline_first_stock_action_max_abs_error": stock_error,
                     "alternate_instruction": env.instructions[alternate], "replans": control.audit}
            save(args.run / target / "contrast.json", audit)
            results.append(audit)
            save(args.run / "summary.json", {"config": config, "cases": results})
            print(json.dumps({"baseline": baseline, "candidate": row, "stock_error": stock_error}), flush=True)


if __name__ == "__main__":
    main()
