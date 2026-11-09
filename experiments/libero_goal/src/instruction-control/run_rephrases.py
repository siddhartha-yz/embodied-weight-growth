"""Rephrase only the instruction on the two crossed-layout wrong-target cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_env import PairedEnv
from screen import PUBLIC, run_episode, save
from smol_actor import SmolActor


CASES = ((2, "bottle"), (3, "bowl"))
NOUNS = {"bowl": "bowl", "bottle": "wine bottle"}
TEMPLATES = {
    "pick-then-place": "pick up the {target}, then put it on top of the cabinet",
    "explicit-exclusion": "pick up the {target}, not the {other}. put the {target} on top of the cabinet",
}


class Rephrase:
    def __init__(self, actor, native_instruction, policy_instruction):
        self.actor = actor
        self.native_instruction = native_instruction
        self.policy_instruction = policy_instruction

    def reset(self, seed):
        self.actor.reset(seed)

    def act(self, obs, instruction):
        if instruction != self.native_instruction:
            raise ValueError(f"Unexpected native instruction: {instruction!r}")
        return self.actor.act(obs, self.policy_instruction)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--render-gpu", type=int, default=4)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    args.run.mkdir()
    actor = SmolActor(n_action_steps=10)
    summary = {"max_steps": 300, "n_action_steps": actor.policy.config.n_action_steps,
               "layout": "swap_xy", "init_source": "bowl", "templates": TEMPLATES, "cases": []}
    with PairedEnv(pair="objects", camera_size=256, render_gpu_device_id=args.render_gpu) as env:
        for init, goal in CASES:
            other = next(name for name in NOUNS if name != goal)
            reference = args.reference / "swap_xy" / f"init-{init:03d}-{goal}"
            stock = json.loads((reference / "result.json").read_text())
            if not stock["wrong_object_lifted_first"] or stock["instruction"] != env.instructions[goal]:
                raise ValueError(f"Reference is not a confirmed wrong lift with the registry prompt: {reference}")
            result = {"stock": stock, "controls": []}
            summary["cases"].append(result)
            for name, template in TEMPLATES.items():
                prompt = template.format(target=NOUNS[goal], other=NOUNS[other])
                wrapped = Rephrase(actor, stock["instruction"], prompt)
                out = args.run / name
                out.mkdir(exist_ok=True)
                row, obs, state = run_episode(env, wrapped, goal, init, stock["seed"], 300, out,
                                              video=not args.no_video, init_source="bowl", layout="swap_xy")
                with np.load(reference / "initial-public.npz", allow_pickle=False) as initial:
                    if not np.array_equal(state, initial["sim_state"]) or any(
                        not np.array_equal(obs[k], initial[k]) for k in PUBLIC
                    ):
                        raise RuntimeError(f"Reset observations or state differ from stock: {reference}")
                row.update(variant=name, policy_instruction=prompt, same_initial_public_obs=True,
                           same_initial_sim_state=True)
                save(out / f"init-{init:03d}-{goal}" / "result.json", row)
                result["controls"].append(row)
                save(args.run / "summary.json", summary)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
