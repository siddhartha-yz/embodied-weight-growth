"""Test one fixed-time prompt switch after the bottle acquisition window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_env import PairedEnv
from screen import PUBLIC, run_episode, save
from smol_actor import SmolActor


SWITCH_AFTER = 80


class SwitchAtReplan:
    def __init__(self, actor, first_prompt):
        self.actor = actor
        self.first_prompt = first_prompt

    def reset(self, seed):
        self.actor.reset(seed)
        self.steps = 0
        self.switched = False

    def act(self, obs, native_instruction):
        if self.steps == SWITCH_AFTER:
            if self.actor.queued_actions:
                raise RuntimeError("The switch would not coincide with a natural replan")
            self.switched = True
        prompt = self.first_prompt if self.steps < SWITCH_AFTER else native_instruction
        self.steps += 1
        return self.actor.act(obs, prompt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--render-gpu", type=int, default=4)
    args = p.parse_args()
    args.run.mkdir()
    ref = json.loads((args.reference / "result.json").read_text())
    assert (ref["variant"], ref["init_index"], ref["goal"], ref["seed"], ref["layout"]) == (
        "pick-then-place", 2, "bottle", 1002, "swap_xy")
    actor = SwitchAtReplan(SmolActor(n_action_steps=10), ref["policy_instruction"])
    with PairedEnv(pair="objects", camera_size=256, render_gpu_device_id=args.render_gpu) as env:
        row, obs, state = run_episode(env, actor, ref["goal"], ref["init_index"], ref["seed"],
                                      300, args.run, video=True, init_source="bowl", layout=ref["layout"])
    with np.load(args.reference / "initial-public.npz", allow_pickle=False) as initial:
        if not np.array_equal(state, initial["sim_state"]) or any(
            not np.array_equal(obs[k], initial[k]) for k in PUBLIC
        ):
            raise RuntimeError("Initial state or observation differed from the prompt-only reference")
    out = args.run / "init-002-bottle"
    prior = json.loads((args.reference / "trace.json").read_text())
    trace = json.loads((out / "trace.json").read_text())
    if not actor.switched or trace[:SWITCH_AFTER] != prior[:SWITCH_AFTER]:
        raise RuntimeError("The pre-switch trace did not reproduce the prompt-only reference exactly")
    row.update({"first_prompt": ref["policy_instruction"], "second_prompt": row["instruction"],
                "switch_before_control": SWITCH_AFTER + 1, "identical_pre_switch_controls": SWITCH_AFTER,
                "same_initial_public_obs": True, "same_initial_sim_state": True})
    save(args.run / "summary.json", row)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
