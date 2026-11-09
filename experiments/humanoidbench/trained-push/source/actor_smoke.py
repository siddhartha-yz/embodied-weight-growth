"""Validate the official DR.Q checkpoint on unmodified HumanoidBench H1."""
import argparse
import json

import gymnasium as gym
import humanoid_bench  # Register tasks.
import numpy as np
import torch

from actor_source import ACTOR_REPEAT, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 271))
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor = load(args.bundle)
    a0, h0 = actor.forward(np.zeros(51, dtype=np.float32))
    _, h1 = actor.forward(np.zeros(51, dtype=np.float32), mute=(0, 1, 2))
    assert a0.shape == (19,) and h0.shape == (512,) and np.all(h1[:3] == 0)
    for seed in args.seeds:
        env = gym.make("h1-walk-v0")
        obs, _ = env.reset(seed=seed)
        task = env.unwrapped
        x0 = float(task.data.qpos[0])
        total = 0.0
        for tick in range(env.spec.max_episode_steps):
            if tick % ACTOR_REPEAT == 0:
                action, _ = actor.forward(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        print(json.dumps({
            "seed": seed, "native_steps": tick + 1, "elapsed_s": round(task.data.time, 6),
            "forward_m": float(task.data.qpos[0]) - x0, "pelvis_z_m": float(task.data.qpos[2]),
            "head_z_m": float(task.robot.head_height()), "return": total,
            "terminated": bool(terminated), "truncated": bool(truncated),
        }), flush=True)
        env.close()


if __name__ == "__main__":
    main()
