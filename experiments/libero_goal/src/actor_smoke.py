"""Check actor preprocessing, fixed-noise chunk and queue on a native LIBERO scene."""
import argparse
import json
import time

import numpy as np

from paired_env import PairedEnv
from smol_actor import SmolActor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-gpu", type=int, default=-1)
    args = parser.parse_args()
    start = time.monotonic()
    actor = SmolActor()
    print("loaded exact checkpoint", flush=True)
    env = PairedEnv(camera_size=256, render_gpu_device_id=args.render_gpu)
    try:
        obs = env.reset("bowl")
        batch = actor.prepare(obs, env.language)
        actor.reset(0)
        chunk = actor.predict_action_chunk(batch, noise=actor.make_noise())
        actor.reset(0)
        first = actor.act(obs, env.language)
        assert actor.queued_actions == 49
        obs, *_ = env.step(first)
        second = actor.act(obs, env.language)
        assert actor.queued_actions == 48
        assert chunk.shape == (1, 50, 7) and first.shape == second.shape == (7,)
        assert np.isfinite(first).all() and np.isfinite(second).all()
        assert np.array_equal(chunk[0, 0].numpy(), first)
        assert np.array_equal(chunk[0, 1].numpy(), second)
        print(json.dumps({
            "chunk_shape": list(chunk.shape), "first": first.tolist(), "second": second.tolist(),
            "chunk_queue_exact": True, "wall_s": round(time.monotonic() - start, 3),
        }), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
