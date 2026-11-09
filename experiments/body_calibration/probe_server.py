"""Trusted MuJoCo process. Its stdin/stdout contain only bounded probe requests."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from plant import DT, HOME, Plant

MAX_STEPS = 48
MAX_EPISODES = 7


def probe(plant, start, segments):
    q = np.asarray(start, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all() or (np.abs(q) > 1.65).any():
        raise ValueError("start must contain two angles in [-1.65, 1.65]")
    if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_STEPS:
        raise ValueError(f"expected 1–{MAX_STEPS} segments")
    checked = []
    for segment in segments:
        if not isinstance(segment, list) or len(segment) != 3:
            raise ValueError("each segment is [motor0, motor1, ticks]")
        u = np.asarray(segment[:2], dtype=float)
        ticks = segment[2]
        if not np.isfinite(u).all() or (np.abs(u) > 0.8).any():
            raise ValueError("probe motor magnitude is limited to 0.8")
        if type(ticks) is not int or not 1 <= ticks <= 16:
            raise ValueError("ticks must be an integer in [1, 16]")
        checked.append((u.tolist(), ticks))
    if sum(t for _, t in checked) > MAX_STEPS:
        raise ValueError(f"episode exceeds {MAX_STEPS} ticks")
    initial = plant.reset(q)
    steps = []
    for u, ticks in checked:
        for _ in range(ticks):
            steps.append({"u": u, **plant.step(u)})
    return {"initial": initial, "steps": steps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--condition", choices=("linear", "backlash"), required=True)
    args = p.parse_args()
    plant = Plant(args.seed, args.condition)
    print(json.dumps({"dt": DT, "home": HOME, "max_steps": MAX_STEPS}), flush=True)
    used = 0
    for line in sys.stdin:
        try:
            data = json.loads(line)
            if used >= MAX_EPISODES:
                raise ValueError("probe budget used")
            result = probe(plant, data["start"], data["segments"])
            used += 1
            reply = {"episode": result, "remaining": MAX_EPISODES - used}
        except (ValueError, TypeError, KeyError) as exc:
            reply = {"error": str(exc)}
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
