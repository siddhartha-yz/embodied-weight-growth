"""Bounded JSONL interface to the drawing arm. Contains no drawing policy."""

from __future__ import annotations

import argparse
import json
import math
import sys

from plant import DT, Plant
from policy import compile_policy


START = (1.03, -1.75)
BOARD = {"x": [0.09, 0.245], "y": [-0.10, 0.105], "units": "m"}
MAX_MOVES = 18
MAX_TICKS = 1000
MAX_TICKS_PER_MOVE = 100
U_LIMIT = 0.8


def vector(value, name):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError(f"{name} must be two finite numbers")
    return [float(v) for v in value]


class Session:
    def __init__(self, arm, seed=271, condition="linear", *, plant=None):
        self.arm = arm
        self.plant = Plant(seed=seed, condition=condition) if plant is None else plant
        self.policy = None
        self.policy_version = 0
        self.attempt = 0
        self.moves_used = 0
        self.draw_moves = 0
        self.ticks_used = 0
        self.finished = False
        self.reset()

    def reset(self):
        self.attempt += 1
        self.state = {
            **self.plant.reset(q=START),
            "tip_v": [0.0, 0.0],
            "prev_u": [0.0, 0.0],
            "prev2_u": [0.0, 0.0],
        }
        self.pen = False
        self.strokes = []

    def snapshot(self, *, trace=None, samples=None, error=None):
        response = {
            "ok": error is None,
            "state": {k: list(v) for k, v in self.state.items()},
            "trace": [] if trace is None else trace,
            "strokes": [list(stroke) for stroke in self.strokes],
            "samples": [] if samples is None else samples,
            "attempt": self.attempt,
            "meters": {
                "moves_used": self.moves_used,
                "moves_left": MAX_MOVES - self.moves_used,
                "ticks_used": self.ticks_used,
                "ticks_left": MAX_TICKS - self.ticks_used,
                "max_ticks_per_move": MAX_TICKS_PER_MOVE,
                "draw_moves": self.draw_moves,
            },
            "board": BOARD,
            "dt": DT,
            "pen": self.pen,
            "policy_version": self.policy_version,
            "finished": self.finished,
        }
        if error is not None:
            response["error"] = error
        return response

    def step(self, u):
        new = self.plant.step(u)
        self.state = {
            **new,
            "tip_v": [(x - y) / DT for x, y in zip(new["tip"], self.state["tip"])],
            "prev_u": list(u),
            "prev2_u": self.state["prev_u"],
        }
        self.ticks_used += 1

    def sample(self, tick):
        return {"tick": tick, "u": self.state["prev_u"], **self.state}

    def motion(self, request, *, scripted):
        ticks = request.get("ticks")
        if type(ticks) is not int or not 1 <= ticks <= MAX_TICKS_PER_MOVE:
            raise ValueError(f"ticks must be an integer in [1, {MAX_TICKS_PER_MOVE}]")
        if self.moves_used >= MAX_MOVES or self.ticks_used + ticks > MAX_TICKS:
            raise ValueError("move or tick budget exceeded; see meters")

        if scripted:
            if self.policy is None:
                raise ValueError("install your policy before using it")
            goal = vector(request.get("goal"), "goal")
            if not all(lo <= value <= hi for value, (lo, hi) in zip(goal, (BOARD["x"], BOARD["y"]))):
                raise ValueError("goal must be on the paper; see board")
            pen = request.get("pen")
            if type(pen) is not bool:
                raise ValueError("use.pen must be true or false")
        else:
            u = vector(request.get("u"), "u")
            if any(abs(v) > U_LIMIT for v in u):
                raise ValueError(f"raw motor commands must be within ±{U_LIMIT}")
            if request.get("pen", False) is not False:
                raise ValueError("raw drive is pen-up only")
            pen = False

        self.moves_used += 1
        trace = [self.state["tip"]]
        samples = [self.sample(0)]
        error = None
        for tick in range(1, ticks + 1):
            if scripted:
                try:
                    obs = {k: list(v) for k, v in self.state.items()}
                    u = vector(self.policy(obs, list(goal)), "policy output")
                    if any(abs(v) > U_LIMIT for v in u):
                        raise ValueError(f"motor commands must be within ±{U_LIMIT}")
                except (ValueError, TypeError, ArithmeticError, RuntimeError, LookupError) as exc:
                    error = f"policy stopped after {tick - 1} ticks: {exc}"
                    break
            if tick == 1:
                if pen:
                    self.draw_moves += 1
                    if not self.pen:
                        self.strokes.append([self.state["tip"]])
                self.pen = pen
            self.step(u)
            tip = self.state["tip"]
            trace.append(tip)
            if pen and self.strokes[-1][-1] != tip:
                self.strokes[-1].append(tip)
            if tick <= 3 or tick % 6 == 0 or tick == ticks:
                samples.append(self.sample(tick))
        if samples[-1]["tick"] < len(trace) - 1:
            samples.append(self.sample(len(trace) - 1))
        return self.snapshot(trace=trace, samples=samples, error=error)

    def request(self, request):
        try:
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            if self.finished:
                raise ValueError("session has finished")
            op = request.get("op")
            if op in ("drive", "use"):
                return self.motion(request, scripted=op == "use")
            if op == "install":
                self.policy = compile_policy(self.arm, request.get("spec"))
                self.policy_version += 1
            elif op == "reset":
                self.reset()
            elif op == "finish":
                self.finished = True
            else:
                raise ValueError("unknown op; use drive, install, use, reset, or finish")
            return self.snapshot()
        except (ValueError, TypeError) as exc:
            return self.snapshot(error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("weights", "code"))
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--condition", choices=("linear", "backlash"), default="linear")
    args = parser.parse_args()
    session = Session(args.arm, args.seed, args.condition)
    print(json.dumps(session.snapshot(), allow_nan=False), flush=True)
    for line in sys.stdin:
        try:
            result = session.request(json.loads(line))
        except json.JSONDecodeError as exc:
            result = session.snapshot(error=f"invalid JSON: {exc.msg}")
        print(json.dumps(result, allow_nan=False, separators=(",", ":")), flush=True)
        if session.finished:
            break


if __name__ == "__main__":
    main()
