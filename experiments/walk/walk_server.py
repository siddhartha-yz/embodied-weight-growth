"""JSONL motor interface for learning to walk with feedback between segments."""

from __future__ import annotations

import argparse
import json
import math
import sys

from walk_plant import DT, PITCH, TARGET_V, WalkPlant
from walk_policy import JOINT_ORDER, N_MOTORS, compile_policy

MAX_MOVES = 24
MAX_TICKS = 1600
MAX_TICKS_PER_MOVE = 250
MIN_DISTANCE = 1.0
MIN_WALK_TIME = 2.0
MIN_STEPS = 4
MIN_FOOT_ADVANCE = 0.045
MIN_SWING_TIME = 0.06


def vector(value):
    if (not isinstance(value, (list, tuple)) or len(value) != N_MOTORS
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1.0 for v in value)):
        raise ValueError(f"u must be {N_MOTORS} finite normalized torques in [-1,1]")
    return list(map(float, value))


class Steps:
    """Measure forward swing touchdowns from geometry and simulated contact."""

    def __init__(self, state):
        self.swing = [None if contact else (self.foot_x(state, i), state["t"], bool(state["contact"][1-i]))
                      for i, contact in enumerate(state["contact"])]
        self.touchdowns = []
        self.alternating_steps = 0

    @staticmethod
    def foot_x(state, i):
        name = ("foot_r", "foot_l")[i]
        segment = next(p for p in state["pose"] if p["name"] == name)
        return (segment["start"][0] + segment["end"][0]) / 2

    def update(self, state):
        landed = []
        t = state["t"]
        for i, contact in enumerate(state["contact"]):
            x = self.foot_x(state, i)
            swing = self.swing[i]
            if not contact:
                if swing is None:
                    swing = (x, t, False)
                self.swing[i] = (swing[0], swing[1], swing[2] or bool(state["contact"][1-i]))
            elif swing is not None:
                old_x, old_t, supported = swing
                self.swing[i] = None
                if t - old_t >= MIN_SWING_TIME - 1e-9 and x - old_x >= MIN_FOOT_ADVANCE and supported:
                    landed.append({"foot": ("R", "L")[i], "time_s": t, "x_m": x,
                                   "advance_m": x-old_x, "swing_s": t-old_t})
        if len(landed) == 1:
            event = landed[0]
            if not self.touchdowns or event["foot"] != self.touchdowns[-1]["foot"]:
                self.alternating_steps += 1
            self.touchdowns.append(event)


class Session:
    def __init__(self, arm, seed=271, scene="train", *, frozen_spec=None, plant=None):
        self.arm = arm
        self.seed = seed
        self.scene = scene
        self.plant = WalkPlant(seed, scene) if plant is None else plant
        self.policy_frozen = frozen_spec is not None
        self.policy = compile_policy(arm, frozen_spec) if self.policy_frozen else None
        self.policy_version = int(self.policy_frozen)
        self.attempt = self.moves_used = self.ticks_used = 0
        self.finished = False
        self.reset()

    def reset(self):
        self.state = self.plant.reset()
        self.start_x = self.peak_x = self.state["x"]
        self.attempt += 1
        self.fall_reason = None
        self.first_success_s = None
        self.steps = Steps(self.state)

    @property
    def success(self):
        return (self.fall_reason is None and self.state["x"] - self.start_x >= MIN_DISTANCE
                and self.state["t"] >= MIN_WALK_TIME - 1e-9
                and self.steps.alternating_steps >= MIN_STEPS)

    def sample(self, tick, u=None):
        return {"tick": tick, **self.state, "u": u, "fall_reason": self.fall_reason,
                "alternating_steps": self.steps.alternating_steps, "success": self.success}

    def snapshot(self, *, trace=None, op_summary=None, error=None):
        trace = [] if trace is None else trace
        samples = [row for row in trace if row["tick"] <= 2 or row["tick"] % 10 == 0]
        if trace and trace[-1]["tick"] != samples[-1]["tick"]:
            samples.append(trace[-1])
        result = {
            "ok": error is None, "state": self.state, "trace": trace, "samples": samples,
            "op_summary": {} if op_summary is None else op_summary,
            "walk": {"distance_m": self.state["x"]-self.start_x, "elapsed_s": self.state["t"],
                     "peak_x_m": self.peak_x-self.start_x, "fall_reason": self.fall_reason,
                     "touchdowns": list(self.steps.touchdowns), "alternating_steps": self.steps.alternating_steps,
                     "first_success_s": self.first_success_s},
            "attempt": self.attempt, "policy_version": self.policy_version,
            "policy_frozen": self.policy_frozen, "finished": self.finished, "success": self.success,
            "dt": DT,
            "scene": {"name": self.scene, "initial_pitch_rad": PITCH[self.scene], "target_v_m_s": TARGET_V,
                      "joint_order": list(JOINT_ORDER), "minimum_distance_m": MIN_DISTANCE,
                      "minimum_time_s": MIN_WALK_TIME, "minimum_alternating_steps": MIN_STEPS},
            "meters": {
                "moves_used": self.moves_used, "moves_left": MAX_MOVES-self.moves_used,
                "ticks_used": self.ticks_used, "ticks_left": MAX_TICKS-self.ticks_used,
                "max_ticks_per_move": MAX_TICKS_PER_MOVE,
            },
        }
        if error is not None:
            result["error"] = error
        return result

    def motion(self, request, *, run):
        ticks = request.get("ticks")
        if type(ticks) is not int or not 1 <= ticks <= MAX_TICKS_PER_MOVE:
            raise ValueError(f"ticks must be an integer in [1, {MAX_TICKS_PER_MOVE}]")
        if self.moves_used >= MAX_MOVES or self.ticks_used + ticks > MAX_TICKS:
            raise ValueError("motion budget exceeded; see meters")
        if self.fall_reason is not None:
            raise ValueError("walker fell; reset before moving again")
        if run:
            if self.policy is None:
                raise ValueError("install your controller before running it")
        else:
            u = vector(request.get("u"))

        self.moves_used += 1
        trace = [self.sample(0)]
        before_x = self.state["x"]
        before_n = len(self.steps.touchdowns)
        error = None
        for tick in range(1, ticks + 1):
            if run:
                try:
                    u = self.policy(self.state, TARGET_V)
                except (ValueError, TypeError, ArithmeticError, RuntimeError, LookupError) as exc:
                    error = f"policy stopped after {tick-1} ticks: {exc}"
                    break
            self.state = self.plant.step(u)
            self.ticks_used += 1
            self.peak_x = max(self.peak_x, self.state["x"])
            self.fall_reason = self.plant.fall_reason(self.state)
            if self.fall_reason is None:
                self.steps.update(self.state)
                if self.first_success_s is None and self.success:
                    self.first_success_s = self.state["t"]
            trace.append(self.sample(tick, u))
            if self.fall_reason is not None:
                break
        summary = {
            "ticks": len(trace)-1, "displacement_m": self.state["x"]-before_x,
            "new_touchdowns": self.steps.touchdowns[before_n:], "fall_reason": self.fall_reason,
        }
        return self.snapshot(trace=trace, op_summary=summary, error=error)

    def request(self, request):
        try:
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            if self.finished:
                raise ValueError("session has finished")
            op = request.get("op")
            if op in ("drive", "run"):
                if self.policy_frozen and op == "drive":
                    raise ValueError("a frozen test permits movement only through its installed controller")
                return self.motion(request, run=op == "run")
            if op == "install":
                if self.policy_frozen:
                    raise ValueError("the controller is frozen; installing is disabled")
                self.policy = compile_policy(self.arm, request.get("spec"))
                self.policy_version += 1
            elif op == "reset":
                self.reset()
            elif op == "finish":
                self.finished = True
            else:
                raise ValueError("unknown op; use drive, install, run, reset, or finish")
            return self.snapshot()
        except (ValueError, TypeError, OverflowError) as exc:
            return self.snapshot(error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("weights", "code"))
    parser.add_argument("--scene", default="train", choices=tuple(PITCH))
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--frozen-spec-stdin", action="store_true")
    args = parser.parse_args()
    spec = json.loads(sys.stdin.readline()) if args.frozen_spec_stdin else None
    if args.frozen_spec_stdin and spec is None:
        parser.error("frozen artifact must be a code string or numeric layers object")
    session = Session(args.arm, args.seed, args.scene, frozen_spec=spec)
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
