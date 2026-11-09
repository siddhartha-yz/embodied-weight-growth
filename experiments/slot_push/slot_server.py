"""Bounded JSONL motor interface for the slot push; no waypoint or controller."""

from __future__ import annotations

import argparse
import json
import math
import sys

from slot_plant import BOUNDS, DT, START_ANGLE, SlotPlant, in_slot
from slot_policy import compile_policy

MAX_MOVES = 36
MAX_TICKS = 2400
MAX_TICKS_PER_MOVE = 100
U_LIMIT = 0.8
EMPTY_CONTACT = {
    "tip_block_ticks": 0, "block_wall_ticks": 0, "tip_wall_ticks": 0,
    "block_displacement_m": 0.0, "block_rotation_deg": 0.0, "stall": False,
}


def vector(value, name):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError(f"{name} must be two finite numbers")
    return list(map(float, value))


class Session:
    def __init__(self, arm, seed=271, scene="train", *, frozen_spec=None, plant=None):
        self.arm = arm
        self.plant = SlotPlant(seed, scene) if plant is None else plant
        self.frozen = frozen_spec is not None
        self.policy = compile_policy(arm, frozen_spec) if self.frozen else None
        self.policy_version = int(self.frozen)
        self.attempt = self.moves_used = self.ticks_used = 0
        self.finished = False
        self.reset()

    def reset(self):
        self.attempt += 1
        self.state = {**self.plant.reset(), "tip_v": [0.0, 0.0],
                      "prev_u": [0.0, 0.0], "prev2_u": [0.0, 0.0]}

    def snapshot(self, *, trace=None, contact_summary=None, error=None):
        trace = [] if trace is None else trace
        samples = [row for row in trace if row["tick"] <= 2 or row["tick"] % 10 == 0]
        if trace and trace[-1]["tick"] != samples[-1]["tick"]:
            samples.append(trace[-1])
        result = {
            "ok": error is None, "state": self.state, "trace": trace, "samples": samples,
            "contact_summary": EMPTY_CONTACT.copy() if contact_summary is None else contact_summary,
            "attempt": self.attempt, "policy_version": self.policy_version,
            "policy_frozen": self.frozen, "finished": self.finished, "success": in_slot(self.state),
            "dt": DT, "scene": self.plant.geometry(),
            "meters": {
                "moves_used": self.moves_used, "moves_left": MAX_MOVES-self.moves_used,
                "ticks_used": self.ticks_used, "ticks_left": MAX_TICKS-self.ticks_used,
                "max_ticks_per_move": MAX_TICKS_PER_MOVE,
            },
        }
        if error is not None:
            result["error"] = error
        return result

    def step(self, u):
        new = self.plant.step(u)
        self.state = {**new, "tip_v": [(x-y)/DT for x,y in zip(new["tip"], self.state["tip"])],
                      "prev_u": list(u), "prev2_u": self.state["prev_u"]}
        self.ticks_used += 1

    def motion(self, request, *, use):
        ticks = request.get("ticks")
        if type(ticks) is not int or not 1 <= ticks <= MAX_TICKS_PER_MOVE:
            raise ValueError(f"ticks must be an integer in [1, {MAX_TICKS_PER_MOVE}]")
        if self.moves_used >= MAX_MOVES or self.ticks_used + ticks > MAX_TICKS:
            raise ValueError("move or tick budget exceeded; see meters")
        if use:
            if self.policy is None:
                raise ValueError("install your policy before using it")
            goal = vector(request.get("goal"), "goal")
            if not all(lo <= value <= hi for value,(lo,hi) in zip(goal,(BOUNDS["x"],BOUNDS["y"]))):
                raise ValueError("goal must be inside the scene coordinate bounds")
        else:
            u = vector(request.get("u"), "u")
            if any(abs(v) > U_LIMIT for v in u):
                raise ValueError(f"raw motor commands must be within ±{U_LIMIT}")
        self.moves_used += 1
        trace = [{"tick": 0, **self.state}]
        summary = EMPTY_CONTACT.copy()
        error = None
        for tick in range(1, ticks+1):
            if use:
                try:
                    u = self.policy(self.state, goal)
                except (ValueError, TypeError, ArithmeticError, RuntimeError, LookupError) as exc:
                    error = f"policy stopped after {tick-1} ticks: {exc}"
                    break
            self.step(u)
            summary["tip_block_ticks"] += int(self.state["contact"][0])
            summary["block_wall_ticks"] += int(self.state["contact"][1])
            summary["tip_wall_ticks"] += int(self.plant.last_tip_wall)
            trace.append({"tick": tick, **self.state})
        summary["block_displacement_m"] = math.dist(trace[0]["block"], self.state["block"])
        angle_start = math.atan2(*trace[0]["block_angle"][::-1])
        angle_end = math.atan2(*self.state["block_angle"][::-1])
        summary["block_rotation_deg"] = math.degrees(math.remainder(angle_end-angle_start, math.tau))
        # A descriptive stall label, not a rule for what action to take.
        summary["stall"] = (
            summary["block_wall_ticks"] >= max(5, (len(trace)-1)//4)
            and summary["tip_block_ticks"] >= max(5, (len(trace)-1)//4)
            and summary["block_displacement_m"] < .002
            and abs(summary["block_rotation_deg"]) < 3.0
        )
        return self.snapshot(trace=trace, contact_summary=summary, error=error)

    def request(self, request):
        try:
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            if self.finished:
                raise ValueError("session has finished")
            op = request.get("op")
            if op in ("drive", "use"):
                if self.frozen and op == "drive":
                    raise ValueError("this transfer session permits movement only through its frozen module")
                return self.motion(request, use=op == "use")
            if op == "install":
                if self.frozen:
                    raise ValueError("this transfer session uses a frozen module; installs are disabled")
                self.policy = compile_policy(self.arm, request.get("spec"))
                self.policy_version += 1
            elif op == "reset":
                self.reset()
            elif op == "finish":
                self.finished = True
            else:
                raise ValueError("unknown op; use drive, install, use, reset, or finish")
            return self.snapshot()
        except (ValueError, TypeError, OverflowError) as exc:
            return self.snapshot(error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("weights", "code"))
    parser.add_argument("--scene", required=True, choices=tuple(START_ANGLE))
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--frozen-spec-stdin", action="store_true")
    args = parser.parse_args()
    spec = json.loads(sys.stdin.readline()) if args.frozen_spec_stdin else None
    if args.frozen_spec_stdin and spec is None:
        parser.error("frozen artifact must be a code string or a numeric layer object")
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
