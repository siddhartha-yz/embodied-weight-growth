"""Run editor-chosen speed-command probes on the fixed native H1 plant."""
import argparse
import math
from pathlib import Path


DT = 0.02
DEV_SEEDS = (271, 272, 273)
UNTIL = ("time", "left_off", "right_off", "left_touchdown", "right_touchdown")


def validate_plan(plan):
    """Validate the bounded development-only command API, without importing MuJoCo."""
    if not isinstance(plan, dict) or set(plan) != {"hypothesis", "readout", "trials"}:
        raise ValueError("Plan requires hypothesis, readout, and trials")
    for key in ("hypothesis", "readout"):
        if not isinstance(plan[key], str) or not plan[key].strip():
            raise ValueError(f"Plan {key} must be nonempty text")
    trials = plan["trials"]
    if not isinstance(trials, list) or not 1 <= len(trials) <= 3:
        raise ValueError("Plan requires 1..3 trials")
    for trial in trials:
        if not isinstance(trial, dict) or set(trial) != {"seed", "segments"}:
            raise ValueError("Each trial requires seed and segments")
        if type(trial["seed"]) is not int or trial["seed"] not in DEV_SEEDS:
            raise ValueError(f"Trial seed must be in {DEV_SEEDS}")
        segments = trial["segments"]
        if not isinstance(segments, list) or not 1 <= len(segments) <= 8:
            raise ValueError("Each trial requires 1..8 command segments")
        ticks = 0
        for segment in segments:
            if not isinstance(segment, dict) or set(segment) != {"seconds", "target_vx", "until"}:
                raise ValueError("Each segment requires seconds, target_vx, and until")
            seconds, speed, until = (segment[k] for k in ("seconds", "target_vx", "until"))
            if (
                type(seconds) not in (int, float)
                or not 0.1 <= seconds <= 20
                or not math.isclose(seconds / DT, round(seconds / DT), abs_tol=1e-8)
            ):
                raise ValueError("Segment seconds must be 0.10..20 on the 0.02-second grid")
            if type(speed) not in (int, float) or not 0 <= speed <= 0.7:
                raise ValueError("Segment target_vx must be 0..0.70 m/s")
            if until not in UNTIL:
                raise ValueError(f"Unknown segment ending; use one of {UNTIL}")
            ticks += round(seconds / DT)
        if ticks > round(20 / DT):
            raise ValueError("Segment time caps must sum to at most 20 seconds per trial")
    return plan


def contact_edges(before, after):
    return [
        f"{side}_{'touchdown' if new else 'off'}"
        for side, old, new in zip(("left", "right"), before, after)
        if old != new
    ]


def t_seconds(tick):
    return round(tick * DT, 2)


def compact_timeline(trace, segments, touchdown, upright, channels):
    n = len(trace) - 1
    # Keep segment boundaries and the physical loss of balance, then spend the
    # remaining sample budget on early motion and observed landings.
    points = [0, n, upright]
    points += [round(s["end_s"] / DT) for s in segments]
    points += [e["step"] for e in touchdown[:3]]
    points += [5, 12, 25, 38, 50, 75, 100, 150, 250, 500, 750]
    pick = sorted(list(dict.fromkeys(i for i in points if i <= n))[:14])
    legs = [(i, c) for i, c in enumerate(channels) if any(k in c["joint"] for k in ("hip", "knee", "ankle"))]
    r3 = lambda values: [round(v, 3) for v in values]
    samples = []
    for i in pick:
        row = trace[i]
        q, v, action = row["qpos"], row["qvel"], row["action"]
        feet = row["foot_xyz_m"]
        samples.append([
            t_seconds(i), row["target_vx"], r3(q[:3]), r3(v[:3]), r3(row["root_rpy"]),
            row["foot_contact"], [round(f, 1) for f in row["foot_normal_n"]],
            [r3((f[0] - q[0], f[1] - q[1], f[2])) for f in feet],
            r3(q[c["qpos"]] for _, c in legs),
            None if action is None else r3(action[j] for j, _ in legs),
        ])
    return {
        "columns": ["t_s", "target_vx_mps", "pelvis_xyz_m", "pelvis_vxyz_mps", "pelvis_rpy_rad",
                    "foot_contact_LR", "foot_normal_N_LR", "foot_relative_xy_world_z_m_LR",
                    "leg_joints_rad", "native_leg_targets"],
        "leg_joint_order": [c["joint"] for _, c in legs],
        "samples": samples,
    }


def rollout_trial(env, policy, trial, channels):
    import numpy as np
    import plant

    tilt = plant.reset_pose(env, trial["seed"])
    sensors = plant.Sensors(env.unwrapped)
    observation, state = sensors.sample(trial["segments"][0]["target_vx"])
    trace = [dict(step=0, t_s=0., native_reward=None, action=None, **state)]
    executed, observed_edges = [], []
    reason = "horizon"
    for index, segment in enumerate(trial["segments"], start=1):
        start = len(trace) - 1
        cmd, until = segment["target_vx"], segment["until"]
        cap = round(segment["seconds"] / DT)
        ended = "time" if until == "time" else "timeout"
        for offset in range(1, cap + 1):
            observation[plant.NQ - 2 + plant.NV] = cmd
            action = np.asarray(policy.act(observation), dtype=np.float64)
            if action.shape != (plant.NU,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                raise ValueError("Checkpoint must produce 19 finite normalized joint targets")
            _, reward, terminated, truncated, _ = env.step(action)
            observation, state = sensors.sample(cmd)
            if not np.isfinite(observation).all() or not math.isfinite(reward):
                reason = ended = "nonfinite_native_transition"
                break
            step = start + offset
            edges = contact_edges(trace[-1]["foot_contact"], state["foot_contact"])
            trace.append(dict(step=step, t_s=t_seconds(step), native_reward=float(reward),
                              action=action.tolist(), **state))
            for edge in edges:
                if len(observed_edges) < 16:
                    observed_edges.append({"t_s": t_seconds(step), "edge": edge})
            hit = until in edges
            if terminated:
                reason = ended = "native_fall"
                break
            final_tick = index == len(trial["segments"]) and (hit or offset == cap)
            if truncated and not final_tick:
                reason = ended = "native_early_truncation"
                break
            if hit:
                ended = until
                break
        executed.append(dict(segment=index, start_s=t_seconds(start), end_s=t_seconds(len(trace) - 1),
                             target_vx=cmd, until=until, ended_by=ended))
        if reason != "horizon":
            break

    touchdown = plant.find_touchdowns(trace)
    goals = [r["target_vx"] for r in trace[1:]]
    # Use the original balance/footfall diagnostics without assigning a gate to
    # a probe. The predefined walk-test and stand-test scores are separate.
    metrics = plant.score({}, [], goals, trace, reason, touchdown)
    del metrics["passed"], metrics["segments"]
    upright = round(metrics["upright_prefix_s"] / DT)
    metrics.update(
        seed=trial["seed"],
        reset_tilt_rad=tilt,
        segments_executed=executed,
        contact_edges_first=observed_edges,
        qualified_touchdowns=touchdown,
        timeline=compact_timeline(trace, executed, touchdown, upright, channels),
    )
    return metrics, trace


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    plan = validate_plan(json.loads(args.plan.read_text()))

    import plant
    from weight_policy import Policy

    if plant.DT != DT:
        raise RuntimeError("Probe and native H1 step intervals differ")
    args.out.mkdir(parents=True, exist_ok=True)
    env = plant.native_env()
    try:
        channels = plant.channel_info(env)
        policy = Policy.load(args.weights, plant.FEATURE_NAMES, tuple(c["joint"] for c in channels))
        cases = []
        for index, trial in enumerate(plan["trials"], start=1):
            ident = f"trial-{index}"
            metrics, trace = rollout_trial(env, policy, trial, channels)
            trace_file = f"{ident}.trace.json"
            plant.write(args.out / trace_file, trace)
            cases.append(dict(case_id=ident, trace=trace_file, **metrics))
        plant.write(args.out / "summary.json", {"plan": plan, "cases": cases})
    finally:
        env.close()


if __name__ == "__main__":
    main()
