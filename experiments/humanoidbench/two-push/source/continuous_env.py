"""Continuous H1 physics, with prescribed host-side pulses and offline rendering."""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import numpy as np

from push_env import (
    DT, N_ACT, N_OBS, PULSE_STEPS,
    _borrow_native_env, _foot_support, _physically_down, _recovery, _wrap_angle, _yaw,
    reusable_native_env,
)


@dataclass(frozen=True)
class Pulse:
    """A world-Y impulse on the torso, distributed over five native ticks."""

    push_step: int
    lateral_impulse_ns: float

    def __post_init__(self):
        if type(self.push_step) is not int or self.push_step < 0:
            raise ValueError("Pulse step must be a nonnegative integer")
        if not math.isfinite(self.lateral_impulse_ns) or not self.lateral_impulse_ns:
            raise ValueError("Pulse impulse must be finite and nonzero")


@dataclass(frozen=True)
class Scenario:
    seed: int
    horizon_steps: int = 2600
    pulses: tuple[Pulse, ...] = ()

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Scenario seed must be a nonnegative integer")
        if type(self.horizon_steps) is not int or self.horizon_steps <= 0:
            raise ValueError("Horizon must be a positive number of native steps")
        pulses = tuple(item if isinstance(item, Pulse) else Pulse(**item) for item in self.pulses)
        last_end = 0
        direction = math.copysign(1.0, pulses[0].lateral_impulse_ns) if pulses else None
        for pulse in pulses:
            if pulse.push_step < last_end or pulse.push_step + PULSE_STEPS > self.horizon_steps:
                raise ValueError("Pulses must be ordered, nonoverlapping, and within the horizon")
            if math.copysign(1.0, pulse.lateral_impulse_ns) != direction:
                raise ValueError("All prescribed impulses must point to the same physical side")
            last_end = pulse.push_step + PULSE_STEPS
        object.__setattr__(self, "pulses", pulses)


def rollout(policy, scenario, *, realtime=False, trace_every=1):
    """One reset and a continuous H1 walk. Policy sees only `(obs, native_step)`.

    The native position controller, task reward, and fall termination are
    unchanged. Only Gym's 1000-step TimeLimit is replaced by `horizon_steps`.
    A trace row at step k+1 holds the observation after action k. With wall
    pacing enabled, late controls are audited, never skipped or slowed by video.
    """
    import mujoco

    if isinstance(scenario, dict):
        scenario = Scenario(**scenario)
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario or the equivalent dictionary")
    if type(trace_every) is not int or trace_every < 0:
        raise ValueError("trace_every must be a nonnegative integer")
    if not callable(policy):
        raise TypeError("policy must be callable")

    with _borrow_native_env() as env:
        native = env.unwrapped
        model, data = native.model, native.data
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if torso < 0:
            raise RuntimeError("Cannot find the native H1 torso")
        support = _foot_support(mujoco, model, data) if scenario.pulses else None
        obs, _ = env.reset(seed=scenario.seed)
        if obs.shape != (N_OBS,) or not np.isfinite(obs).all():
            raise RuntimeError("Unexpected native H1 observation")
        trajectory = [obs.copy()]
        trace = [{"step": 0, "time_s": 0.0, "wall_s": 0.0,
                  "obs": obs.tolist(), "action": None, "reward": None}] if trace_every else []
        pulse_records = [{**asdict(pulse), "applied_steps": 0, "support_at_push": None, "onset_wall_s": None}
                         for pulse in scenario.pulses]
        pulse_index = 0
        pulse = scenario.pulses[0] if scenario.pulses else None
        reward_sum = 0.0
        fallen = native_truncated = False
        late_ticks = 0
        max_lag = 0.0
        wall_start = time.monotonic()
        for step in range(scenario.horizon_steps):
            if realtime:
                due = wall_start + step * DT
                remaining = due - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                lag = max(0.0, time.monotonic() - due)
                late_ticks += int(lag >= DT)
                max_lag = max(max_lag, lag)
            action = np.asarray(policy(obs.copy(), step), dtype=np.float64)
            if action.shape != (N_ACT,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
                raise ValueError("Policy must return 19 finite normalized targets in [-1,1]")

            active = pulse is not None and pulse.push_step <= step < pulse.push_step + PULSE_STEPS
            try:
                if active:
                    row = pulse_records[pulse_index]
                    if step == pulse.push_step:
                        row["support_at_push"] = support()
                        row["onset_wall_s"] = time.monotonic() - wall_start
                    data.xfrc_applied[torso, 1] = pulse.lateral_impulse_ns / (PULSE_STEPS * DT)
                    row["applied_steps"] += 1
                obs, reward, fallen, native_truncated, _ = native.step(action)
            finally:
                if active:
                    data.xfrc_applied[torso, 1] = 0.0

            if active and step + 1 == pulse.push_step + PULSE_STEPS:
                pulse_index += 1
                pulse = scenario.pulses[pulse_index] if pulse_index < len(scenario.pulses) else None
            if not np.isfinite(obs).all() or not math.isfinite(reward):
                raise RuntimeError("Nonfinite native H1 transition")
            reward_sum += float(reward)
            trajectory.append(obs.copy())
            steps = step + 1
            if trace_every and (steps % trace_every == 0 or fallen or native_truncated
                                or steps == scenario.horizon_steps):
                trace.append({"step": steps, "time_s": steps * DT, "wall_s": time.monotonic() - wall_start,
                              "obs": obs.tolist(), "action": action.tolist(), "reward": float(reward)})
            if fallen or native_truncated:
                break

        wall_duration = time.monotonic() - wall_start
        states = np.stack(trajectory)
        nq = int(model.nq)
        yaw = _yaw(states)
        completed = steps == scenario.horizon_steps and not (fallen or native_truncated)
        physically_down = _physically_down(states[1:, 2], fallen)
        # Recovery reads only each pulse's onset, including beyond 20 seconds.
        recovery = [_recovery(states, item, nq=nq) for item in scenario.pulses]
        metrics = {
            "steps": steps, "elapsed_s": steps * DT, "horizon_steps": scenario.horizon_steps,
            "fell": bool(fallen), "completed_horizon": completed, "physically_down": physically_down,
            "return": reward_sum, "mean_native_reward": reward_sum / steps,
            "x_m": float(states[-1, 0] - states[0, 0]), "y_m": float(states[-1, 1] - states[0, 1]),
            "min_root_z_m": float(states[:, 2].min()), "final_root_z_m": float(states[-1, 2]),
            "mean_vx_mps": float(states[1:, nq].mean()),
            "mean_abs_vy_mps": float(np.abs(states[1:, nq + 1]).mean()),
            "mean_abs_yaw_drift_deg": float(np.degrees(np.abs(_wrap_angle(yaw - yaw[0]))).mean()),
            "applied_pulse_steps": [row["applied_steps"] for row in pulse_records], "recovery": recovery,
            "recovered_and_finished": bool(completed and not physically_down and recovery
                                           and all(item["recovered_at_2s"] for item in recovery)),
        }
        audit = {"reset_count": 1, "pulses": pulse_records,
                 "support_at_pulses": [row["support_at_push"] for row in pulse_records],
                 "realtime": bool(realtime), "wall_start_monotonic_s": wall_start,
                 "elapsed_wall_s": wall_duration, "late_native_ticks": late_ticks,
                 "max_tick_lag_s": max_lag}
        serialized = {"seed": scenario.seed, "horizon_steps": scenario.horizon_steps,
                      "pulses": [asdict(pulse) for pulse in scenario.pulses]}
        return {"scenario": serialized, "metrics": metrics, "trace": trace, "audit": audit}


def render_trace(trace, video, *, camera="cam_default", width=256, height=256):
    """Render a dense recorded trajectory at 50 FPS; no rollout or policy call."""
    import imageio.v2 as iio
    import mujoco

    if not trace or any(row["step"] != i for i, row in enumerate(trace)):
        raise ValueError("Offline rendering needs a dense trace starting at step zero")
    states = np.asarray([row["obs"] for row in trace], dtype=np.float64)
    if states.shape != (len(trace), N_OBS) or not np.isfinite(states).all():
        raise ValueError("Trace observations must be finite native H1 states")
    output = Path(video)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        env = stack.enter_context(_borrow_native_env())
        model = env.unwrapped.model
        data = mujoco.MjData(model)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor < 0 or model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise RuntimeError("Offline render expects the native H1 collision plane")
        floor_size = model.geom_size[floor, :2].copy()
        stack.callback(model.geom_size[floor, :2].__setitem__, slice(None), floor_size)
        # The collision plane is infinite. Extend only the visible rectangle
        # for the recorded trajectory, without modifying the simulated rollout.
        model.geom_size[floor, :2] = np.maximum(floor_size, np.abs(states[:, :2]).max(axis=0) + 8)
        writer = stack.enter_context(iio.get_writer(str(output), fps=round(1 / DT), codec="libx264"))
        renderer = stack.enter_context(mujoco.Renderer(model, height=height, width=width))
        for row in states:
            data.qpos[:] = row[:model.nq]
            data.qvel[:] = row[model.nq:]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    return output
