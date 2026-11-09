"""Native HumanoidBench H1 Walk with a brief, unobserved lateral force pulse."""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

import numpy as np


ENV_ID = "h1-walk-v0"
DT = 0.02
HORIZON = 1000
N_OBS, N_ACT = 51, 19
PULSE_STEPS = 5
PRE_STEPS, RECOVERY_STEPS, RECOVERY_AVERAGE_STEPS = 20, 100, 15
SUPPORT_LABELS = ("neither", "left_only", "right_only", "both")
DOWN_HEIGHT_M, DOWN_STEPS = 0.60, 10  # 200 ms at native control rate
_REUSABLE_ENV = None
_BORROWED = False


@dataclass(frozen=True)
class Scenario:
    """Pulse begins before native control `push_step` (zero-based); +Y is positive."""

    seed: int
    push_step: int | None = None
    lateral_impulse_ns: float = 0.0

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Scenario seed must be a nonnegative integer")
        if not math.isfinite(self.lateral_impulse_ns):
            raise ValueError("Impulse must be finite")
        if self.push_step is None:
            if self.lateral_impulse_ns:
                raise ValueError("A nonzero impulse needs a push_step")
        elif (type(self.push_step) is not int or not 0 <= self.push_step <= HORIZON - PULSE_STEPS
              or not self.lateral_impulse_ns):
            raise ValueError("A push needs a nonzero impulse and an in-episode step")


def _make_env():
    import gymnasium as gym
    import humanoid_bench  # registers the native task

    env = gym.make(ENV_ID, render_mode=None)
    native = env.unwrapped
    if (env.spec.max_episode_steps != HORIZON or env.action_space.shape != (N_ACT,)
            or native.model.nq + native.model.nv != N_OBS or not math.isclose(native.dt, DT)):
        env.close()
        raise RuntimeError("Staged environment does not match native 50 Hz H1 Walk")
    return env


@contextmanager
def reusable_native_env():
    """Reuse one native model for a serial batch of independently seeded episodes."""
    global _REUSABLE_ENV
    if _REUSABLE_ENV is not None:
        yield
        return
    env = _make_env()
    _REUSABLE_ENV = env
    try:
        yield
    finally:
        _REUSABLE_ENV = None
        env.close()


@contextmanager
def _borrow_native_env():
    global _BORROWED
    env = _REUSABLE_ENV
    if env is None:
        env = _make_env()
        try:
            yield env
        finally:
            env.close()
        return
    if _BORROWED:
        raise RuntimeError("A reusable H1 environment supports serial rollouts only")
    _BORROWED = True
    try:
        yield env
    finally:
        _BORROWED = False


def describe():
    """Static controller I/O metadata; no privileged state."""
    import mujoco

    with _borrow_native_env() as env:
        native = env.unwrapped
        model = native.model
        channels = []
        for i, joint_id in enumerate(model.actuator_trnid[:, 0]):
            joint_id = int(joint_id)
            channels.append({
                "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                "qpos": int(model.jnt_qposadr[joint_id]), "qvel": int(model.jnt_dofadr[joint_id]),
                "min_rad": float(native.action_low[i]), "max_rad": float(native.action_high[i]),
            })
        return {"task": ENV_ID, "dt_s": DT, "horizon": HORIZON,
                "qpos_size": int(model.nq), "qvel_size": int(model.nv), "channels": channels}


def _wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _yaw(obs):
    w, x, y, z = obs[:, 3:7].T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _average_yaw(yaw):
    return math.atan2(float(np.sin(yaw).mean()), float(np.cos(yaw).mean()))


def _physically_down(root_heights, fell):
    low = root_heights < DOWN_HEIGHT_M
    return bool(fell or low[-1] or (len(low) >= DOWN_STEPS and np.any(
        np.convolve(low.astype(np.int8), np.ones(DOWN_STEPS, dtype=np.int8), mode="valid") == DOWN_STEPS)))


def _recovery(obs, scenario, *, nq):
    if scenario.push_step is None:
        return None
    start = scenario.push_step
    pulse_end = start + PULSE_STEPS
    target = pulse_end + RECOVERY_STEPS
    report = {"recovered_at_2s": False, "observable_window": False,
              "pulse_end_s": pulse_end * DT, "target_s": target * DT}
    if start + 1 < PRE_STEPS or target >= len(obs):
        return report

    before = obs[start + 1 - PRE_STEPS:start + 1]
    after = obs[target + 1 - RECOVERY_AVERAGE_STEPS:target + 1]
    ref_v = before[:, nq:nq + 2].mean(axis=0)
    post_v = after[:, nq:nq + 2].mean(axis=0)
    yaw_deg = math.degrees(float(_wrap_angle(_average_yaw(_yaw(after)) - _average_yaw(_yaw(before)))))
    # Extrapolate the pre-push lateral drift; an offset is reported, but a
    # stable parallel path can still count as balance recovery.
    lat_offset = float(obs[target, 1] - (obs[start, 1] + (target - start) * DT * ref_v[1]))
    speed_error, lateral_error = (float(x) for x in post_v - ref_v)
    report.update(observable_window=True, before_vx_mps=float(ref_v[0]), before_vy_mps=float(ref_v[1]),
                  after_vx_mps=float(post_v[0]), after_vy_mps=float(post_v[1]),
                  vx_error_mps=speed_error, vy_error_mps=lateral_error, yaw_error_deg=yaw_deg,
                  lateral_offset_m=lat_offset,
                  recovered_at_2s=bool(ref_v[0] >= 0.5 and abs(speed_error) <= 0.2
                                        and abs(lateral_error) <= 0.15 and abs(yaw_deg) <= 10.0))
    return report


def _foot_support(mujoco, model, data):
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_bodies = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                        for name in ("left_ankle_link", "right_ankle_link"))
    if floor < 0 or min(foot_bodies) < 0:
        raise RuntimeError("Cannot find native H1 floor or foot collision bodies")
    foot_index = {body: i for i, body in enumerate(foot_bodies)}
    wrench = np.empty(6)

    def read():
        normal = [0.0, 0.0]
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.efc_address < 0:
                continue
            if contact.geom1 == floor:
                other = contact.geom2
            elif contact.geom2 == floor:
                other = contact.geom1
            else:
                continue
            foot = foot_index.get(model.geom_bodyid[other])
            if foot is not None:
                mujoco.mj_contactForce(model, data, i, wrench)
                normal[foot] += wrench[0]
        # Only load-bearing contacts, not hovering feet or collision margins.
        return SUPPORT_LABELS[int(normal[0] > 5.0) + 2 * int(normal[1] > 5.0)]

    return read


def rollout(policy, scenario, *, video=None, trace_every=5, max_steps=None, record_support=False):
    """Run a fresh native episode; `policy` sees only one detached 51-vector per tick.

    Trace contains only observations and submitted actions, not force or hidden
    sensors. `trace_every=0` omits it. `max_steps` is for smoke checks only. Pass
    a fresh callable per rollout if a policy maintains its own state. Contact
    audits are returned separately and never sent to `policy`.
    """
    import mujoco

    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if type(trace_every) is not int or trace_every < 0:
        raise ValueError("trace_every must be a nonnegative integer")
    if max_steps is not None and (type(max_steps) is not int or not 1 <= max_steps <= HORIZON):
        raise ValueError("max_steps must be between 1 and the native horizon")
    with ExitStack() as stack:
        env = stack.enter_context(_borrow_native_env())
        native = env.unwrapped
        model, data = native.model, native.data
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if torso < 0:
            raise RuntimeError("Cannot find torso_link in the native H1 model")
        support = _foot_support(mujoco, model, data) if record_support or scenario.push_step is not None else None
        obs, _ = env.reset(seed=scenario.seed)
        if obs.shape != (N_OBS,) or not np.isfinite(obs).all():
            raise RuntimeError("Unexpected H1 initial observation")
        writer = renderer = None
        if video is not None:
            import imageio.v2 as iio

            output = Path(video)
            output.parent.mkdir(parents=True, exist_ok=True)
            writer = stack.enter_context(iio.get_writer(str(output), fps=round(1 / DT), codec="libx264"))
            renderer = stack.enter_context(mujoco.Renderer(model, height=256, width=256))

        def frame():
            if renderer is not None:
                renderer.update_scene(data, camera="cam_default")
                writer.append_data(renderer.render())

        trajectory = [obs.copy()]
        trace = [{"step": 0, "time_s": 0.0, "obs": obs.tolist(), "action": None}] if trace_every else []
        returns = 0.0
        fallen = truncated = False
        push_ticks = 0
        support_at_push = None
        support_steps = {name: [] for name in SUPPORT_LABELS} if record_support else None
        frame()
        for step in range(max_steps or HORIZON):
            if record_support or scenario.push_step == step:
                support_now = support()
                if record_support:
                    support_steps[support_now].append(step)
                if scenario.push_step == step:
                    support_at_push = support_now
            action = np.asarray(policy(obs.copy()), dtype=np.float64)
            if action.shape != (N_ACT,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
                raise ValueError("Policy must return 19 finite normalized targets in [-1,1]")
            pulse = (scenario.push_step is not None
                     and scenario.push_step <= step < scenario.push_step + PULSE_STEPS)
            try:
                if pulse:
                    data.xfrc_applied[torso, 1] = scenario.lateral_impulse_ns / (PULSE_STEPS * DT)
                    push_ticks += 1
                obs, reward, fallen, truncated, _ = env.step(action)
            finally:
                if pulse:
                    data.xfrc_applied[torso, 1] = 0.0
            if not np.isfinite(obs).all() or not math.isfinite(reward):
                raise RuntimeError("Nonfinite native H1 transition")
            trajectory.append(obs.copy())
            returns += float(reward)
            steps = step + 1
            if trace_every and (steps % trace_every == 0 or fallen or truncated or steps == max_steps):
                trace.append({"step": steps, "time_s": steps * DT, "obs": obs.tolist(), "action": action.tolist()})
            frame()
            if fallen or truncated:
                break

        states = np.stack(trajectory)
        nq = int(model.nq)
        completed_horizon = bool(truncated and steps == HORIZON and not fallen)
        physically_down = _physically_down(states[1:, 2], fallen)
        recovery = _recovery(states, scenario, nq=nq)
        metrics = {
            "steps": steps, "elapsed_s": steps * DT, "fell": bool(fallen),
            "completed_horizon": completed_horizon, "physically_down": physically_down, "return": returns,
            "native_success_bar_met": completed_horizon and returns >= native.task.success_bar,
            "x_m": float(states[-1, 0] - states[0, 0]),
            "y_m": float(states[-1, 1] - states[0, 1]), "min_root_z_m": float(states[:, 2].min()),
            "final_root_z_m": float(states[-1, 2]),
            "mean_vx_mps": float(states[1:, nq].mean()),
            "mean_abs_vy_mps": float(np.abs(states[1:, nq + 1]).mean()),
            "mean_abs_yaw_drift_deg": float(np.degrees(np.abs(_wrap_angle(_yaw(states) - _yaw(states)[0]))).mean()),
            "applied_pulse_steps": push_ticks, "recovery": recovery,
            "recovered_and_finished": bool(completed_horizon and not physically_down
                                           and recovery and recovery["recovered_at_2s"]),
        }
        result = {"scenario": asdict(scenario), "metrics": metrics, "trace": trace}
        if scenario.push_step is not None:
            result["audit"] = {"support_at_push": support_at_push}
        if record_support:
            result["support_steps"] = support_steps
        return result


def make_scenarios(impulse_ns, *, seed=20261109, n_calibration=8, n_train=12, n_holdout=16, n_nominal=8):
    """Deterministic, mutually exclusive seeds and train/holdout time bands.

    Only the calibration split may be used to choose pulse magnitude. Final
    holdout and nominal_holdout scenarios are for frozen evaluation only.
    """
    if not math.isfinite(impulse_ns) or impulse_ns <= 0:
        raise ValueError("impulse_ns must be positive and finite")
    counts = (n_calibration, n_train, n_holdout, n_nominal)
    if any(type(n) is not int or n < 2 for n in counts):
        raise ValueError("Scenario split sizes must be integers of at least two")
    def split(tag, n):
        # Independent RNG streams and disjoint seed ranges keep the held-out
        # assignments separate even when another split's size changes.
        width = 2**27
        rng = np.random.default_rng(np.random.SeedSequence([seed, tag]))
        seeds = (tag * width + rng.choice(width, size=n, replace=False)).tolist()
        return rng, seeds

    def pushed(tag, n, bands, factors):
        rng, seeds = split(tag, n)
        combos = [(band, factor) for band in bands for factor in factors]
        perm = rng.permutation(len(combos))
        result = []
        sign = int(rng.choice((-1, 1)))
        for i, episode_seed in enumerate(seeds):
            band, factor = combos[perm[(i // 2) % len(combos)]]
            result.append(Scenario(episode_seed, int(rng.integers(*band)), float(sign * factor * impulse_ns)))
            sign = -sign
        return result

    return {
        "calibration": pushed(0, n_calibration, ((175, 310), (350, 510), (560, 740)), (1.0,)),
        "train": pushed(1, n_train, ((150, 220), (330, 400), (510, 580), (690, 760)), (0.90, 1.0)),
        "holdout": pushed(2, n_holdout, ((240, 310), (420, 490), (600, 670), (780, 850)), (0.95, 1.05)),
        "nominal_train": [Scenario(n) for n in split(3, n_nominal)[1]],
        "nominal_holdout": [Scenario(n) for n in split(4, n_nominal)[1]],
    }


def single_support_scenarios(policy, scenarios, *, even_steps=False, desired_supports=None, max_shift_steps=35):
    """Fix each pulse at a load-bearing single-support tick in the original gait.

    This is host-only scheduling; use the resulting fixed scenarios for all
    candidates and report any actual phase deviations at pulse onset. Supply
    `desired_supports` (left_only/right_only) to explicitly balance phases.
    """
    scenarios = list(scenarios)
    desired = [None] * len(scenarios) if desired_supports is None else list(desired_supports)
    if len(desired) != len(scenarios) or any(side not in (None, "left_only", "right_only") for side in desired):
        raise ValueError("Expected one optional left_only/right_only label per scenario")
    if type(max_shift_steps) is not int or max_shift_steps < 0:
        raise ValueError("max_shift_steps must be a nonnegative integer")
    result = []
    reset = getattr(policy, "reset", None)
    for scenario, side in zip(scenarios, desired, strict=True):
        if scenario.push_step is None:
            raise ValueError("Only pushed scenarios can be scheduled at single support")
        end = min(HORIZON - PULSE_STEPS - RECOVERY_STEPS, scenario.push_step + max_shift_steps)
        if callable(reset):
            reset()
        reference = rollout(policy, Scenario(scenario.seed), trace_every=0, max_steps=end + 1, record_support=True)
        phases = reference["support_steps"]
        ticks = phases[side] if side else phases["left_only"] + phases["right_only"]
        start = max(PRE_STEPS, scenario.push_step - max_shift_steps)
        eligible = (tick for tick in ticks if start <= tick <= end and (not even_steps or tick % 2 == 0))
        step = min(eligible, key=lambda k: (abs(k - scenario.push_step), k), default=None)
        if step is None:
            raise ValueError(f"No load-bearing {side or 'single support'} tick near step {scenario.push_step}, seed {scenario.seed}")
        result.append(replace(scenario, push_step=step))
    return result


def calibrate_impulse(policy, strengths_ns=(8, 12, 16, 24, 32, 48), *, seed=20261109, n_calibration=8):
    """Pick the pulse with original-policy physical down rate closest to 50%.

    Returns None for the chosen magnitude if no candidate caused both falls
    and survivals (or if the original policy failed before a pulse). The
    supplied original policy should be stateless.
    """
    calibration = make_scenarios(1.0, seed=seed, n_calibration=n_calibration)["calibration"]
    summaries = []
    for magnitude in strengths_ns:
        if not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError("Calibration strengths must be positive and finite")
        n_falls = native_falls = invalid = 0
        for item in calibration:
            scenario = Scenario(item.seed, item.push_step, math.copysign(magnitude, item.lateral_impulse_ns))
            result = rollout(policy, scenario, trace_every=0)["metrics"]
            invalid += int(result["applied_pulse_steps"] == 0)
            n_falls += int(result["physically_down"])
            native_falls += int(result["fell"])
        summaries.append({"impulse_ns": float(magnitude), "falls": n_falls, "native_falls": native_falls,
                          "episodes": len(calibration), "prepush_failures": invalid})
    acceptable = [s for s in summaries if not s["prepush_failures"] and 0 < s["falls"] < s["episodes"]]
    best = min(acceptable, key=lambda s: (abs(s["falls"] / s["episodes"] - 0.5), s["impulse_ns"])) if acceptable else None
    return {"impulse_ns": best["impulse_ns"] if best else None, "calibration": summaries}


def nominal_preservation(reference, candidate):
    """Pair clean episodes by seed; preserve fall-free horizon, return, and progress."""
    if (reference["scenario"] != candidate["scenario"] or reference["scenario"]["push_step"] is not None):
        raise ValueError("Nominal comparisons require the same clean scenario")
    a, b = reference["metrics"], candidate["metrics"]
    return_ratio = b["return"] / a["return"] if a["return"] > 0 else None
    x_ratio = b["x_m"] / a["x_m"] if a["x_m"] > 0 else None
    speed_error = b["mean_vx_mps"] - a["mean_vx_mps"]
    yaw_drift_change = b["mean_abs_yaw_drift_deg"] - a["mean_abs_yaw_drift_deg"]
    return {"return_ratio": return_ratio, "progress_ratio": x_ratio,
            "mean_vx_error_mps": speed_error, "yaw_drift_change_deg": yaw_drift_change,
            "preserved": bool(a["completed_horizon"] and b["completed_horizon"]
                              and not a["physically_down"] and not b["physically_down"]
                              and return_ratio is not None and return_ratio >= 0.95
                              and x_ratio is not None and x_ratio >= 0.90
                              and abs(speed_error) <= 0.15 and yaw_drift_change <= 5.0)}
