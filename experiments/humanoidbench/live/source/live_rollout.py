"""In-episode intervention on the native H1 rollout, using proprioception only."""

from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np

from push_env import DT, rollout


ACTOR_STRIDE = 2  # DR.Q infers at 25 Hz and holds the intervening native tick.


@dataclass(frozen=True)
class TriggerConfig:
    """The selected code arm's causal anomaly trigger, at the same 25 Hz."""

    lateral_delta_mps: float = 0.9
    root_wx_radps: float = 2.5
    warmup_actor_ticks: int = 30
    ema_alpha: float = 0.12
    history_steps: int = 40

    def __post_init__(self):
        limits = (self.lateral_delta_mps, self.root_wx_radps)
        if any(not math.isfinite(value) or value <= 0 for value in limits):
            raise ValueError("Trigger thresholds must be positive and finite")
        if (type(self.warmup_actor_ticks) is not int or self.warmup_actor_ticks < 0
                or not 0 < self.ema_alpha <= 1 or type(self.history_steps) is not int or self.history_steps < 1):
            raise ValueError("Invalid causal trigger history configuration")


def _roll(obs):
    w, x, y, z = obs[3:7]
    return math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))


def _anomaly(step, obs, history, config, ref_vy):
    tick = step // ACTOR_STRIDE
    vy = float(obs[27])  # native qvel[1], world lateral velocity
    wx = float(obs[29])  # native qvel[3], root X angular velocity
    if not (tick > config.warmup_actor_ticks and abs(vy - ref_vy) > config.lateral_delta_mps
            and abs(wx) > config.root_wx_radps):
        return None
    past = list(history)
    return {
        "trigger_step": step, "actor_tick": tick, "time_s": step * DT,
        "reason": ["lateral_velocity", "root_roll_rate"], "obs": obs.tolist(),
        "lateral_velocity_mps": vy, "reference_lateral_velocity_mps": ref_vy,
        "roll_rad": _roll(obs), "root_wx_radps": wx,
        "history": [{"step": step - len(past) + i, "obs": past[i].tolist()}
                    for i in range(0, len(past), 5)],
    }


def live_rollout(policy, scenario, *, on_event=None, on_ready=None, mode="pause", realtime=None,
                 video=None, trace_every=5, trigger=TriggerConfig(), max_steps=None):
    """Run the native task and dispatch at most one detected intervention.

    `pause`: call on_event(event), then optional on_ready(result, event) before
    the same actor tick's inference. Physics stays at that tick while waiting.

    `async`: on_event must quickly return a Future; poll it before each later
    25 Hz actor tick and call on_ready in the simulator thread when it is done.
    This mode paces the native simulation to wall clock by default. Do not
    mutate actor weights in the future's worker: installation belongs in
    on_ready. The runtime never waits for a future after the episode ends.

    Return native metrics/video plus `live` timing and intervention evidence.
    Callback results should be JSON-friendly. Return a boolean or a dictionary
    with `applied` from on_ready to say whether the candidate was installed.
    """
    if mode not in ("pause", "async"):
        raise ValueError("mode must be 'pause' or 'async'")
    if not isinstance(trigger, TriggerConfig):
        raise TypeError("trigger must be a TriggerConfig")
    if on_ready is not None and not callable(on_ready):
        raise TypeError("on_ready must be callable")
    if on_event is not None and not callable(on_event):
        raise TypeError("on_event must be callable")
    if mode == "async" and on_event is not None and on_ready is None:
        raise ValueError("Async interventions require an on_ready installation callback")
    realtime = mode == "async" if realtime is None else bool(realtime)
    history = deque(maxlen=trigger.history_steps)
    pending = None
    event = None
    audit = None
    start = None
    event_wall = None
    step = 0
    missed_ticks = 0
    max_late = 0.0
    paused_wall = 0.0
    ref_vy = None

    def error(exc):
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:600])

    def ready(decision, obs):
        started = time.monotonic()
        audit.update(decision_step=step, decision_sim_s=step * DT,
                     physical_delay_steps=step - event["trigger_step"],
                     physical_delay_s=(step - event["trigger_step"]) * DT,
                     root_z_m_at_decision=float(obs[2]),
                     wall_delay_s=started - event_wall)
        try:
            outcome = on_ready(decision, event) if on_ready is not None else decision
            audit["result"] = outcome
            declared = outcome.get("applied") if isinstance(outcome, dict) else outcome
            if declared is False:
                audit["status"] = "not_applied"
            elif declared is True:
                audit["status"] = "applied"
                audit["first_affected_actor_step"] = step
            else:
                audit["status"] = "callback_returned"
        except Exception as exc:
            error(exc)
        finally:
            audit["installation_wall_s"] = time.monotonic() - started

    def controller(obs):
        nonlocal step, start, max_late, missed_ticks, event, event_wall, audit, pending, paused_wall, ref_vy
        now = time.monotonic()
        if start is None:
            start = now
        if realtime:
            due = start + step * DT + paused_wall
            if now < due:
                time.sleep(due - now)
            late = max(0.0, time.monotonic() - due)
            max_late = max(max_late, late)
            missed_ticks += int(late >= DT)

        if step % ACTOR_STRIDE == 0:
            # Poll before dispatch, so even an already-completed Future cannot
            # install on its dispatch tick in the live, asynchronous condition.
            if pending is not None and pending.done():
                try:
                    ready(pending.result(), obs)
                except Exception as exc:
                    error(exc)
                pending = None
            if event is None and on_event is not None:
                vy = float(obs[27])
                if ref_vy is None:
                    ref_vy = vy
                event = _anomaly(step, obs, history, trigger, ref_vy)
                ref_vy += trigger.ema_alpha * (vy - ref_vy)
                if event is not None:
                    event_wall = time.monotonic()
                    audit = {"event": event, "status": "pending", "trigger_wall_s": event_wall - start}
                    try:
                        response = on_event(event)
                        audit["dispatch_wall_s"] = time.monotonic() - event_wall
                        if mode == "pause":
                            ready(response, obs)
                        elif not callable(getattr(response, "done", None)) or not callable(getattr(response, "result", None)):
                            raise TypeError("Async on_event must return a Future")
                        else:
                            pending = response
                    except Exception as exc:
                        error(exc)
                    finally:
                        if mode == "pause":
                            paused_wall = time.monotonic() - event_wall
                            audit["paused_wall_s"] = paused_wall
                            audit["paused_native_steps"] = 0
        history.append(obs.copy())
        action = policy(obs)
        step += 1
        return action

    result = rollout(controller, scenario, video=video, trace_every=trace_every, max_steps=max_steps)
    wall = time.monotonic() - start
    if pending is not None:
        audit["status"] = "episode_ended_before_install"
        audit["future_done_at_episode_end"] = pending.done()
    result["live"] = {"mode": mode, "realtime": realtime, "actor_hz": 1 / (DT * ACTOR_STRIDE),
                      "control_wall_s": wall, "paused_wall_s": paused_wall,
                      "late_native_ticks": missed_ticks, "max_tick_lag_s": max_late,
                      "intervention": audit}
    return result
