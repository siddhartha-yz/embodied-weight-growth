"""Evaluate a frozen actuator model on tasks absent from calibration.

The shared controller gets encoder observations and tip geometry. The only
candidate-specific inputs are bounded numeric weights for a forward model;
nothing in an adapter file is executable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


HOME = np.array([0.45, -0.85])
FORMS = {
    "mix": (2, 2),
    "bias": (2,),
    "state": (2, 4),
    "history": (2, 2),
    "history2": (2, 2),
    "deadzone": (2,),
}
DEFAULT_DZ = (0.0, 0.08, 0.16, 0.24)
P = 12.0
D = 7.0
ACC_LIMIT = 20.0
COMMAND_BLEND = 0.35


def shrink(u, d):
    """Motor command after a static dead zone (a deliberately approximate model)."""
    return np.sign(u) * np.maximum(np.abs(u) - d, 0)


@dataclass
class Adapter:
    mix: np.ndarray
    bias: np.ndarray
    state: np.ndarray
    history: np.ndarray
    history2: np.ndarray
    deadzone: np.ndarray

    @classmethod
    def from_dict(cls, obj):
        if not isinstance(obj, dict) or "mix" not in obj or set(obj) - FORMS.keys():
            raise ValueError("adapter requires mix; optional bias/state/history/history2/deadzone")
        arrays = {}
        for key, shape in FORMS.items():
            a = np.asarray(obj.get(key, np.zeros(shape)), dtype=float)
            if a.shape != shape or not np.all(np.isfinite(a)) or np.any(np.abs(a) > 10_000):
                raise ValueError(f"{key}: expected finite values of shape {shape} within +/-10000")
            arrays[key] = a
        svals = np.linalg.svd(arrays["mix"], compute_uv=False)
        if svals[-1] < 1e-3 or svals[0] > 100 * svals[-1]:
            raise ValueError("mix must be invertible and well conditioned")
        if np.any((arrays["deadzone"] < 0) | (arrays["deadzone"] > 0.8)):
            raise ValueError("deadzone values must be in [0, 0.8]")
        return cls(**arrays)

    def predict(self, q, dq, u, prev, prev2):
        return (
            self.mix @ shrink(u, self.deadzone)
            + self.bias
            + self.state @ np.r_[q, dq]
            + self.history @ prev
            + self.history2 @ prev2
        )

    def commands(self, q, dq, prev, prev2, target_acc):
        offset = (
            self.bias + self.state @ np.r_[q, dq]
            + self.history @ prev + self.history2 @ prev2
        )
        u_eff = np.linalg.solve(self.mix, target_acc - offset)
        # Shrink's inverse is multivalued at zero. Use the zero command rather
        # than sending a small alternating command across the dead zone.
        u = np.where(
            np.abs(u_eff) < 1e-5, 0.0, u_eff + np.sign(u_eff) * self.deadzone
        )
        # A large delayed coefficient can make exact one-step inversion
        # oscillate, even for an exact plant model. Same damping for all arms.
        return prev + COMMAND_BLEND * (np.clip(u, -1.0, 1.0) - prev)

    def as_dict(self):
        return {key: getattr(self, key).tolist() for key in FORMS}


def samples(records, dt):
    rows = []
    for episode_id, ep in enumerate(records["episodes"]):
        obs = ep["initial"]
        prev = prev2 = np.zeros(2)
        for step in ep["steps"]:
            q = np.asarray(obs["q"], dtype=float)
            dq = np.asarray(obs["dq"], dtype=float)
            u = np.asarray(step["u"], dtype=float)
            nxt = np.asarray(step["dq"], dtype=float)
            if any(x.shape != (2,) for x in (q, dq, u, nxt)):
                raise ValueError("each observation and control vector must have length 2")
            rows.append(np.r_[q, dq, prev, prev2, u, (nxt - dq) / dt, episode_id])
            prev2, prev, obs = prev, u, step
    data = np.asarray(rows)
    if len(data) < 12 or not np.all(np.isfinite(data)):
        raise ValueError("calibration needs at least 12 finite transitions")
    if np.linalg.matrix_rank(data[:, 8:10]) < 2:
        raise ValueError("calibration did not excite both motor channels")
    return data


def design(data, order, deadzone):
    u = shrink(data[:, 8:10], deadzone)
    parts = [u, np.ones((len(data), 1)), data[:, :4]]
    if order:
        parts.append(data[:, 4:4+2*order])
    return np.column_stack(parts)


def fit(data, order, deadzone):
    x = design(data, order, deadzone)
    y = data[:, 10:12]
    # Ridge on rescaled features. The bias is not penalized.
    scale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 0.05)
    reg = 0.004 * len(x) * np.eye(x.shape[1])
    reg[2, 2] = 0.0
    z = x / scale
    coeff = np.linalg.solve(z.T @ z + reg, z.T @ y) / scale[:, None]
    return Adapter.from_dict({
        "mix": coeff[:2].T,
        "bias": coeff[2],
        "state": coeff[3:7].T,
        "history": coeff[7:9].T if order > 0 else np.zeros((2, 2)),
        "history2": coeff[9:11].T if order > 1 else np.zeros((2, 2)),
        "deadzone": np.full(2, deadzone),
    })


def calibration_score(adapter, data):
    predicted = (
        shrink(data[:, 8:10], adapter.deadzone) @ adapter.mix.T
        + adapter.bias
        + data[:, :4] @ adapter.state.T
        + data[:, 4:6] @ adapter.history.T
        + data[:, 6:8] @ adapter.history2.T
    )
    return float(np.sqrt(np.mean((predicted - data[:, 10:12]) ** 2)))


def fit_baselines(data):
    """Choose the nonlinear variant on withheld probes, never task outcomes."""
    episode_ids = np.unique(data[:, -1])
    if len(episode_ids) > 1 and np.sum(data[:, -1] != episode_ids[-1]) >= 12:
        select = data[:, -1] == episode_ids[-1]
        calibration, selection = data[~select], data[select]
    else:
        cut = max(10, int(0.75 * len(data)))
        calibration, selection = data[:cut], data[cut:]
    # Regularized ARX plus static dead zone, including the plain linear model.
    candidates = [(0, 0.0)] + [
        (order, d) for order in (1, 2) for d in DEFAULT_DZ
        if len(calibration) > 7 + 2 * order
    ]
    scored = []
    for order, deadzone in candidates:
        try:
            model = fit(calibration, order, deadzone)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if len(selection):
            scored.append((calibration_score(model, selection), order, deadzone))
    # If an episode split is underactuated, there is not enough information to
    # select the richer model, even though a whole-record linear fit is possible.
    _, history_order, deadzone = min(scored) if scored else (None, 0, 0.0)
    return {
        "linear": fit(data, 0, 0.0),
        "selected_code": fit(data, history_order, deadzone),
    }, {
        "selection_transitions": len(selection),
        "candidate_qdd_rmse_rad_s2": [
            {"history_order": order, "deadzone": dz, "rmse": score}
            for score, order, dz in scored
        ],
        "selected_history_order": history_order,
        "selected_deadzone": deadzone,
    }


def smoothstep(x):
    return x**3 * (10 + x * (-15 + 6 * x))


def geometry(tip_from_q):
    p0 = np.asarray(tip_from_q(np.zeros(2)))
    p1 = np.asarray(tip_from_q(np.array([0.0, np.pi / 2])))
    return float(p0[0] - p1[1]), float(p1[1])


def inverse_xy(tip, lengths):
    # Negative-elbow branch; the pilot keeps trajectories on this branch.
    l1, l2 = lengths
    x, y = np.asarray(tip).T
    c2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if np.any(np.abs(c2) > 1 + 1e-9):
        raise ValueError("held-out task not reachable on the nominal geometry")
    q2 = -np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.column_stack((q1, q2))


def tasks(dt, tip_from_q):
    """Fixed, unseen Cartesian tasks plus hold; all sampled at plant DT."""
    lengths = geometry(tip_from_q)
    home = np.asarray(tip_from_q(HOME))
    starts = [
        np.array([0.28, -1.04]),
        np.array([0.56, -0.71]),
        HOME.copy(),
    ]
    goals = [
        np.array([0.176, 0.056]),
        np.array([0.168, -0.049]),
    ]
    travel_s = (1.50, 1.50, 2.40)
    hold = 0.60
    for i, (name, initial) in enumerate(zip(("reach_up", "reach_down", "loop"), starts)):
        start = np.asarray(tip_from_q(initial))
        times = np.arange(round((travel_s[i] + hold) / dt) + 1) * dt
        frac = smoothstep(np.minimum(times / travel_s[i], 1.0))
        if i < len(goals):
            tip = start + frac[:, None] * (goals[i] - start)
        else:
            theta = 2 * np.pi * frac
            radius = 0.027
            tip = home + np.column_stack(
                (radius * (np.cos(theta) - 1), radius * 1.4 * np.sin(theta))
            )
        q = inverse_xy(tip, lengths)
        dq = np.gradient(q, dt, axis=0, edge_order=2)
        ddq = np.gradient(dq, dt, axis=0, edge_order=2)
        yield name, q, dq, ddq, tip


def track(plant, adapter, task, dt):
    name, ref_q, ref_dq, ref_ddq, ref_tip = task
    obs = plant.reset(q=ref_q[0].tolist())
    prev = prev2 = np.zeros(2)
    actual_tip = [obs["tip"]]
    controls = []
    for i in range(len(ref_q) - 1):
        q, dq = np.asarray(obs["q"]), np.asarray(obs["dq"])
        acceleration = (
            ref_ddq[i]
            + P * (ref_q[i] - q)
            + D * (ref_dq[i] - dq)
        )
        cmd = adapter.commands(q, dq, prev, prev2, np.clip(acceleration, -ACC_LIMIT, ACC_LIMIT))
        obs = plant.step(cmd.tolist())
        actual_tip.append(obs["tip"])
        controls.append(cmd)
        prev2, prev = prev, cmd
    error = np.linalg.norm(np.asarray(actual_tip) - ref_tip, axis=1)
    last = max(1, round(0.30 / dt))
    active = error[: -round(0.60 / dt)]
    tip_rmse = float(np.sqrt(np.mean(active**2)))
    final_error = float(np.mean(error[-last:]))
    success = final_error <= 0.015 and tip_rmse <= 0.030
    return {
        "task": name,
        "motion_tip_rmse_m": tip_rmse,
        "final_hold_tip_error_m": final_error,
        "max_tip_error_m": float(np.max(error)),
        "control_rms": float(np.sqrt(np.mean(np.square(controls)))),
        "control_saturation_fraction": float(np.mean(np.abs(controls) >= 0.999)),
        "success": bool(success),
        "trajectory": {"target_tip": ref_tip.tolist(), "actual_tip": actual_tip},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="same hidden body seed as probes")
    parser.add_argument("--condition", choices=["linear", "backlash"], required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="optional numeric WSP weights in JSON")
    args = parser.parse_args()

    from plant import DT, Plant, tip_from_q

    records = json.loads(args.records.read_text())
    data = samples(records, DT)
    adapters, fit_report = fit_baselines(data)
    if args.adapter:
        adapters["wsp"] = Adapter.from_dict(json.loads(args.adapter.read_text()))

    results = {}
    for label, adapter in adapters.items():
        plant = Plant(seed=args.seed, condition=args.condition)
        trials = [
            track(plant, adapter, t, DT) for t in tasks(DT, tip_from_q)
        ]
        results[label] = {
            "probe_qdd_rmse_rad_s2": calibration_score(adapter, data),
            "frozen_adapter": adapter.as_dict(),
            "n_success": sum(t["success"] for t in trials),
            "mean_motion_tip_rmse_m": float(np.mean([t["motion_tip_rmse_m"] for t in trials])),
            "mean_final_hold_tip_error_m": float(np.mean([t["final_hold_tip_error_m"] for t in trials])),
            "trials": trials,
        }
    print(json.dumps({
        "condition": args.condition,
        "seed": args.seed,
        "probe_transitions": len(data),
        "fit_selection": fit_report,
        "success_criteria": {"motion_tip_rmse_m_max": 0.030, "final_hold_tip_error_m_max": 0.015},
        "controllers": results,
    }, allow_nan=False))


if __name__ == "__main__":
    main()
