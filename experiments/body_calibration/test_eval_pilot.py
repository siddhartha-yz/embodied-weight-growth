"""Deterministic evaluator checks, without MuJoCo or a model/API dependency."""

import unittest

import numpy as np

from eval_pilot import Adapter, calibration_score, fit_baselines, samples, tasks, track


DT = 0.02
DAMPING = np.array([6.0, 9.0])
GAIN = np.array([[38.0, -13.0], [5.0, 50.0]])


def tip_from_q(q):
    shoulder, elbow = q
    return [
        0.14 * np.cos(shoulder) + 0.11 * np.cos(shoulder + elbow),
        0.14 * np.sin(shoulder) + 0.11 * np.sin(shoulder + elbow),
    ]


class SyntheticPlant:
    """Different from Plant: an exactly identifiable delayed model."""

    def __init__(self, lag, deadzone):
        self.lag = lag
        self.deadzone = deadzone

    def observe(self):
        return dict(q=self.q.tolist(), dq=self.dq.tolist(), tip=tip_from_q(self.q))

    def reset(self, q=(0.45, -0.85)):
        self.q = np.array(q, dtype=float)
        self.dq = np.zeros(2)
        self.last = np.zeros(2)
        return self.observe()

    def step(self, u):
        u = np.asarray(u)
        eff = np.sign(u) * np.maximum(np.abs(u) - self.deadzone, 0)
        acc = GAIN @ ((1 - self.lag) * eff + self.lag * self.last) - DAMPING * self.dq
        self.dq += DT * acc
        self.q += DT * self.dq
        self.last = u
        return self.observe()


def make_probes(plant):
    rng = np.random.default_rng(3)
    episodes = []
    for _ in range(5):
        init = plant.reset(np.array([0.45, -0.85]) + rng.normal(0, 0.25, 2))
        steps = []
        for _ in range(30):
            u = rng.uniform(-1, 1, 2).tolist()
            steps.append({"u": u, **plant.step(u)})
        episodes.append(dict(initial=init, steps=steps))
    return samples({"episodes": episodes}, DT)


class EvaluatorTests(unittest.TestCase):
    def test_linear_calibration_recovers_control(self):
        plant = SyntheticPlant(0, 0)
        adapters, report = fit_baselines(make_probes(plant))
        self.assertEqual(report["selected_history_order"], 0)
        for task in tasks(DT, tip_from_q):
            out = track(plant, adapters["selected_code"], task, DT)
            self.assertTrue(out["success"])
            self.assertLess(out["motion_tip_rmse_m"], 0.006)

    def test_strong_lag_model_has_stable_control(self):
        lag, deadzone = 0.6, 0.08
        plant = SyntheticPlant(lag, deadzone)
        data = make_probes(plant)
        adapters, report = fit_baselines(data)
        self.assertEqual(report["selected_history_order"], 1)
        self.assertEqual(report["selected_deadzone"], deadzone)
        self.assertLess(calibration_score(adapters["selected_code"], data), 0.2)
        self.assertGreater(calibration_score(adapters["linear"], data), 5)
        for task in tasks(DT, tip_from_q):
            out = track(plant, adapters["selected_code"], task, DT)
            self.assertTrue(out["success"])
            self.assertEqual(out["control_saturation_fraction"], 0)

    def test_weight_schema_rejects_invalid_artifacts(self):
        for invalid in (
            {"mix": [[0, 0], [0, 0]]},
            {"mix": [[1, 0], [0, 1]], "deadzone": [0, 1]},
            {"mix": [[1, 0], [0, 1]], "state": [[1], [1]]},
            {"mix": [[1, 0], [0, 1]], "extra": "not executable"},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    Adapter.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()
