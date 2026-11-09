"""Deterministic fixture for the weight interpreter, not a robot policy prior."""

import math
import unittest

from experiments.physevo_wsp.weights import (
    FEATURES, OPS, OUTPUTS, PolicyError, load_policy,
)


def fixture_stage(op, next_ok, next_bad, *, xyz=None, dz=0.0, jaw=0.0, yaw=None):
    bias = [0.0] * len(OUTPUTS)
    bias[OPS.index(op)] = 1.0
    bias[6:9] = [dz, jaw, next_bad]
    weights = [[8, FEATURES.index("ok"), next_ok - next_bad]]
    if xyz is not None:
        weights += [[i, FEATURES.index(f"{xyz}_{axis}"), 1.0] for i, axis in enumerate("xyz", 4)]
    if yaw is not None:
        weights.append([9, FEATURES.index(yaw), 1.0])
    return {"layers": [{"activation": "identity", "bias": bias, "weights": weights}]}


def fixture():
    return {"stages": [
        fixture_stage("move_tcp", 1, 8, xyz="source", dz=0.1, yaw="source_yaw"),
        fixture_stage("move_tcp", 2, 8, xyz="source", yaw="source_yaw"),
        fixture_stage("grip", 3, 8, jaw=0.75),
        fixture_stage("move_tcp", 4, 8, xyz="source", dz=0.1, yaw="source_yaw"),
        fixture_stage("move_tcp", 5, 8, xyz="dest"),
        fixture_stage("grip", 6, 8, jaw=0.25),
        fixture_stage("move_tcp", 7, 8, xyz="dest", dz=0.1),
        fixture_stage("done", 7, 7),
        fixture_stage("fail_reobserve", 8, 8),
    ]}


SENSORS = {
    "source_x": 0.4, "source_y": 0.2, "source_z": 0.5,
    "dest_x": 0.7, "dest_y": -0.1, "dest_z": 0.6,
    "tcp_x": 0.0, "tcp_y": 0.0, "tcp_z": 0.8, "jaw": 0.25,
    "ok": True, "source_yaw": 0.3,
}


class WeightsTest(unittest.TestCase):
    def test_numeric_closed_loop_and_transfer(self):
        policy = load_policy(fixture())
        stage = 0
        sensors = SENSORS.copy()
        ops = []
        for expected in range(7):
            self.assertEqual(stage, expected)
            result = policy.decide(stage, sensors)
            ops.append(result.op)
            if result.op == "move_tcp":
                sensors.update(zip(("tcp_x", "tcp_y", "tcp_z"), result.xyz))
            else:
                sensors["jaw"] = result.jaw
            stage = policy.advance(stage, sensors)
        self.assertEqual(ops, ["move_tcp", "move_tcp", "grip", "move_tcp", "move_tcp", "grip", "move_tcp"])
        self.assertEqual(policy.decide(stage, sensors).op, "done")
        moved = {**SENSORS, "source_x": 0.55, "source_z": 0.61, "source_yaw": -0.4}
        result = policy.decide(0, moved)
        self.assertEqual(result.xyz, (0.55, 0.2, 0.71))
        self.assertEqual(result.yaw, -0.4)

    def test_post_action_failure_routes_to_numeric_reobserve_stage(self):
        policy = load_policy(fixture())
        self.assertEqual(policy.decide(2, SENSORS).op, "grip")
        failure = policy.advance(2, {**SENSORS, "ok": False})
        self.assertEqual(failure, 8)
        self.assertEqual(policy.decide(failure, SENSORS).op, "fail_reobserve")

    def test_hidden_layer_and_observation_dependent_logits(self):
        spec = {"stages": [{"layers": [
            {"activation": "tanh", "bias": [-1.0], "weights": [[0, FEATURES.index("jaw"), 2.0]]},
            {"activation": "identity", "bias": [0.0] * len(OUTPUTS), "weights": [[0, 0, 1.0], [1, 0, -1.0]]},
        ]}]}
        policy = load_policy(spec)
        first = policy.decide(0, SENSORS)
        self.assertEqual(first.op, "grip")
        self.assertAlmostEqual(first.logits[1], math.tanh(0.5))
        self.assertEqual(policy.decide(0, {**SENSORS, "jaw": 0.8}).op, "move_tcp")

    def test_reject_invalid_artifacts_and_feedback(self):
        with self.assertRaises(PolicyError):
            load_policy("return [0, 1]")
        for weight in ([0, 99, 1], [0, 0, True], [0, 0, math.nan]):
            spec = fixture()
            spec["stages"][0]["layers"][0]["weights"].append(weight)
            with self.subTest(weight=weight), self.assertRaises(PolicyError):
                load_policy(spec)
        bad = fixture()
        bad["stages"][0]["layers"][0]["bias"][8] = 50
        policy = load_policy(bad)
        with self.assertRaisesRegex(PolicyError, "next_state"):
            policy.advance(0, SENSORS)
        with self.assertRaisesRegex(PolicyError, "missing sensors"):
            policy.decide(0, {})


if __name__ == "__main__":
    unittest.main()
