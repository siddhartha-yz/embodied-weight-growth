import math
import unittest

from walk_plant import WalkPlant
from walk_policy import FEATURES, compile_policy
from walk_server import Session, Steps


ZERO = "def policy(obs, goal):\n    return [0,0,0,0,0,0]"


def pose(t, contact, right_x, left_x):
    return {"t": t, "contact": contact, "pose": [
        {"name": "foot_r", "start": [right_x, 0], "end": [right_x, 0]},
        {"name": "foot_l", "start": [left_x, 0], "end": [left_x, 0]},
    ]}


class StepTest(unittest.TestCase):
    def test_sliding_and_hopping_do_not_count(self):
        s = Steps(pose(0, [1, 1], 0, 0))
        s.update(pose(.08, [1, 1], .1, .1))
        s.update(pose(.16, [0, 0], .1, .1))
        s.update(pose(.26, [1, 1], .2, .2))
        self.assertEqual(s.alternating_steps, 0)

    def test_supported_swing_and_alternation(self):
        s = Steps(pose(0, [1, 1], 0, 0))
        s.update(pose(.02, [0, 1], 0, 0))
        s.update(pose(.12, [1, 1], .08, 0))
        s.update(pose(.14, [0, 1], .08, 0))
        s.update(pose(.24, [1, 1], .16, 0))
        s.update(pose(.26, [1, 0], .16, 0))
        s.update(pose(.36, [1, 1], .16, .11))
        self.assertEqual([e["foot"] for e in s.touchdowns], ["R", "R", "L"])
        self.assertEqual(s.alternating_steps, 2)


class WalkerTest(unittest.TestCase):
    def test_same_body_and_reset_with_only_pitch_change(self):
        a, b = WalkPlant(), WalkPlant(scene="transfer")
        initial = a.reset()
        self.assertEqual(a.model.nu, 6)
        self.assertEqual(a.model.actuator_gear.tolist(), b.model.actuator_gear.tolist())
        self.assertAlmostEqual(b.reset()["pitch"] - initial["pitch"], .06)
        self.assertEqual(b.data.qpos[3:].tolist(), a.data.qpos[3:].tolist())
        self.assertEqual(b.data.qvel.tolist(), a.data.qvel.tolist())
        a.step([.1, .1, 0, 0, 0, 0])
        self.assertEqual(a.reset(), initial)

    def test_raw_and_policy_fall_identically(self):
        a, b = Session("code"), Session("code", frozen_spec=ZERO)
        raw = a.request({"op": "drive", "u": [0]*6, "ticks": 250})
        controlled = b.request({"op": "run", "ticks": 250})
        self.assertEqual(raw["trace"], controlled["trace"])
        self.assertTrue(raw["walk"]["fall_reason"])
        self.assertFalse(raw["success"])
        self.assertEqual(raw["walk"]["alternating_steps"], 0)
        self.assertFalse(b.request({"op": "run", "ticks": 5})["ok"])
        self.assertFalse(b.request({"op": "drive", "u": [0]*6, "ticks": 5})["ok"])
        self.assertFalse(b.request({"op": "install", "spec": ZERO})["ok"])
        self.assertEqual(b.request({"op": "reset"})["policy_version"], 1)
        self.assertEqual(b.request({"op": "run", "ticks": 250})["trace"], raw["trace"])

    def test_numeric_layers_and_code_see_same_pitch(self):
        state = WalkPlant().reset()
        linear = {"rows": [{"terms": [{"feature": FEATURES.index("pitch"), "coef": 2.0}], "bias": 0.0}]}
        out = {"rows": [{"terms": [{"feature": 0, "coef": .5}], "bias": .1} for _ in range(6)]}
        fn = compile_policy("weights", {"layers": [linear, out]})
        value = .1 + .5 * math.tanh(2 * state["pitch"])
        code = "def policy(obs, goal):\n    u=0.1+0.5*tanh(2*obs['pitch'])\n    return [u,u,u,u,u,u]"
        self.assertEqual(fn(state, .5), [value]*6)
        self.assertEqual(compile_policy("code", code)(state, .5), [value]*6)


if __name__ == "__main__":
    unittest.main()
