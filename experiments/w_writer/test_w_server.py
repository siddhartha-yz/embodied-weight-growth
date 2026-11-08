"""Exercise the protocol and ink bookkeeping without requiring MuJoCo locally."""
import unittest

from w_server import DT, MAX_MOVES, Session, START


class TestPlant:
    def reset(self, q):
        self.q = q
        self.tip = [0.16, 0.04]
        return self.observation([0, 0])

    def step(self, u):
        self.tip = [x + v * DT for x, v in zip(self.tip, u)]
        return self.observation(u)

    def observation(self, v):
        return {"q": list(self.q), "dq": list(v), "tip": self.tip}


CODE = "def policy(obs, goal):\n    return [0.25, -0.25]"


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.s = Session("code", plant=TestPlant())

    def send(self, op, **kwargs):
        return self.s.request({"op": op, **kwargs})

    def test_observation_budget_and_pen_up(self):
        initial = self.s.snapshot()
        self.assertEqual(initial["state"]["q"], list(START))
        self.assertEqual(initial["meters"]["moves_left"], 18)
        self.assertFalse(self.send("drive", u=[0.3, 0.0], ticks=5, pen=True)["ok"])
        self.assertFalse(self.send("drive", u=[0.9, 0.0], ticks=5)["ok"])
        self.assertFalse(self.send("drive", u=[0.3, 0.0], ticks=True)["ok"])
        r = self.send("drive", u=[0.3, 0.0], ticks=5)
        self.assertTrue(r["ok"])
        self.assertEqual(r["strokes"], [])
        self.assertEqual(len(r["trace"]), 6)
        self.assertEqual(r["state"]["prev_u"], [0.3, 0.0])
        self.assertAlmostEqual(r["state"]["tip_v"][0], 0.3)
        self.assertEqual(r["samples"][0]["tick"], 0)
        self.assertEqual(r["samples"][-1]["tick"], 5)
        self.assertEqual(r["meters"]["moves_used"], 1)

    def test_installed_module_draws_and_resets(self):
        use = {"goal": [0.17, 0.0], "ticks": 2, "pen": True}
        self.assertFalse(self.send("use", **use)["ok"])
        self.assertEqual(self.send("install", spec=CODE)["policy_version"], 1)
        self.assertFalse(self.send("install", spec="import os")["ok"])
        a = self.send("use", **use)
        b = self.send("use", **use)
        self.assertEqual(len(a["strokes"]), 1)
        self.assertEqual(len(a["strokes"][0]), 3)
        self.assertEqual(len(b["strokes"][0]), 5)
        self.send("drive", u=[0.0, 0.0], ticks=1)
        c = self.send("use", **use)
        self.assertEqual(len(c["strokes"]), 2)
        self.assertEqual(c["meters"]["draw_moves"], 3)
        reset = self.send("reset")
        self.assertEqual(reset["attempt"], 2)
        self.assertEqual(reset["strokes"], [])
        self.assertEqual(reset["policy_version"], 1)
        self.assertEqual(reset["meters"]["ticks_used"], 7)
        self.assertTrue(self.send("use", **use)["ok"])

    def test_policy_error_and_fixed_budget(self):
        code = "def policy(obs, goal):\n    return [1 / 0, 0.0]"
        self.assertTrue(self.send("install", spec=code)["ok"])
        bad = self.send("use", goal=[0.17, 0.0], ticks=2, pen=True)
        self.assertFalse(bad["ok"])
        self.assertIn("policy stopped after 0 ticks", bad["error"])
        self.assertEqual(bad["strokes"], [])
        self.assertEqual(bad["meters"]["moves_used"], 1)
        for _ in range(MAX_MOVES - 1):
            self.assertTrue(self.send("drive", u=[0.0, 0.0], ticks=1)["ok"])
        self.assertFalse(self.send("drive", u=[0.0, 0.0], ticks=1)["ok"])
        self.assertTrue(self.send("finish")["finished"])
        self.assertFalse(self.send("reset")["ok"])

    def test_numeric_module_uses_goal_and_returns_motor_sample(self):
        s = Session("weights", plant=TestPlant())
        rows = [[0.0] * 15, [0.0] * 15]
        rows[0][9] = 8.0  # first motor responds to goal-x error
        rows[1][10] = 8.0  # second motor responds to goal-y error
        layer = {"weight": rows, "bias": [0.0, 0.0]}
        self.assertTrue(s.request({"op": "install", "spec": {"layers": [layer]}})["ok"])
        r = s.request({"op": "use", "goal": [0.19, 0.02], "ticks": 3, "pen": True})
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["samples"][1]["u"][0], 0.24)
        self.assertGreater(r["state"]["tip"][0], 0.16)
        self.assertLess(r["state"]["tip"][1], 0.04)


if __name__ == "__main__":
    unittest.main()
