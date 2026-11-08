import math
import unittest

from policy import FEATURES, PolicyError, compile_policy


OBS = {
    "q": [0.4, -0.3], "dq": [0.2, -0.1], "tip": [0.05, 0.02],
    "tip_v": [0.07, -0.01], "prev_u": [0.12, -0.17], "prev2_u": [-0.09, 0.15],
}
GOAL = [0.10, -0.03]


def select(i, n=len(FEATURES), factor=1):
    return [factor if j == i else 0 for j in range(n)]


class WeightsTests(unittest.TestCase):
    def test_feature_order(self):
        expected = [
            0.1, 0.04, -0.03, 0.02, -0.01, 0.005, 0.002, 0.007, -0.001,
            0.005, -0.005, 0.012, -0.017, -0.009, 0.015,
        ]
        for j, value in enumerate(expected):
            with self.subTest(feature=FEATURES[j]):
                run = compile_policy("weights", {"layers": [
                    {"weight": [select(j, factor=.1), select(j, factor=-.1)], "bias": [0, 0]}
                ]})
                u0, u1 = run(OBS, GOAL)
                self.assertAlmostEqual(u0, value)
                self.assertAlmostEqual(u1, -value)

    def test_hidden_tanh_then_linear_and_clamp(self):
        run = compile_policy("weights", {"layers": [
            {"weight": [select(9, factor=4)], "bias": [0.1]},
            {"weight": [[2], [-6]], "bias": [0.1, 0.0]},
        ]})
        a, b = run(OBS, GOAL)
        self.assertAlmostEqual(a, 2 * math.tanh(.3) + .1)
        self.assertEqual(b, -.8)

    def test_incompatible_shapes_nonfinite_and_too_wide(self):
        cases = [
            {"layers": [{"weight": [[1], [2]], "bias": [0, 0]}]},
            {"layers": [{"weight": [select(0), select(1)], "bias": [0, float("nan")]}]},
            {"layers": [
                {"weight": [select(0)] * 17, "bias": [0] * 17},
                {"weight": [select(0, 17), select(1, 17)], "bias": [0, 0]},
            ]},
            {"layers": []},
        ]
        for spec in cases:
            with self.subTest(spec=spec):
                with self.assertRaises(PolicyError):
                    compile_policy("weights", spec)


class CodeTests(unittest.TestCase):
    def test_index_unpack_if_short_circuit_and_math(self):
        run = compile_policy("code", """def policy(obs, goal):
    x, y = obs["tip"]
    dx, dy = goal[0] - x, goal[1] - y
    if dx > 0 and not False:
        u = 2 * dx + sin(0)
        u += 0.2
    else:
        u = 1 / 0
    return u, max(-1.0, -sqrt(abs(dy)) - 1) if True or 1/0 else 0
""")
        u0, u1 = run(OBS, GOAL)
        self.assertAlmostEqual(u0, 0.3)
        self.assertEqual(u1, -0.8)

    def test_code_is_stateless(self):
        run = compile_policy("code", """def policy(obs, goal):
    u = obs["prev_u"][0]
    u += 0.01
    return [u, 0]
""")
        for _ in range(5):
            self.assertEqual(run(OBS, GOAL), [0.13, 0.0])
        self.assertEqual(OBS["prev_u"], [0.12, -0.17])

    def test_reject_escape_and_unbounded_programs(self):
        cases = [
            "import os\ndef policy(obs, goal):\n    return [0, 0]",
            'def policy(obs, goal):\n    return obs.__class__',
            'def policy(obs, goal):\n    return open("/tmp/forbidden")',
            "def policy(obs, goal):\n    return policy(obs, goal)",
            "def policy(obs, goal):\n    while True:\n        pass\n    return [0, 0]",
            'def policy(obs, goal):\n    obs["q"][0] = 5\n    return [0, 0]',
            "def policy(obs, goal, x=0):\n    return [0, 0]",
            "def policy(obs, goal):\n    return [x for x in goal]",
        ]
        for source in cases:
            with self.subTest(source=source):
                with self.assertRaises(PolicyError):
                    compile_policy("code", source)

    def test_runtime_errors_are_safe_and_readable(self):
        for body, message in (
            ("return [obs['q'][8], 0]", "list index out of range"),
            ("return [1/0, 0]", "division by zero"),
            ("return [obs == obs, 0]", "compare scalar"),
            ("if False:\n        return [0, 0]", "did not return"),
            ("return [0]", "must contain 2"),
        ):
            with self.subTest(body=body):
                run = compile_policy("code", "def policy(obs, goal):\n    " + body)
                with self.assertRaisesRegex(PolicyError, message):
                    run(OBS, GOAL)


if __name__ == "__main__":
    unittest.main()
