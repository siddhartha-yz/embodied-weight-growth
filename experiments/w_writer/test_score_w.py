"""Fixtures exercise the shape and attempt-boundary contracts."""
import unittest

from score_w import assess, score_log


W = [[.11, .04], [.13, -.03], [.15, .027], [.17, -.03], [.19, .04]]


def turn(index, attempt, strokes, op="move"):
    return dict(turn=index, request={"op": op},
                result={"attempt": attempt, "ok": True, "strokes": strokes})


class WScoreTests(unittest.TestCase):
    def test_w_accepts_either_drawing_direction(self):
        for glyph in (W, W[::-1]):
            self.assertTrue(assess([glyph])["passed"])

    def test_no_credit_for_single_v_retracing_or_tiny_w(self):
        for glyph in (W[:3], W + W[::-1], [[x / 10, y / 10] for x, y in W]):
            self.assertFalse(assess([glyph])["passed"])

    def test_ink_must_be_one_stroke(self):
        self.assertFalse(assess([W[:3], W[2:]])["passed"])

    def test_attempt_uses_its_final_canvas(self):
        extended = W + [[.14, 0], [.11, -.015]]
        score = score_log([turn(1, 1, [W]), turn(2, 1, [extended]),
                           turn(3, 2, [], "reset"), turn(4, 2, [W])])
        self.assertFalse(score["attempts"][0]["passed"])
        self.assertEqual(score["attempts"][0]["turn"], 2)
        self.assertEqual(score["best"]["turn"], 4)

    def test_blank_final_attempt_does_not_erase_prior_recorded_ink(self):
        score = score_log([turn(1, 1, [W]), turn(2, 2, [], "reset")])
        self.assertEqual(score["last"]["turn"], 1)


if __name__ == "__main__":
    unittest.main()
