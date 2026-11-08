"""Plant-contract checks; run with python -m unittest test_plant."""

import unittest
from importlib.util import find_spec

import numpy as np

from plant import ARM_LENGTHS, HOME, Plant, _Transmission, tip_from_q


class TransmissionTests(unittest.TestCase):
    def test_tip_geometry(self):
        upper, lower = ARM_LENGTHS
        np.testing.assert_allclose(tip_from_q((0, 0)), (upper + lower, 0), atol=1e-14)
        np.testing.assert_allclose(tip_from_q((np.pi / 2, -np.pi / 2)), (lower, upper))

    def test_linear_is_fixed_under_reset(self):
        for seed in range(10):
            motor = _Transmission(seed, "linear")
            self.assertGreater(np.linalg.det(motor.mix), 0.4)
            positive = motor.apply(np.array([0.6, -0.2]))
            motor.reset()
            np.testing.assert_allclose(motor.apply(np.array([-0.6, 0.2])), -positive)
        self.assertFalse(np.allclose(_Transmission(0, "linear").mix, _Transmission(1, "linear").mix))

    def test_backlash_depends_on_history_and_can_reset(self):
        motor = _Transmission(42, "backlash")
        command = np.array([0.5, 0.0])
        initial = motor.apply(-command)
        motor.reset()
        for _ in range(12):
            motor.apply(command)
        reversal = motor.apply(-command)
        self.assertLess(initial[0], 0)
        self.assertGreater(reversal[0], 0)
        for _ in range(14):
            reached = motor.apply(-command)
        self.assertLess(reached[0], initial[0])
        motor.reset()
        np.testing.assert_allclose(motor.apply(-command), initial)


@unittest.skipUnless(find_spec("mujoco") is not None, "MuJoCo not installed in this Python")
class PlantTests(unittest.TestCase):
    def test_encoder_tip_and_simulator_geometry_agree(self):
        plant = Plant(4, "linear")
        for q in [HOME, (-0.4, 0.9), (1.0, -1.2)]:
            obs = plant.reset(q)
            self.assertEqual(set(obs), {"q", "dq", "tip"})
            np.testing.assert_allclose(obs["q"], q)
            np.testing.assert_allclose(obs["tip"], plant._data.site("tip").xpos[:2])

    def test_reset_preserves_body_and_clears_history(self):
        plant = Plant(4, "backlash")
        sequence = [[0.5, 0.0]] * 8 + [[-0.5, 0.3]] * 12
        plant.reset()
        reference = [plant.step(u) for u in sequence]
        plant.reset()
        replay = [plant.step(u) for u in sequence]
        self.assertEqual(replay, reference)
        self.assertGreater(np.linalg.norm(np.asarray(reference[-1]["q"]) - HOME), 0.01)

    def test_motion_in_both_channels_and_clipping(self):
        plant = Plant(8, "linear")
        plant.reset()
        positive = plant.step([1.0, 0])
        plant.reset()
        self.assertEqual(plant.step([3.0, 0]), positive)
        plant.reset()
        self.assertFalse(np.allclose(plant.step([0, 1.0])["dq"], positive["dq"]))


if __name__ == "__main__":
    unittest.main()
