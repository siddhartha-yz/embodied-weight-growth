import importlib.util
import math
import unittest

from slot_plant import START_Q, SlotPlant, in_slot
from slot_policy import FEATURES, compile_policy
from slot_server import Session
from plant import Plant


def code(source):
    return "def policy(obs, goal):\n    return " + source


class FakePlant:
    last_tip_wall = False

    def geometry(self):
        return {}

    def reset(self):
        return {
            "q": list(START_Q), "dq": [0.0, 0.0], "tip": [.15, .04],
            "block": [.148, 0], "block_angle": [1, 0], "block_v": [0, 0], "contact": [0, 0],
        }

    def step(self, u):
        state = self.reset()
        state["contact"] = [1, 1]
        return state


class ProtocolTest(unittest.TestCase):
    def test_frozen_only_allows_module_to_move(self):
        s = Session("code", scene="transfer", frozen_spec=code("[0.1,0.2]"), plant=FakePlant())
        self.assertTrue(s.snapshot()["policy_frozen"])
        self.assertFalse(s.request({"op":"drive", "u":[.1,0], "ticks":1})["ok"])
        self.assertFalse(s.request({"op":"install", "spec":code("[0,0]")})["ok"])
        self.assertEqual(s.snapshot()["meters"]["ticks_used"], 0)
        used = s.request({"op":"use", "goal":[.15,.04], "ticks":7})
        self.assertTrue(used["ok"])
        self.assertEqual(used["state"]["prev_u"], [0.1,0.2])
        self.assertEqual(used["contact_summary"]["tip_block_ticks"], 7)
        self.assertTrue(used["contact_summary"]["stall"])
        reset = s.request({"op":"reset"})
        self.assertEqual(reset["policy_version"], 1)
        self.assertEqual(reset["meters"]["ticks_used"], 7)

    def test_policy_fields_match_in_both_arms(self):
        obs = Session("code", plant=FakePlant()).state
        obs["block_angle"] = [.6,.8]
        obs["contact"] = [1,0]
        run_code = compile_policy("code", code("[obs['block_angle'][1], obs['contact'][0] * 0.2]"))
        rows = [[0.0]*len(FEATURES) for _ in range(2)]
        rows[0][FEATURES.index("block_sin")] = 1.0
        rows[1][FEATURES.index("tip_block_contact")] = .2
        run_weights = compile_policy("weights", {"layers":[{"weight":rows,"bias":[0,0]}]})
        self.assertEqual(run_code(obs, [.15,.04]), [.8,.2])
        self.assertEqual(run_weights(obs, [.15,.04]), [.8,.2])

    def test_fit_requires_footprint_alignment_and_settle(self):
        base = {"block":[.21,0],"block_angle":[1,0],"block_v":[0,0]}
        self.assertTrue(in_slot(base))
        self.assertFalse(in_slot({**base, "block":[.20,0]}))
        self.assertFalse(in_slot({**base, "block":[.21,.006]}))
        self.assertFalse(in_slot({**base, "block_angle":[math.cos(.25),math.sin(.25)]}))
        self.assertFalse(in_slot({**base, "block_v":[.02,0]}))


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo required")
class PhysicsTest(unittest.TestCase):
    def test_two_scenes_have_same_body_and_differ_only_in_initial_angle(self):
        train, transfer = SlotPlant(271,"train"), SlotPlant(271,"transfer")
        a,b = train.reset(), transfer.reset()
        self.assertEqual(a.pop("block_angle"), [b["block_angle"][0],-b.pop("block_angle")[1]])
        self.assertEqual(a,b)
        old = Plant(271,"linear")
        old.reset(q=START_Q)
        for cmd in ([.2,0],[.2,0],[0,.2],[0,.2]):
            s = old.step(cmd)
            a, b = train.step(cmd), transfer.step(cmd)
            for key in ("q","dq","tip"):
                for actual, expected in zip(a[key],s[key]):
                    self.assertAlmostEqual(actual,expected,places=12)
                self.assertEqual(a[key],b[key])

    def test_real_pusher_collision_moves_the_block(self):
        plant = SlotPlant(271,"train")
        data = plant._data
        data.qpos[2:] = [.158,.029,0]
        plant._mj.mj_forward(plant._model, data)
        contacted = False
        for _ in range(30):
            state = plant.step([-.4,-.1])
            contacted |= bool(state["contact"][0])
        self.assertTrue(contacted)
        self.assertGreater(math.dist(state["block"], [.158,.029]), .001)

    def test_rail_is_a_real_barrier(self):
        plant = SlotPlant(271,"train")
        data = plant._data
        data.qpos[2:] = [.160,.021,0]
        data.qvel[2] = .12
        plant._mj.mj_forward(plant._model, data)
        contacted = False
        for _ in range(15):
            state = plant.step([0,0])
            contacted |= bool(state["contact"][1])
        self.assertTrue(contacted)
        self.assertLess(state["block"][0], .166)


if __name__ == "__main__":
    unittest.main()
