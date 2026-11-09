import unittest

from code_skill import call_skill, validate_source


class SkillSandboxTest(unittest.TestCase):
    def test_executes_feedback_branch_with_state(self):
        source = validate_source("""def decide(o):
    if o['arms']['left']['jaw'] < 0.04:
        return {'op': 'fail_reobserve', 'state': {'stage': 'empty'}}
    return {'op': 'move_tcp', 'xyz': o['dest'], 'yaw': 0}
""")
        inputs = {"arms": {"left": {"jaw": 0.0}}, "dest": [0.0, 0.0, 0.8]}
        self.assertEqual(call_skill(source, inputs)["state"], {"stage": "empty"})
        inputs["arms"]["left"]["jaw"] = 0.2
        self.assertEqual(call_skill(source, inputs)["xyz"], inputs["dest"])

    def test_no_host_home_access(self):
        reply = call_skill("""def decide(o):
    import os
    return {'op': 'done', 'state': {'accessible': os.path.exists(o['host_path'])}}
""", {"host_path": __file__})
        self.assertFalse(reply["state"]["accessible"])

    def test_rejects_unknown_actuator(self):
        with self.assertRaisesRegex(ValueError, "supported op"):
            call_skill("def decide(o):\n    return {'op':'shell'}", {})
