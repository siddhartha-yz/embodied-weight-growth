"""Exercise the real WSP gate and one native rollout; no SDK model calls."""
import json
from pathlib import Path
import sys
import tempfile

from PIL import Image

from orchestrator.config import load_task
from orchestrator.session import Session
from orchestrator.session_runner import prepare_inputs, validate
from simulators.backend import build


def main():
    base = Path(__file__).resolve().parent
    cfg = load_task(base / "task")
    validate(cfg)
    valid = (base / "smoke-weights.json").read_text()
    with tempfile.TemporaryDirectory(prefix="hb-wsp-smoke-") as directory:
        run = Path(directory)
        prepare_inputs(cfg, run)
        (run / "inputs").rename(run / "workspace")
        (run / "workspace/playground.md").rename(run / "workspace/instruction.md")
        with build(cfg) as env:
            session = Session(env, run, phase="playground", seed=0, budget=6,
                              seconds=120, execution_seconds=15, fps=50)

            def submit(op, steps=0, policy="", reset_before=False):
                payload = json.dumps(dict(op=op, steps=steps, policy=policy, reset_before=reset_before))
                return session.handle({"op": "exec", "files": [{"path": "weights.json", "code": payload}]})

            assert session.status()["native_steps_remaining"] == 2400
            denied = submit("roll", 1)
            assert not denied["ok"] and denied["native_steps_used"] == 0
            denied = submit("install", 1, "print('not allowed')")
            assert not denied["ok"] and denied["native_steps_used"] == 0
            result = submit("install", 1, valid)
            assert result["ok"], result
            assert result["native_steps_used"] == 1 and result["success"] is False
            observation = run / "workspace" / result["observation"]
            assert (observation / "current_cam_head.png").stat().st_size > 0
            assert (observation / "execution_cam_head.mp4").stat().st_size > 0
            reset = submit("roll", 1, reset_before=True)
            assert reset["ok"] and reset["native_steps_used"] == 2, reset
            assert env.stats()["steps"] == 1 and env.policy_revision == 1
            print("Native WSP: no prior policy; rejects code; runs numeric policy; camera and same-budget reset.")
            Image.fromarray(env.render()).save(sys.argv[1])


if __name__ == "__main__":
    main()
