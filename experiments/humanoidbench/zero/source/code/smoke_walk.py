"""Preflight the real author session with a single native position action."""
from pathlib import Path
import tempfile

from orchestrator.config import load_task
from orchestrator.session import Session
from orchestrator.session_runner import prepare_inputs, validate
from simulators.backend import build


def main():
    cfg = load_task(Path(__file__).resolve().parent / "task")
    validate(cfg)
    with tempfile.TemporaryDirectory(prefix="hb-evo-smoke-") as directory:
        run = Path(directory)
        prepare_inputs(cfg, run)
        (run / "inputs").rename(run / "workspace")
        (run / "workspace" / "playground.md").rename(run / "workspace" / "instruction.md")
        with build(cfg) as env:
            session = Session(env, run, phase="playground", seed=0, budget=4,
                              seconds=120, execution_seconds=15, fps=50)
            initial = session.status()
            assert initial["native_steps_remaining"] == 2400
            result = session.handle({
                "op": "exec", "files": [{"path": "submission/solution.py",
                                        "code": "o, r, t, u, info = step([0.0] * 19)\nprint(len(o['state']['qpos']),len(o['state']['qvel']))"}],
            })
            assert result["ok"], result
            assert result["stdout"].strip() == "26 25", result
            assert result["native_steps_used"] == 1, result
            assert result["success"] is False, result
            observation = run / "workspace" / result["observation"]
            assert (observation / "current_cam_head.png").stat().st_size > 0
            assert (observation / "execution_cam_head.mp4").stat().st_size > 0
            reset = session.handle({"op": "reset"})
            assert reset["ok"] and session.status()["native_steps_used"] == 1, reset
            assert env.stats()["steps"] == 0
            print("Native H1 author session: 51 states / 19 actions; one step, RGB, MP4 and non-renewing reset.")
            return env.render()


if __name__ == "__main__":
    from PIL import Image
    import sys

    Image.fromarray(main()).save(sys.argv[1])
