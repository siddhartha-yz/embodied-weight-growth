"""Check an unchanged WSP candidate beyond an early-stopped pilot rollout."""
import json
import os
from pathlib import Path
import sys

from orchestrator.config import load_task
from orchestrator.session import Session
from simulators.backend import build


def main(policy, output):
    root = Path(__file__).resolve().parent
    cfg = load_task(root / "task")
    output.mkdir()
    workspace = output / "workspace"
    workspace.mkdir()
    os.environ["HB_EVO_METRICS"] = str(output / "native.jsonl")
    with build(cfg) as env:
        session = Session(env, output, phase="playground", seed=0, budget=1,
                          seconds=120, execution_seconds=60, fps=50)
        payload = json.dumps(dict(op="install", steps=1000, policy=policy.read_text(),
                                  reset_before=False))
        result = session.handle({"op": "exec", "files": [{"path": "weights.json", "code": payload}]})
        if not result["ok"]:
            raise RuntimeError(result)
        session.handle({"op": "finish"})
        print(json.dumps({**env.stats(), "fell": env.terminated, "horizon": env.horizon,
                          "return": env.total_return}, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]).resolve())
