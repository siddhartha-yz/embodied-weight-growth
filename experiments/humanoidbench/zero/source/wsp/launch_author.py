"""Use the author simulator/session/freeze lifecycle with a remote Codex SDK brain."""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

from orchestrator import session_runner
from orchestrator.config import load_task
from orchestrator.replay import assemble_history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("explore",))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--gpu", default=None)
    p.add_argument("--sim-env", type=Path, required=True, help="Native simulator environment directory containing bin/python")
    p.add_argument("--brain", required=True)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--effort", default="high")
    p.add_argument("--seconds", type=int, default=900)
    p.add_argument("--turns", type=int, default=25)
    args = p.parse_args()
    site = Path(__file__).resolve().parent
    run = args.run.resolve()
    cfg = load_task(site / "task")
    cfg = replace(cfg,
        simulator=replace(cfg.simulator, interpreter=str(args.sim_env)),
        agent=replace(cfg.agent, model="gpt-6-astra", reasoning_effort=args.effort,
                      timeout_sec=args.seconds),
        test=replace(cfg.test, generation_timeout_sec=args.seconds))

    def agent_command(_cfg, episode, socket_dir, phase):
        unique = site.name + "-" + phase + "-" + episode.name
        return None, [sys.executable, str(site / "sdk_remote_agent.py"),
            "--episode", str(episode), "--socket", str(socket_dir / "session.sock"),
            "--phase", phase, "--thread", unique, "--brain", args.brain,
            "--token-file", str(args.token_file), "--turns", str(args.turns)]

    def recorded_video_only(_cfg, episode, _experiment, *, phase, seed, gpu=None):
        # Preserve the author's source archive but use actual per-submission
        # camera recordings, not a second simulator replay.
        manifest = assemble_history(episode, phase=phase, seed=seed)
        return {"status": "recorded_live", "segments_total": len(manifest["segments"]),
                "video_source": "workspace/observations/*/execution_cam_*.mp4"}

    session_runner.agent_command = agent_command
    session_runner.run_replay = recorded_video_only
    os.environ["HB_WSP_ARTIFACT_DIR"] = str(run / "installed-weights")
    os.environ["HB_EVO_METRICS"] = str(run / f"walk-{args.phase}.jsonl")
    os.environ["EMBODIEDEVO_USE_RECIPE"] = "0"
    os.environ.pop("EMBODIEDEVO_AGENT_RECOVERY", None)
    return session_runner.explore(cfg, run, gpu=args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
