"""Weights-only numerical execution for the author H1 session (no Python eval)."""
import json
import os
from pathlib import Path

from simulators.humanoidbench_adapter import HumanoidBenchTask
from simulators.weight_policy import WeightPolicy


class HumanoidBenchWeightTask(HumanoidBenchTask):
    def __init__(self, task_name):
        super().__init__(task_name)
        self.policy = None
        self.policy_revision = 0
        self.active_weights = None
        path = os.environ.get("HB_WSP_ARTIFACT_DIR")
        self.artifact_dir = Path(path) if path else None

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.initial_qpos = self.data.qpos.copy()
        return result

    def _step(self, submission):
        trace = []
        error = ""
        try:
            plan = json.loads(submission)
            if not isinstance(plan, dict) or set(plan) != {"op", "steps", "policy", "reset_before"}:
                raise ValueError("Expected a weight rollout plan")
            op, steps, weights, reset_before = (plan[k] for k in ("op", "steps", "policy", "reset_before"))
            if (op not in {"install", "roll", "reset"} or type(steps) is not int
                    or not 0 <= steps <= self.native_horizon
                    or type(reset_before) is not bool or type(weights) is not str):
                raise ValueError("Invalid operation or rollout settings")
            if op == "install":
                if self.execution_phase != "playground":
                    raise ValueError("Only Playground may install new weights")
                candidate = WeightPolicy.from_json(weights, self.channels)
            elif weights or (op == "roll" and steps < 1) or (op == "reset" and (steps or reset_before)):
                raise ValueError("Invalid fields for roll/reset")
            if op == "roll" and self.policy is None:
                raise ValueError("No active policy; install weights first")
            if reset_before or op == "reset":
                self.reset_from_program()
            if op == "install":
                self.policy = candidate
                self.policy_revision += 1
                self.active_weights = weights
                if self.artifact_dir:
                    self.artifact_dir.mkdir(parents=True, exist_ok=True)
                    (self.artifact_dir / f"{self.policy_revision:03d}.json").write_text(weights + "\n")
            for i in range(steps):
                if self.done:
                    break
                a = self.policy.action(self.data.qpos, self.data.qvel, self.initial_qpos,
                                       self.dt * self.attempt_steps)
                self.control_step(a)
                if i % 20 == 19 or self.done or i == steps - 1:
                    trace.append(self.trace_point())
        except (ValueError, TypeError, OverflowError, PermissionError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        frames, self._frames = self._frames, {}
        obs, reward, terminated, truncated, _ = self.transition(self.reward())
        summary = {"revision": self.policy_revision, "trace": trace,
                   "final": self.trace_point(), "fell": terminated, "truncated": truncated}
        return self.finish_step(
            obs, reward, terminated, truncated,
            {"ok": not error, "stdout": json.dumps(summary, separators=(",", ":")),
             "stderr": error, "native_steps": self.control_steps},
            frames=frames, success=self.success,
        )

    def trace_point(self):
        return {**self.stats(), "vx": round(float(self.data.qvel[0]), 6),
                "vy": round(float(self.data.qvel[1]), 6),
                "quat": self.data.qpos[3:7].round(6).tolist(),
                "angular_v": self.data.qvel[3:6].round(6).tolist()}
