"""Run official h1-walk-v0 under EmbodiedEvo's author CodeTask."""
import json
import os
from pathlib import Path

import gymnasium as gym
import humanoid_bench  # registers Gymnasium environments
import mujoco
import numpy as np

from simulators.execution import CodeTask
from simulators.humanoidbench_api import HumanoidBenchLowLevelApi

PROMPT = "Make the HumanoidBench H1 walk forward along world +X and stay upright."
EXPLORE_ACTIONS = 2400


class HumanoidBenchTask(CodeTask):
    def __init__(self, task_name):
        super().__init__()
        if task_name != "h1-walk-v0":
            raise ValueError("This adapter is configured for h1-walk-v0")
        self.env = gym.make(task_name, render_mode=None)
        self.configure(self.env)
        native = self.env.unwrapped
        self.model, self.data = native.model, native.data
        self.native_horizon = int(self.env.spec.max_episode_steps)
        self.success_bar = float(native.task.success_bar)
        self.dt = float(native.dt)
        self._task_prompt = PROMPT
        self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        self.initialize(HumanoidBenchLowLevelApi(self))
        self.attempt = 0
        self.max_steps = self.native_horizon
        path = os.environ.get("HB_EVO_METRICS")
        self.metrics_path = Path(path) if path else None
        self.channels = []
        for i, j in enumerate(self.model.actuator_trnid[:, 0]):
            j = int(j)
            self.channels.append({
                "joint": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j),
                "qpos": int(self.model.jnt_qposadr[j]),
                "qvel": int(self.model.jnt_dofadr[j]),
                "min_rad": float(native.action_low[i]),
                "max_rad": float(native.action_high[i]),
            })

    def reset(self, *, seed=None, options=None):
        self.max_steps = (EXPLORE_ACTIONS if self.execution_phase == "playground"
                          else self.native_horizon)
        self.native_step_limit = self.max_steps
        _obs, info = self.env.reset(seed=seed, options=options)
        self.attempt += 1
        self.attempt_steps = 0
        self.total_return = 0.0
        self.initial_x = float(self.data.qpos[0])
        self.horizon = False
        result = self.start_episode(seed, self.snapshot(), reset_info=info, reset_options=options)
        self.record()
        return result

    def stats(self):
        return {
            "attempt": self.attempt,
            "steps": self.attempt_steps,
            "seconds": round(self.dt * self.attempt_steps, 4),
            "x_m": float(self.data.qpos[0] - self.initial_x),
            "root_z_m": float(self.data.qpos[2]),
        }

    def record(self):
        if self.metrics_path:
            row = {**self.stats(), "native_steps": self.control_steps,
                   "return": self.total_return, "fell": self.terminated,
                   "horizon": self.horizon, "success": self.success}
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    def snapshot(self):
        self.renderer.update_scene(self.data, camera="cam_default")
        return {
            "instruction": self._task_prompt,
            "state": {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()},
            "action": {"channels": self.channels},
            "additional_info": self.stats(),
            "vision": {"cam_head": {"rgb": self.renderer.render()}},
        }

    def control_step(self, action):
        if self.done:
            return self.transition()
        command = np.asarray(action, dtype=np.float64)
        if command.shape != self.env.action_space.shape or not np.isfinite(command).all():
            raise ValueError("Expected a finite 19-element action vector")
        if (np.abs(command) > 1).any():
            raise ValueError("Position commands must be in [-1,1]")
        _obs, reward, terminated, truncated, _info = self.env.step(command)
        self.control_steps += 1
        self.attempt_steps += 1
        self.total_return += float(reward)
        self.horizon = bool(truncated)
        self.terminated = bool(terminated)
        self.truncated = self.horizon or self.control_steps >= self.max_steps
        # Keep the raw native return private; only publish the completed target.
        self.success = self.horizon and not self.terminated and self.total_return >= self.success_bar
        self.observe(self.snapshot())
        self.record()
        return self.transition()

    def reward(self):
        return self.total_return

    def close(self):
        self.renderer.close()
        self.env.close()
