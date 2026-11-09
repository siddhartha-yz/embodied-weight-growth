"""Trusted side of the body-growth pilot: a horizontal two-link MuJoCo arm.

An agent can ask the host to reset or step, but receives only encoder angles,
encoder velocities, and the base-relative Cartesian tip location. A reset
changes the starting pose, never the sampled body. The host keeps the seed,
motor mixing matrix, and transmission state private.
"""

from __future__ import annotations

import math

import numpy as np

DT = 0.02  # seconds per public step
ARM_LENGTHS = (0.14, 0.11)  # metres
TORQUE_SCALE = 0.12  # N m, before the private motor mixing
JOINT_LIMIT = 2.55  # radians
HOME = (0.45, -0.85)
_SUBSTEPS = 4


def tip_from_q(q) -> tuple[float, float]:
    """Forward kinematics for two relative, counter-clockwise joint angles."""
    a, b = q
    l1, l2 = ARM_LENGTHS
    return l1 * math.cos(a) + l2 * math.cos(a + b), l1 * math.sin(a) + l2 * math.sin(a + b)


_XML = f"""
<mujoco model="body_growth_reacher">
  <compiler angle="radian"/>
  <option timestep="{DT / _SUBSTEPS}" gravity="0 0 0" integrator="implicitfast">
    <flag contact="disable"/>
  </option>
  <default>
    <joint type="hinge" axis="0 0 1" range="{-JOINT_LIMIT} {JOINT_LIMIT}"
           limited="true" damping=".045" armature=".0018"/>
    <motor ctrllimited="true" ctrlrange="-1.6 1.6" gear="{TORQUE_SCALE}"/>
  </default>
  <worldbody>
    <body name="upper">
      <joint name="shoulder"/>
      <geom type="capsule" size=".012" fromto="0 0 0 {ARM_LENGTHS[0]} 0 0" mass=".16"/>
      <body name="lower" pos="{ARM_LENGTHS[0]} 0 0">
        <joint name="elbow"/>
        <geom type="capsule" size=".010" fromto="0 0 0 {ARM_LENGTHS[1]} 0 0" mass=".10"/>
        <site name="tip" pos="{ARM_LENGTHS[1]} 0 0" size=".006"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint_torque_0" joint="shoulder"/>
    <motor name="joint_torque_1" joint="elbow"/>
  </actuator>
</mujoco>
"""


class _Transmission:
    """Sample a body once; reset only its transient transmission state."""

    def __init__(self, seed: int, condition: str):
        if condition not in {"linear", "backlash"}:
            raise ValueError("condition must be 'linear' or 'backlash'")
        rng = np.random.default_rng(seed)
        diag = rng.uniform(0.76, 1.14, size=2)
        cross = rng.uniform(0.14, 0.30, size=2) * rng.choice((-1, 1), size=2)
        self.mix = np.array([[diag[0], cross[0]], [cross[1], diag[1]]])
        self.condition = condition
        self.slack = rng.uniform(0.012, 0.037, size=2)  # normalized command × seconds
        tau = rng.uniform(0.02, 0.055, size=2)
        self.response = -np.expm1(-DT / tau)
        self.reset()

    def reset(self) -> None:
        self.direction = np.zeros(2)
        self.gap = np.zeros(2)
        self.engaged = np.zeros(2)

    def apply(self, cmd: np.ndarray) -> np.ndarray:
        if self.condition == "linear":
            return self.mix @ cmd
        # A stylized lost-motion transmission: on reversal, command first
        # consumes slack. A zero command doesn't select a new direction.
        cmd = np.where(np.abs(cmd) > 0.03, cmd, 0.0)
        sign = np.sign(cmd)
        reversing = (sign != 0) & (self.direction != 0) & (sign != self.direction)
        self.gap[reversing] = self.slack[reversing]
        self.direction[sign != 0] = sign[sign != 0]
        travel = np.abs(cmd) * DT
        through = np.maximum(travel - self.gap, 0.0)
        self.gap = np.maximum(self.gap - travel, 0.0)
        fraction = np.divide(through, travel, out=np.zeros(2), where=travel > 0)
        target = fraction * cmd
        self.engaged += self.response * (target - self.engaged)
        return self.mix @ self.engaged


def _vector2(value, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.shape != (2,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite two-element vector")
    return out


class Plant:
    """One unknown fixed body. Units: radians, radians/second, metres."""

    def __init__(self, seed: int, condition: str):
        import mujoco

        self._mujoco = mujoco
        self._motor = _Transmission(seed, condition)
        self._model = mujoco.MjModel.from_xml_string(_XML)
        self._data = mujoco.MjData(self._model)
        self.reset()

    def reset(self, q=HOME) -> dict:
        start = _vector2(q, "q")
        if np.any(np.abs(start) > JOINT_LIMIT):
            raise ValueError(f"starting pose must lie within ±{JOINT_LIMIT} rad")
        self._motor.reset()
        self._mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:] = start
        self._mujoco.mj_forward(self._model, self._data)
        return self._observe()

    def step(self, u) -> dict:
        cmd = np.clip(_vector2(u, "u"), -1.0, 1.0)
        self._data.ctrl[:] = self._motor.apply(cmd)
        for _ in range(_SUBSTEPS):
            self._mujoco.mj_step(self._model, self._data)
        return self._observe()

    def _observe(self) -> dict:
        q = self._data.qpos.tolist()
        return {"q": q, "dq": self._data.qvel.tolist(), "tip": list(tip_from_q(q))}
