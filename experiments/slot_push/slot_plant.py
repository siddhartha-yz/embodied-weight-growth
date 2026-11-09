"""Contact-rich tabletop scene with the same two-link body as the W pilot."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "w_writer"))
from plant import (ARM_LENGTHS, DT, JOINT_LIMIT, TORQUE_SCALE, _SUBSTEPS,
                   _Transmission, _vector2, tip_from_q)

START_Q = (1.03, -1.75)
START_BLOCK = (0.148, 0.0)
START_ANGLE = {"train": math.radians(30), "transfer": math.radians(-30)}
PUSHER_RADIUS = 0.005
BLOCK_SIZE = (0.038, 0.020)
RAILS = ((0.183, -0.025, 0.246, -0.0145), (0.183, 0.0145, 0.246, 0.025))
TARGET = (0.183, -0.0145, 0.240, 0.0145)
BOUNDS = {"x": [0.08, 0.247], "y": [-0.095, 0.095]}
MAX_ALIGNMENT = math.radians(9)
_MAX_SETTLED_SPEED = 0.014


def _rail_xml(index, rect):
    x0, y0, x1, y1 = rect
    return (f'<geom name="rail{index}" type="box" pos="{(x0+x1)/2} {(y0+y1)/2} 0" '
            f'size="{(x1-x0)/2} {(y1-y0)/2} .010" contype="4" conaffinity="3"/>')

_XML = f"""
<mujoco model="slot_push">
  <compiler angle="radian"/>
  <option timestep="{DT/_SUBSTEPS}" gravity="0 0 0" integrator="implicitfast"
          cone="elliptic" iterations="70" tolerance="1e-10"/>
  <default>
    <geom friction=".34 .005 .0001" solref=".012 1" solimp=".94 .99 .0005"/>
    <motor ctrllimited="true" ctrlrange="-1.6 1.6" gear="{TORQUE_SCALE}"/>
    <default class="arm">
      <joint type="hinge" axis="0 0 1" range="{-JOINT_LIMIT} {JOINT_LIMIT}"
             limited="true" damping=".045" armature=".0018"/>
      <geom contype="0" conaffinity="0"/>
    </default>
  </default>
  <worldbody>
    <body name="upper" childclass="arm">
      <joint name="shoulder"/>
      <geom type="capsule" size=".012" fromto="0 0 0 {ARM_LENGTHS[0]} 0 0" mass=".16"/>
      <body name="lower" pos="{ARM_LENGTHS[0]} 0 0">
        <joint name="elbow"/>
        <geom type="capsule" size=".010" fromto="0 0 0 {ARM_LENGTHS[1]} 0 0" mass=".10"/>
        <geom name="pusher" type="sphere" pos="{ARM_LENGTHS[1]} 0 0" size="{PUSHER_RADIUS}"
              mass="0" contype="1" conaffinity="6"/>
      </body>
    </body>
    <body name="block">
      <joint name="block_x" type="slide" axis="1 0 0" damping=".12" frictionloss=".021"/>
      <joint name="block_y" type="slide" axis="0 1 0" damping=".12" frictionloss=".021"/>
      <joint name="block_yaw" type="hinge" axis="0 0 1" damping=".00007" frictionloss=".00012"/>
      <geom name="block_geom" type="box" size="{BLOCK_SIZE[0]/2} {BLOCK_SIZE[1]/2} .007"
            mass=".025" contype="2" conaffinity="5"/>
    </body>
    {"".join(_rail_xml(i, rect) for i, rect in enumerate(RAILS))}
  </worldbody>
  <actuator>
    <motor name="joint_torque_0" joint="shoulder"/>
    <motor name="joint_torque_1" joint="elbow"/>
  </actuator>
</mujoco>
"""


def block_corners(center, angle):
    x, y = center
    c, s = angle
    half_length, half_width = (d/2 for d in BLOCK_SIZE)
    return [(x+c*u-s*v, y+s*u+c*v)
            for u in (-half_length, half_length) for v in (-half_width, half_width)]


def in_slot(state):
    """The entire block fits inside, near-parallel, at a low translational speed."""
    x0, y0, x1, y1 = TARGET
    if abs(state["block_angle"][0]) < math.cos(MAX_ALIGNMENT):
        return False
    if math.hypot(*state["block_v"]) > _MAX_SETTLED_SPEED:
        return False
    return all(x0 <= x <= x1 and y0 <= y <= y1
               for x, y in block_corners(state["block"], state["block_angle"]))


class SlotPlant:
    """The pusher sphere and block interact only through MuJoCo contacts.

    The planar sliders/hinge use dry friction and damping for tabletop drag.
    Visual arm links do not contact the table, block, or slot rails.
    """

    def __init__(self, seed, scene):
        import mujoco

        self.scene = scene
        self._mj = mujoco
        self._motor = _Transmission(seed, "linear")
        self._model = mujoco.MjModel.from_xml_string(_XML)
        self._data = mujoco.MjData(self._model)
        self._pusher = self._model.geom("pusher").id
        self._block = self._model.geom("block_geom").id
        self._rails = {self._model.geom(f"rail{i}").id for i in range(len(RAILS))}
        self.reset()

    def geometry(self):
        return {
            "name": self.scene, "units": "m", **BOUNDS, "block_size": list(BLOCK_SIZE),
            "rails": [list(rect) for rect in RAILS], "target": list(TARGET),
            "pusher_radius": PUSHER_RADIUS,
            "max_alignment_deg": math.degrees(MAX_ALIGNMENT),
            "max_settled_block_speed_m_s": _MAX_SETTLED_SPEED,
        }

    def reset(self):
        self._motor.reset()
        self._mj.mj_resetData(self._model, self._data)
        self._data.qpos[:] = [*START_Q, *START_BLOCK, START_ANGLE[self.scene]]
        self._mj.mj_forward(self._model, self._data)
        self.last_tip_wall = False
        return self._observe([0.0, 0.0])

    def step(self, u):
        cmd = np.clip(_vector2(u, "u"), -1.0, 1.0)
        self._data.ctrl[:] = self._motor.apply(cmd)
        contact = [0.0, 0.0]
        self.last_tip_wall = False
        for _ in range(_SUBSTEPS):
            self._mj.mj_step(self._model, self._data)
            for pair in self._data.contact:
                if pair.efc_address < 0:
                    continue
                a, b = pair.geom1, pair.geom2
                if self._pusher in (a, b):
                    if self._block in (a, b):
                        contact[0] = 1.0
                    if a in self._rails or b in self._rails:
                        self.last_tip_wall = True
                if self._block in (a, b) and (a in self._rails or b in self._rails):
                    contact[1] = 1.0
        return self._observe(contact)

    def _observe(self, contact):
        pos = self._data.qpos
        q = pos[:2].tolist()
        yaw = float(pos[4])
        return {
            "q": q, "dq": self._data.qvel[:2].tolist(), "tip": list(tip_from_q(q)),
            "block": pos[2:4].tolist(), "block_angle": [math.cos(yaw), math.sin(yaw)],
            "block_v": self._data.qvel[2:4].tolist(), "contact": contact,
        }
