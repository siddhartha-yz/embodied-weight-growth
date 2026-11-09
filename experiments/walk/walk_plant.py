"""Direct torque access to a sagittal Walker2d, without a stabilizer or gait."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

DT = 0.02
PITCH = {"train": 0.02, "transfer": 0.08}
TARGET_V = 0.5
ASSET = Path(__file__).with_name("walker2d.xml")
PARTS = (
    ("torso", "torso_geom"), ("thigh_r", "thigh_geom"),
    ("shin_r", "leg_geom"), ("foot_r", "foot_geom"),
    ("thigh_l", "thigh_left_geom"), ("shin_l", "leg_left_geom"),
    ("foot_l", "foot_left_geom"),
)


class WalkPlant:
    def __init__(self, seed=271, scene="train"):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(ASSET))
        self.data = mujoco.MjData(self.model)
        self.scene = scene
        self.substeps = round(DT / self.model.opt.timestep)
        self.parts = [(name, self.model.geom(geom).id) for name, geom in PARTS]
        self.floor = self.model.geom("floor").id
        self.feet = [self.model.geom("foot_geom").id, self.model.geom("foot_left_geom").id]
        self.torso = self.model.geom("torso_geom").id
        rng = np.random.default_rng(seed)
        # Slightly bent, nearly upright pose; same sampled state for every reset.
        # The transfer scene changes only the root's initial pitch coordinate.
        self.start_q = np.array([0.0, 1.225, PITCH[scene], -.03, -.04, 0, -.04, -.035, 0])
        self.start_q[3:] += rng.uniform(-0.002, 0.002, size=6)
        self.reset()

    def reset(self):
        self._mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.start_q
        self._mj.mj_forward(self.model, self.data)
        return self.observe()

    def step(self, u):
        self.data.ctrl[:] = u
        for _ in range(self.substeps):
            self._mj.mj_step(self.model, self.data)
        self._mj.mj_forward(self.model, self.data)
        return self.observe()

    def ground_contacts(self):
        grounded = set()
        for pair in self.data.contact:
            if pair.efc_address < 0:
                continue
            a, b = pair.geom1, pair.geom2
            if a == self.floor:
                grounded.add(b)
            elif b == self.floor:
                grounded.add(a)
        return grounded

    def observe(self):
        data = self.data
        q = data.qpos.tolist()
        dq = data.qvel.tolist()
        grounded = self.ground_contacts()
        t = float(data.time)
        pose = []
        for name, geom in self.parts:
            center = data.geom_xpos[geom][[0, 2]]
            axis = data.geom_xmat[geom].reshape(3, 3)[[0, 2], 2]
            radius, half_length = self.model.geom_size[geom, :2]
            offset = axis * half_length
            pose.append({"name": name, "start": (center - offset).tolist(),
                         "end": (center + offset).tolist(), "radius": float(radius)})
        return {
            "x": q[0], "z": q[1], "pitch": q[2], "vx": dq[0], "vz": dq[1], "pitch_v": dq[2],
            "q": q[3:], "dq": dq[3:], "contact": [int(geom in grounded) for geom in self.feet],
            "t": t, "clock_sin": math.sin(math.tau * t), "clock_cos": math.cos(math.tau * t),
            "pose": pose,
        }

    def fall_reason(self, state):
        if self.torso in self.ground_contacts():
            return "torso touched the floor"
        if state["z"] < 0.8:
            return "torso height dropped below 0.8 m"
        if abs(state["pitch"]) > 1.0:
            return "torso tilted beyond 1.0 rad"
        return None
