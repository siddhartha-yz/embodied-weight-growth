"""Validated rigid XY offsets from the official LIBERO-Goal initial state."""
import math
import re

import numpy as np

from paired_env import OBJECTS, PairedEnv

_PATTERN = re.compile(r"(bowl|bottle)_xy:([+-]?\d+(?:\.\d+)?),([+-]?\d+(?:\.\d+)?)")
_MAX_OFFSET_M = .15
_TOP = "wooden_cabinet_1_top_side"
_CABINET = "wooden_cabinet_1"


def parse_layout(layout):
    """Return (object name, delta x, delta y), or None for an existing layout."""
    if layout in ("canonical", "swap_xy"):
        return None
    if not isinstance(layout, str):
        raise ValueError("Layout must be a string")
    m = _PATTERN.fullmatch(layout)
    if not m:
        raise ValueError("Expected canonical, swap_xy, or bowl_xy:+0.060,+0.040 (m)")
    name = m[1]
    dx, dy = float(m[2]), float(m[3])
    if not all(math.isfinite(v) and abs(v) <= _MAX_OFFSET_M for v in (dx, dy)) or not (dx or dy):
        raise ValueError(f"Layout offsets must be nonzero and each within ±{_MAX_OFFSET_M:g} m")
    return name, dx, dy


class ObjectOffsetEnv(PairedEnv):
    """Reuse native reset, safety checks, settling, observations, and goal.

    For an offset layout, change a single object's XY free-joint coordinates
    at reset. Optional cabinet telemetry is for the experiment logger only.
    """
    def __init__(self, *args, record_cabinet=False, **kwargs):
        self._pending_xy_offset = None
        self._record_cabinet = bool(record_cabinet)
        super().__init__(*args, **kwargs)

    def reset(self, goal, init_index=0, seed=0, *, init_source=None, layout="canonical"):
        offset = parse_layout(layout)
        if offset is None:
            return super().reset(goal, init_index, seed, init_source=init_source, layout=layout)
        if self.pair != "objects":
            raise ValueError("XY offsets require the bowl/bottle task pair")
        self._pending_xy_offset = offset
        try:
            # The native swap path calls the pose hook before ordinary
            # settling, then verifies support, XY, orientation and no goal.
            obs = super().reset(goal, init_index, seed, init_source=init_source, layout="swap_xy")
        finally:
            self._pending_xy_offset = None
        self._layout = layout
        self._state["layout"] = layout
        return obs

    def _swap_object_xy(self):
        if self._pending_xy_offset is None:
            return super()._swap_object_xy()
        name, dx, dy = self._pending_xy_offset
        native = self._env.env
        sim = native.sim
        joints = [native.get_object(OBJECTS[k]).joints[0] for k in ("bowl", "bottle")]
        poses = {joint: np.asarray(sim.data.get_joint_qpos(joint)).copy() for joint in joints}
        if not all(pose.shape == (7,) for pose in poses.values()):
            raise RuntimeError("Expected native free-joint object poses")
        joint = native.get_object(OBJECTS[name]).joints[0]
        poses[joint][:2] += np.asarray([dx, dy])
        sim.data.set_joint_qpos(joint, poses[joint])
        sim.forward()
        self._validate_swap_contacts(require_support=False)
        return poses

    def _observe_status(self):
        status = super()._observe_status()
        if self._record_cabinet and self.goal in ("bowl", "bottle"):
            status["cabinet_probe"] = self.cabinet_probe()
        return status

    def cabinet_probe(self):
        """Native scorer geometry and actual contact, never used by the policy."""
        env = self._env.env
        data = env.sim.data
        target = env.get_object(OBJECTS[self.goal])
        cabinet = env.get_object(_CABINET)
        xyz = np.asarray(data.body_xpos[env.obj_body_id[OBJECTS[self.goal]]])
        origin = np.asarray(data.get_site_xpos(_TOP))
        matrix = np.asarray(data.get_site_xmat(_TOP)).reshape(3, 3)
        site = env.object_sites_dict[_TOP]
        delta = matrix @ (xyz - origin)  # native SiteObject.under convention
        clearance = np.asarray(site.size)[:2] - np.abs(delta[:2])
        model = env.sim.model
        target_geoms = {model.geom_name2id(name) for name in target.contact_geoms}
        cabinet_geoms = {model.geom_name2id(name) for name in cabinet.contact_geoms}
        contact_points = []
        for c in data.contact[:data.ncon]:
            if (c.geom1 in target_geoms and c.geom2 in cabinet_geoms) or (c.geom2 in target_geoms and c.geom1 in cabinet_geoms):
                p = matrix @ (np.asarray(c.pos) - origin)
                contact_points.append(p.tolist())
        return {
            "contact": bool(env.check_contact(cabinet, target)),
            "contact_site_points_m": contact_points,
            "target_in_site_m": delta.tolist(),
            "xy_margin_m": float(clearance.min()),
            "inside_native_top_region": bool(site.under(origin, matrix, xyz)),
        }


BowlOffsetEnv = ObjectOffsetEnv
