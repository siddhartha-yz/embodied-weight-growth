"""Same-physics LIBERO-Goal forks with two real native success predicates."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


TASKS = {
    "objects": {
        "bowl": (4, "put_the_bowl_on_top_of_the_cabinet"),
        "bottle": (2, "put_the_wine_bottle_on_top_of_the_cabinet"),
    },
    "destinations": {
        "plate": (8, "put_the_bowl_on_the_plate"),
        "stove": (1, "put_the_bowl_on_the_stove"),
    },
}
OBJECTS = {
    "bowl": "akita_black_bowl_1", "bottle": "wine_bottle_1",
    "plate": "plate_1", "cream_cheese": "cream_cheese_1",
}
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIOCEPTION = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
SETTLE_STEPS = 10
SETTLE_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
_RUNTIME_CONFIG = None


def _runtime(source):
    global _RUNTIME_CONFIG
    source = source or os.environ.get("LIBERO_SOURCE")
    if not source:
        raise ValueError("Set LIBERO_SOURCE to the pinned LIBERO-PRO checkout.")
    source = Path(source).expanduser().resolve(strict=True)
    package = source / "libero/libero"
    if not (package / "bddl_files/libero_goal").is_dir():
        raise ValueError(f"Not an existing LIBERO source checkout: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    if "libero.libero" not in sys.modules:
        # LIBERO otherwise prompts interactively on first import. Keep this
        # evaluator's asset paths separate from the prior ORA run and user HOME.
        _RUNTIME_CONFIG = tempfile.TemporaryDirectory(prefix="paired-libero-config-")
        config = {
            "benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
            "init_states": str(package / "init_files"), "assets": str(package / "assets"),
            "datasets": str(package.parent / "datasets"),
        }
        (Path(_RUNTIME_CONFIG.name) / "config.yaml").write_text(json.dumps(config) + "\n")
        os.environ["LIBERO_CONFIG_PATH"] = _RUNTIME_CONFIG.name
    import libero.libero
    from libero.libero.envs import bddl_utils
    from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv

    if Path(libero.libero.__file__).resolve().parent != package:
        raise RuntimeError("Already imported a different LIBERO source in this interpreter")
    return package, bddl_utils, ControlEnv, OffScreenRenderEnv


class PairedEnv:
    """Single native scene, alternate official language and goal states.

    Only raw cameras and robot proprioception should go to a policy. `status`
    contains host-only task predicates, object poses, and contact telemetry.
    """

    def __init__(self, pair="objects", *, camera_size=256, render_gpu_device_id=-1,
                 source=None, render=True):
        if pair not in TASKS:
            raise ValueError(f"pair must be one of {tuple(TASKS)}")
        import torch

        package, bddl_utils, control_env, offscreen_env = _runtime(source)
        from libero.libero import benchmark

        self.pair = pair
        self.task_ids = {key: item[0] for key, item in TASKS[pair].items()}
        self.task_names = {key: item[1] for key, item in TASKS[pair].items()}
        suite = benchmark.get_benchmark("libero_goal")()
        registry_tasks = {name: suite.get_task(task_id) for name, task_id in self.task_ids.items()}
        if any(task.name != self.task_names[name] or task.bddl_file != f"{self.task_names[name]}.bddl"
               for name, task in registry_tasks.items()):
            raise RuntimeError("LIBERO-Goal registry IDs no longer match the paired tasks")
        self._anchor = next(iter(self.task_names))
        bddls = {name: package / "bddl_files/libero_goal" / f"{task}.bddl"
                 for name, task in self.task_names.items()}
        self._problem = {name: bddl_utils.robosuite_parse_problem(str(path)) for name, path in bddls.items()}
        # LeRobot gives the policy the registry wording, which can differ
        # from the embedded BDDL prose even for the same native goal.
        self.instructions = {name: task.language for name, task in registry_tasks.items()}
        self.bddl_instructions = {name: bddl_utils.get_problem_info(str(path))["language_instruction"]
                                  for name, path in bddls.items()}
        common_fields = {"language", "language_instruction", "obj_of_interest", "goal_state"}
        common = [{k: v for k, v in parsed.items() if k not in common_fields}
                  for parsed in self._problem.values()]
        if any(scene != common[0] for scene in common[1:]):
            raise RuntimeError("Official tasks no longer have identical scene/init BDDL")
        self.predicates = {name: deepcopy(p["goal_state"]) for name, p in self._problem.items()}
        if list(self.predicates.values())[0] == list(self.predicates.values())[1]:
            raise RuntimeError("Paired tasks unexpectedly share one native goal")
        self.init_paths = {name: package / "init_files/libero_goal" / f"{task}.pruned_init"
                           for name, task in self.task_names.items()}
        self.anchor_init_path = self.init_paths[self._anchor]
        self._init_states = {}
        for name, path in self.init_paths.items():
            # Official pruned_init files store numeric NumPy arrays rather
            # than tensors; allow only those array/dtype constructors.
            safe_arrays = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                           type(np.dtype("float32")), type(np.dtype("float64"))]
            with torch.serialization.safe_globals(safe_arrays):
                states = np.asarray(torch.load(path, map_location="cpu", weights_only=True))
            if states.ndim != 2 or not np.isfinite(states).all():
                raise RuntimeError(f"Malformed official LIBERO initial states: {path}")
            self._init_states[name] = states
        if len({states.shape[1] for states in self._init_states.values()}) != 1:
            raise RuntimeError("Paired initial states have incompatible native dimensions")
        self.init_counts = {name: len(states) for name, states in self._init_states.items()}
        self.num_init_states = self.init_counts[self._anchor]
        env_cls = offscreen_env if render else control_env
        self._env = env_cls(
            bddl_file_name=str(bddls[self._anchor]),
            camera_heights=camera_size, camera_widths=camera_size,
            render_gpu_device_id=render_gpu_device_id,
            use_camera_obs=render, has_offscreen_renderer=render, hard_reset=False,
        )
        self.action_dim = int(self._env.env.action_dim)
        if self.action_dim != 7:
            self.close()
            raise RuntimeError(f"Expected 7-D native LIBERO OSC_POSE actions, got {self.action_dim}")
        self.goal = None
        self._state = None

    @property
    def language(self):
        return None if self.goal is None else self.instructions[self.goal]

    def set_goal(self, goal):
        """Switch only the native predicate/language; do not reset or step physics."""
        if goal not in self.predicates:
            raise ValueError(f"Unknown goal {goal!r}; expected {tuple(self.predicates)}")
        self.goal = goal
        native = self._env.env
        for key in ("language", "language_instruction", "obj_of_interest", "goal_state"):
            if key in self._problem[goal]:
                native.parsed_problem[key] = deepcopy(self._problem[goal][key])
        native.obj_of_interest = native.parsed_problem["obj_of_interest"]
        self._env.language_instruction = self.bddl_instructions[goal]

    def reset(self, goal, init_index=0, seed=0, *, init_source=None, layout="canonical"):
        """Use a shared official state, optionally exchanging bowl/bottle x/y."""
        if layout not in ("canonical", "swap_xy"):
            raise ValueError("layout must be 'canonical' or 'swap_xy'")
        if layout == "swap_xy" and self.pair != "objects":
            raise ValueError("swap_xy is defined for the bowl/bottle pair only")
        init_source = self._anchor if init_source is None else init_source
        if init_source not in self._init_states:
            raise ValueError(f"Unknown init_source {init_source!r}; expected {tuple(self._init_states)}")
        if type(init_index) is not int or not 0 <= init_index < self.init_counts[init_source]:
            raise IndexError(f"init_index must be in [0,{self.init_counts[init_source]})")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.set_goal(goal)
        self._env.seed(seed)
        self._env.reset()
        state = self._init_states[init_source][init_index]
        if state.shape != self._env.get_sim_state().shape:
            raise RuntimeError("Official init does not match the paired native state layout")
        obs = self._env.set_init_state(state.copy())
        swapped = self._swap_object_xy() if layout == "swap_xy" else None
        # LeRobot v0.4.4 LIBERO evaluation: settle objects, then explicitly
        # select the standard relative OSC_POSE action controller.
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = self._env.step(SETTLE_ACTION)
        for robot in self._env.robots:
            robot.controller.use_delta = True
        if swapped is not None:
            self._validate_swap_contacts(require_support=True)
            native = self._env.env
            for joint, pose in swapped.items():
                settled = np.asarray(native.sim.data.get_joint_qpos(joint))
                overlap = abs(float(np.dot(pose[3:], settled[3:])))
                # The official states start objects above the table; the
                # standard wait lowers z in both layouts. Compare XY here.
                if np.linalg.norm(settled[:2] - pose[:2]) > 0.005 or overlap < np.cos(0.03):
                    raise RuntimeError(f"Crossed layout did not settle at the intended pose: {joint}")
            if any(all(native._eval_predicate(p) for p in group) for group in self.predicates.values()):
                raise RuntimeError("Crossed layout cannot start in either success state")
        self._steps = 0
        self._init_index = init_index
        self._init_source = init_source
        self._seed = seed
        self._layout = layout
        self._initial_positions = self._positions()
        self._first_grasp = None
        self._first_grasp_step = None
        self._first_goal_commit = None
        self._first_goal_commit_step = None
        self._state = self._observe_status()
        return obs

    def _swap_object_xy(self):
        native = self._env.env
        sim = native.sim
        joints = [native.get_object(OBJECTS[name]).joints[0] for name in ("bowl", "bottle")]
        a, b = [np.asarray(sim.data.get_joint_qpos(joint)).copy() for joint in joints]
        if a.shape != (7,) or b.shape != (7,):
            raise RuntimeError("Expected two free-joint object poses")
        a[:2], b[:2] = b[:2].copy(), a[:2].copy()
        poses = dict(zip(joints, (a, b), strict=True))
        for joint, pose in poses.items():
            sim.data.set_joint_qpos(joint, pose)
        sim.forward()
        self._validate_swap_contacts(require_support=False)
        return poses

    def _validate_swap_contacts(self, *, require_support):
        native = self._env.env
        model, data = native.sim.model, native.sim.data
        table = model.geom_name2id("table_collision")
        target_geoms = {model.geom_name2id(geom): short for short in ("bowl", "bottle")
                        for geom in native.get_object(OBJECTS[short]).contact_geoms}
        supported = set()
        for contact in data.contact[:data.ncon]:
            for obj, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if obj not in target_geoms:
                    continue
                if other != table or contact.dist < -0.002:
                    raise RuntimeError(f"Unsafe crossed-layout contact: {target_geoms[obj]} against "
                                       f"{model.geom_id2name(other)} ({contact.dist:.4g} m)")
                supported.add(target_geoms[obj])
        if require_support and supported != {"bowl", "bottle"}:
            raise RuntimeError(f"Both crossed objects must be supported by the tabletop; got {supported}")

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Reset before stepping a paired episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite 7-D native LIBERO action")
        obs, reward, done, info = self._env.step(action)
        self._steps += 1
        self._state = self._observe_status()
        if bool(done) != self._state["goal_achieved"]:
            raise RuntimeError("Paired goal telemetry differs from native done predicate")
        return obs, float(reward), bool(done), info

    def _positions(self):
        native = self._env.env
        return {short: np.asarray(native.sim.data.body_xpos[native.obj_body_id[name]]).copy()
                for short, name in OBJECTS.items()}

    def _observe_status(self):
        native = self._env.env
        goals = {goal: all(native._eval_predicate(p) for p in predicates)
                 for goal, predicates in self.predicates.items()}
        gripper = native.robots[0].gripper
        contacts = [short for short, full in OBJECTS.items()
                    if native._check_grasp(gripper, native.get_object(full))]
        grasped = contacts[0] if len(contacts) == 1 else ("both" if contacts else None)
        if grasped is not None and self._first_grasp is None:
            self._first_grasp, self._first_grasp_step = grasped, self._steps
        true_goals = [goal for goal, achieved in goals.items() if achieved]
        if true_goals and self._first_goal_commit is None:
            self._first_goal_commit = true_goals[0] if len(true_goals) == 1 else "both"
            self._first_goal_commit_step = self._steps
        positions = self._positions()
        return {
            "goal": self.goal, "language": self.language, "step": self._steps,
            "init_index": self._init_index, "init_source": self._init_source, "seed": self._seed,
            "layout": self._layout,
            **{f"{name}_goal": value for name, value in goals.items()},
            "goal_achieved": goals[self.goal], "other_goal": any(value for name, value in goals.items() if name != self.goal),
            "grasped": grasped, "grasped_objects": contacts,
            "first_grasp": self._first_grasp, "first_grasp_step": self._first_grasp_step,
            "first_goal_commit": self._first_goal_commit, "first_goal_commit_step": self._first_goal_commit_step,
            "object_xyz_m": {name: xyz.tolist() for name, xyz in positions.items()},
            "object_motion_m": {name: float(np.linalg.norm(xyz - self._initial_positions[name]))
                                for name, xyz in positions.items()},
        }

    def status(self):
        """Host-only telemetry. Grasp means contact with both native fingerpads."""
        if self._state is None:
            raise RuntimeError("Reset before reading paired status")
        # A caller may switch goal while continuing the same physical episode.
        self._state = self._observe_status()
        return deepcopy(self._state)

    def sim_state(self):
        """Detached native simulator state for paired-reset assertions only."""
        return self._env.get_sim_state().copy()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
