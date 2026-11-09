"""Host-side bridge to the installed RoboDojo worker. No task policy lives here."""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import select
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid


WIRE = b"AURIGA_JSON "
RUN_DIR = b"PHYSEVO_RUN_DIR "
OBSERVATION_PROFILE = "rgb_proprio_calibrated_v1"
OBSERVATION_FIELDS = (
    "raw_observation_id", "data_origin", "world_kind", "observation_profile", "robot_model_id",
    "cameras", "robot_state", "sim_step", "step_domain", "control_dt_s", "sim_time_s",
    "instruction", "camera_calibration", "robot_model", "camera_pose_sample",
)
RECEIPT_FIELDS = (
    "status", "reason", "executed_steps", "executed_native_steps", "executed_control_steps",
    "completed_native_steps", "physics_advanced", "execution_interval", "gripper_execution",
)

# This starts the existing worker, not the Auriga agent or its policy/tools.
LAUNCH = r"""
set -eu
umask 077
base=${1:-"$HOME/data/robodojo-work"}
parent=${2:-"$HOME/data/physevo-wsp-runs"}
gpu=$3
task=$4
seed=$5
layout=$6
profile=$7
mkdir -p -- "$parent"
run=$(mktemp -d -- "$parent/$task.XXXXXX")
mkdir -- "$run/tmp"
printf 'PHYSEVO_RUN_DIR %s\n' "$run"
export TMPDIR="$run/tmp"
export CUDA_VISIBLE_DEVICES="$gpu" ROBODOJO_RENDER_GPU="$gpu"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N
export PYTHONDONTWRITEBYTECODE=1
# Match the cached-asset preflight: no user proxy, while local IPC still works.
export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
export PYTHONPATH="$base/src/Auriga:$base/src/RoboDojo:$base/src/RoboDojo/XPolicyLab"
export PATH="$base/miniforge3/envs/RoboDojo/bin:$PATH"
cd -- "$base/src/RoboDojo"
exec "$base/miniforge3/envs/RoboDojo/bin/python" -u -m auriga.robot.backends.robodojo.robot_worker \
    --robodojo-root "$base/src/RoboDojo" --robot-model-id arx-x5 --gpu "$gpu" \
    --task "$task" --run-dir "$run" --seed "$seed" --layout-id "$layout" \
    --observation-profile "$profile"
"""


class WorkerError(RuntimeError):
    """The worker rejected an RPC; this does not necessarily lose the episode."""


def _pick(value, fields):
    return {k: value[k] for k in fields if k in value}


def _name(name, label):
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        raise ValueError(f"invalid {label}: {name!r}")
    return name


def _vector(values, n, label):
    values = list(values)
    if len(values) != n or any(
        isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in values
    ):
        raise ValueError(f"{label} must contain {n} finite numbers")
    return values


def ee_action(ready, arm, xyz=None, quaternion_xyzw=None, grip=None):
    """Build one absolute Link6 target or standalone jaw command (0 closed, 1 open)."""
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    if (xyz is None) != (quaternion_xyzw is None):
        raise ValueError("position and xyzw quaternion must be supplied together")
    points = []
    if xyz is not None:
        quat = _vector(quaternion_xyzw, 4, "quaternion_xyzw")
        if abs(math.sqrt(sum(x*x for x in quat)) - 1) > 1e-3:
            raise ValueError("quaternion_xyzw must be normalized")
        points.append({arm: {"position_m": _vector(xyz, 3, "xyz"), "quaternion_xyzw": quat}})
    events = []
    if grip is not None:
        goal, = _vector([grip], 1, "grip")
        if not 0 <= goal <= 1:
            raise ValueError("grip must be in [0, 1]")
        events.append({"arm": arm, "phase": "after_motion", "goal_normalized": goal})
    if not points and not events:
        raise ValueError("a pose or jaw goal is required")
    return {
        "type": "EEAction", "schema_version": "0.1", "action_id": f"physevo-{uuid.uuid4().hex}",
        "mode": "absolute", "world_kind": "observed", "frame_id": ready["frame_id"],
        "tcp_config_id": ready["tcp_config_id"], "limits_profile": ready["limits_profile"],
        "path": {"kind": "backend_free" if points else "hold", "orientation": "backend_free",
                 "blend_radius_m": 0.0, "position_tolerance_m": 0.005, "waypoints": points},
        "gripper_events": events, "timeout_s": 20,
    }


def joint_action(rows):
    """Build a native joint prefix: [left q × 6, left jaw, right q × 6, right jaw]."""
    actions = [_vector(row, 14, "native joint row") for row in rows]
    if not 1 <= len(actions) <= 50:
        raise ValueError("a joint prefix requires 1 to 50 native rows")
    if any(not 0 <= row[jaw] <= 1 for row in actions for jaw in (6, 13)):
        raise ValueError("native jaw goals must be in [0, 1]")
    return {
        "type": "JointPrefixAction", "schema_version": "0.1", "action_id": f"physevo-{uuid.uuid4().hex}",
        "actions": actions, "policy_provenance": {"source": "experiment_caller"}, "timeout_s": 20,
    }


def _joint_ramp(ready, state, arm, joints, grip, steps):
    """Expand a single absolute joint target at hardware command-rate limits."""
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    target = _vector(joints, 6, "joint target (radians)")
    if steps is not None and (type(steps) is not int or not 1 <= steps <= 250):
        raise ValueError("steps must be an integer in [1, 250]")
    limits = ready["arm_joint_limits_rad"][arm]
    if any(not lo <= value <= hi for value, (lo, hi) in zip(target, limits, strict=True)):
        raise ValueError("joint target exceeds native joint limits")
    poses = state["arms"]
    held = {side: [*poses[side]["q_rad"], poses[side]["gripper"]["goal_normalized"]]
            for side in ("left", "right")}
    initial = held[arm]
    goal, = _vector([initial[6] if grip is None else grip], 1, "grip")
    if not 0 <= goal <= 1:
        raise ValueError("grip must be in [0, 1]")
    target.append(goal)
    delta = [b-a for a, b in zip(initial, target, strict=True)]
    rates = [*ready["arm_max_step_rad"][arm], ready["gripper_max_step"]]
    count = max(steps or 1, *(math.ceil(abs(d) / rate) for d, rate in zip(delta, rates, strict=True)))
    if count > 250:
        raise ValueError("joint target requires more than 250 native steps")
    rows = []
    for k in range(1, count + 1):
        point = target if k == count else [a + d*k/count for a, d in zip(initial, delta, strict=True)]
        rows.append([*(point if arm == "left" else held["left"]), *(point if arm == "right" else held["right"])])
    return rows


class Native:
    """One sequential, long-lived SSH/worker session; the caller owns policy and scoring."""

    def __init__(self, out_dir, *, gpu, host="robot-sim", task="arrange_largest_number", seed=0,
                 layout_id=0, base=None, remote_parent=None, startup_timeout=240, rpc_timeout=90,
                 close_timeout=180):
        self.out_dir = Path(out_dir)
        self.host = host
        self.task = _name(task, "task name")
        for key, value in (("gpu", gpu), ("seed", seed), ("layout_id", layout_id)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        self.gpu, self.seed, self.layout_id = gpu, seed, layout_id
        self.base = base if base is not None else os.environ.get("PHYSEVO_ROBODOJO_BASE", "")
        self.remote_parent = remote_parent if remote_parent is not None else os.environ.get("PHYSEVO_REMOTE_RUNS", "")
        for p in (self.base, self.remote_parent):
            if p and not PurePosixPath(p).is_absolute():
                raise ValueError("remote overrides must be absolute paths")
        self.startup_timeout, self.rpc_timeout = startup_timeout, rpc_timeout
        self.close_timeout = close_timeout
        self.close_report = None
        self.ready = None
        self.remote_dir = None
        self._proc = self._stderr = None
        self._buffer = bytearray()
        self._faulted = False
        self._seq = self._episode = 0

    def _ssh(self, command):
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "--", self.host, command]

    def start(self):
        if self._proc is not None:
            raise RuntimeError("worker already started")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._episode += 1
        self._buffer.clear()
        self._faulted = False
        self.remote_dir = self.ready = None
        self.close_report = None
        self.log_path = self.out_dir / f"worker-{self._episode:03d}.stderr.log"
        self._stderr = self.log_path.open("xb")
        try:
            command = shlex.join([
                "bash", "-c", LAUNCH, "physevo-worker", str(self.base), str(self.remote_parent),
                str(self.gpu), self.task, str(self.seed), str(self.layout_id), OBSERVATION_PROFILE,
            ])
            self._proc = subprocess.Popen(self._ssh(command), stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, stderr=self._stderr)
            until = time.monotonic() + self.startup_timeout
            while True:
                line = self._readline(until)
                if line.startswith(RUN_DIR):
                    path = PurePosixPath(os.fsdecode(line[len(RUN_DIR):]))
                    if not path.is_absolute():
                        raise RuntimeError("worker announced a non-absolute run directory")
                    self.remote_dir = str(path)
                    break
            self.ready = self.call("ready", timeout=max(0.001, until - time.monotonic()))
            return self
        except BaseException:
            self._terminate()
            raise

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception as close_error:
                exc.add_note(f"closing RoboDojo worker: {close_error}")

    def _readline(self, until):
        assert self._proc is not None and self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                return line
            wait = until - time.monotonic()
            if wait <= 0 or not select.select([fd], [], [], wait)[0]:
                raise TimeoutError(f"RoboDojo reply timed out; see {self.log_path}")
            block = os.read(fd, 65536)
            if not block:
                raise RuntimeError(f"RoboDojo worker closed its stream; see {self.log_path}")
            self._buffer.extend(block)

    def call(self, op, payload=None, *, timeout=None, _on_progress=None):
        """Return one terminal RPC result; intermediate worker progress is not actor input."""
        if self._proc is None or self._faulted:
            raise RuntimeError("no usable RoboDojo worker")
        assert self._proc.stdin is not None
        self._seq += 1
        request_id = f"physevo-{self._seq}"
        line = json.dumps({"id": request_id, "op": op, "payload": {} if payload is None else payload},
                          allow_nan=False).encode() + b"\n"
        until = time.monotonic() + (self.rpc_timeout if timeout is None else timeout)
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
            while True:
                msg = self._readline(until)
                if not msg.startswith(WIRE):
                    continue
                response = json.loads(msg[len(WIRE):])
                if response.get("id") != request_id:
                    raise RuntimeError(f"unexpected worker response id for {op}")
                if "error" in response:
                    raise WorkerError(f"{op}: {response['error']}")
                if "result" in response:
                    return response["result"]
                if "progress" not in response:
                    raise RuntimeError(f"malformed worker response to {op}")
                if _on_progress is not None:
                    _on_progress(response["progress"])
        except WorkerError:
            raise
        except BaseException:
            # A lost acknowledgement does not say whether native motion happened.
            self._faulted = True
            raise

    def _terminate(self):
        proc, self._proc = self._proc, None
        try:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                proc.stdin.close()
                proc.stdout.close()
        finally:
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None

    def close(self):
        proc = self._proc
        if proc is None:
            return self.close_report
        report = self.close_report = {"native_close_acknowledged": False, "progress": [],
                                      "ssh_returncode": None}
        try:
            if not self._faulted and proc.poll() is None:
                # The ACK follows Isaac's full extension shutdown. Use the
                # same deadline as the installed Auriga RobotProcess owner.
                reply = self.call("close", timeout=self.close_timeout,
                                  _on_progress=report["progress"].append)
                if reply.get("closed") is not True:
                    raise RuntimeError("worker did not confirm closure")
                report["native_close_acknowledged"] = True
                if proc.wait(timeout=30) != 0:
                    raise RuntimeError("RoboDojo SSH exited unsuccessfully after close acknowledgement")
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            self._terminate()
            report["ssh_returncode"] = proc.returncode
        return report

    def reset(self):
        """Use the native fresh-worker reset for the same seed and layout."""
        self.close()
        self.start()
        return self.observe()

    def observe(self):
        """Read calibrated RGB and proprioception; no privileged evaluator call."""
        return _pick(self.call("observe"), OBSERVATION_FIELDS)

    def move(self, arm, xyz, quaternion_xyzw, grip=None):
        return _pick(self.call("execute", ee_action(self.ready, arm, xyz, quaternion_xyzw, grip)), RECEIPT_FIELDS)

    def grip(self, arm, value):
        return _pick(self.call("execute", ee_action(self.ready, arm, grip=value)), RECEIPT_FIELDS)

    def joint_prefix(self, rows):
        """Execute 1–50 caller-selected native rows, without Cartesian planning."""
        return _pick(self.call("execute", joint_action(rows)), RECEIPT_FIELDS)

    def move_joints(self, arm, joints, grip=None, *, steps=None):
        """Ramp to six absolute joint angles (rad) and an optional normalized jaw.

        The other arm holds its measured joints and latched jaw goal. The
        optional steps is a minimum, not a command-rate override. This direct
        joint path has no collision-planner guarantee.
        """
        rows = _joint_ramp(self.ready, self.observe()["robot_state"], arm, joints, grip, steps)
        totals = {}
        interval = None
        for start in range(0, len(rows), 50):
            receipt = self.joint_prefix(rows[start:start + 50])
            for key in ("executed_steps", "executed_native_steps", "executed_control_steps", "completed_native_steps"):
                if key in receipt:
                    totals[key] = totals.get(key, 0) + receipt[key]
            if "execution_interval" in receipt:
                part = receipt["execution_interval"]
                if interval is None:
                    interval = dict(part)
                else:
                    interval["end_sim_step"] = part["end_sim_step"]
            if receipt.get("status") != "completed":
                break
        result = {**receipt, **totals}
        if interval is not None:
            result["execution_interval"] = interval
        if "physics_advanced" in result and interval is not None:
            result["physics_advanced"] = interval["end_sim_step"] != interval["start_sim_step"]
        return result

    def pose_preview(self, arm, xyz, quat):
        """Plan one Link6 pose without executing; return the endpoint and sampled native rows."""
        if not self.ready.get("motion_preview", {}).get("available"):
            raise RuntimeError("installed worker does not offer native pose preview")
        result = self.call("preview_motion", ee_action(self.ready, arm, xyz, quat))
        trace = result.get("planned_trace", {})
        trajectory = trace.get("q_rad", [])
        return {
            **_pick(result, ("status", "reason", "planned_native_steps", "physics_advanced")),
            "joint_names": trace.get("joint_names", []),
            "q_rad": trajectory[-1] if trajectory else None,
            "trajectory_q_rad": trajectory,
            "native_rows": result.get("native_rows", []),
        }

    def begin_recording(self):
        """Start native RGB recording before actions; no physics advance."""
        return self.call("record", {"enabled": True})

    def fetch_recording(self, directory=None, *, camera_ids=("cam_head",)):
        """Export and copy a live worker's observer-only MP4s and frame manifest."""
        if self.remote_dir is None:
            raise RuntimeError("worker run directory is not available")
        requested = list(camera_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("camera_ids must be nonempty and unique")
        manifest = self.call("save_video", timeout=180)
        videos = {video["camera_id"]: video for video in manifest["videos"]}
        root = PurePosixPath(self.remote_dir) / "videos"
        dst = Path(directory) if directory is not None else self.out_dir / "video"
        dst.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in requested:
            _name(name, "camera id")
            path = PurePosixPath(videos[name]["video_path"])
            if path.parent.parent != root or path.parent.name == ".." or path.name != f"{name}.mp4":
                raise ValueError(f"video path is not in this worker's output: {path}")
            target = dst / path.name
            with tempfile.TemporaryDirectory(dir=dst) as staging:
                temporary = Path(staging) / target.name
                with temporary.open("wb") as stream:
                    subprocess.run(self._ssh(shlex.join(["cat", "--", str(path)])),
                                   check=True, stdout=stream, stderr=subprocess.PIPE, timeout=120)
                temporary.replace(target)
            files[name] = target
        (dst / "video-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return files

    def fetch_images(self, observation, directory=None, *, camera_ids=None):
        """Copy only the requested RGB frames for this worker's observation."""
        if self.remote_dir is None:
            raise RuntimeError("worker run directory is not available")
        obs_id = _name(observation["raw_observation_id"], "observation id")
        remote = PurePosixPath(self.remote_dir) / "observations" / obs_id
        cameras = {c["camera_id"]: c for c in observation["cameras"]}
        requested = list(cameras) if camera_ids is None else list(camera_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("camera_ids must be nonempty and unique")
        files = {}
        for camera_id in requested:
            _name(camera_id, "camera id")
            info = cameras[camera_id]
            path = PurePosixPath(info.get("rgb_path") or info["png_path"])
            if path.parent != remote or path.stem != camera_id or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"RGB path is not in this worker's observation: {path}")
            files[camera_id] = path.name
        dst = Path(directory) if directory is not None else self.out_dir / "frames" / obs_id
        archive = subprocess.run(
            self._ssh(shlex.join(["tar", "-cf", "-", "-C", str(remote), "--", *files.values()])),
            check=True, capture_output=True, timeout=30,
        )
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as bundle:
            for filename in files.values():
                member = bundle.getmember(filename)
                if not member.isfile():
                    raise ValueError(f"not a regular image file: {filename}")
                with bundle.extractfile(member) as f:
                    (dst / filename).write_bytes(f.read())
        return {camera_id: dst / filename for camera_id, filename in files.items()}
