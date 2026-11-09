"""Exercise the RoboDojo bridge protocol without SSH, Isaac Sim or a GPU."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from native import Native, WorkerError, ee_action, joint_action


WORKER = r"""
import json
import os
from pathlib import Path
import sys
import time

args = sys.argv
run = Path(args[args.index("--run-dir") + 1])
seed = int(args[args.index("--seed") + 1])
layout = int(args[args.index("--layout-id") + 1])
gpu = args[args.index("--gpu") + 1]
task = args[args.index("--task") + 1]
assert Path(os.environ["TMPDIR"]) == run / "tmp"
assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu
assert Path(os.environ["PATH"].split(":")[0]) == Path(args[0]).parent
assert all(os.environ[key] == "http://127.0.0.1:9" for key in (
    "http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))
assert os.environ["no_proxy"] == os.environ["NO_PROXY"] == "127.0.0.1,localhost"
assert args[args.index("--observation-profile") + 1] == "rgb_proprio_calibrated_v1"
assert task == "arrange_largest_number"

def emit(value):
    print("AURIGA_JSON " + json.dumps(value), flush=True)

q = [0.] * 6 + [.2] + [.1] * 6 + [.4]
clock = 0
recording = False
for line in sys.stdin:
    req = json.loads(line)
    op, payload, i = req["op"], req["payload"], req["id"]
    if op == "ready":
        result = {"frame_id":"env_origin", "tcp_config_id":"arx-x5-link6-v1",
                  "limits_profile":"robodojo-native-v1", "seed":seed, "layout":layout,
                  "motion_preview":{"available":True},
                  "arm_joint_limits_rad":{s:[[-3.,3.]]*6 for s in ("left","right")},
                  "arm_max_step_rad":{s:[.05]*6 for s in ("left","right")}, "gripper_max_step":.25}
    elif op == "observe":
        folder = run / "observations" / "raw-rgb"
        folder.mkdir(parents=True, exist_ok=True)
        camera = folder / "cam_head.png"
        camera.write_bytes(b"actual observation bytes")
        arms = {side:{"q_rad":q[start:start+6], "gripper":{"goal_normalized":q[start+6]}}
                for side,start in (("left",0),("right",7))}
        result = {"raw_observation_id":"raw-rgb", "cameras":[{"camera_id":"cam_head","png_path":str(camera)}],
                  "robot_state":{"arms":arms}, "sim_step":clock, "object_ground_truth":{"never":"public"}}
    elif op == "execute":
        emit({"id":i, "progress":{"episode_terminal":{"score":100}}})
        if payload["type"] == "JointPrefixAction":
            rows = payload["actions"]
            assert 1 <= len(rows) <= 50 and payload["policy_provenance"]
            for row in rows:
                assert len(row) == 14
                assert max(abs(a-b) for a,b in zip(row[:6], q[:6])) <= .050000001
                assert row[7:] == [.1]*6 + [.4]
                assert abs(row[6] - q[6]) <= .250000001
                q = row
            prev = clock
            clock += len(rows)*10
            result = {"status":"completed", "executed_native_steps":len(rows), "physics_advanced":True,
                      "execution_interval":{"start_sim_step":prev,"end_sim_step":clock,"step_domain":"adapter_control_step"}}
        else:
            assert payload["type"] == "EEAction"
            if payload["path"]["kind"] == "hold":
                assert not payload["path"]["waypoints"] and payload["gripper_events"]
            else:
                assert len(payload["path"]["waypoints"]) == 1
            result = {"status":"completed", "executed_native_steps":3, "episode_terminal":{"score":100}}
    elif op == "preview_motion":
        assert payload["type"] == "EEAction" and len(payload["path"]["waypoints"]) == 1
        rows = [[.025] * 6, [.05] * 6]
        result = {"status":"planned", "reason":None, "physics_advanced":False, "planned_native_steps":2,
                  "planned_trace":{"joint_names":[f"q{i}" for i in range(6)],"q_rad":rows},
                  "native_rows":[r+[.2]+q[7:] for r in rows],"privileged_debug":"omit"}
    elif op == "record":
        assert payload == {"enabled":True}
        recording = True
        result = {"enabled":True}
    elif op == "save_video":
        assert recording
        folder = run / "videos" / "one"
        folder.mkdir(parents=True, exist_ok=True)
        mp4 = folder / "cam_head.mp4"
        mp4.write_bytes(b"native video bytes")
        result = {"videos":[{"camera_id":"cam_head", "video_path":str(mp4),
                            "frame_mapping":[{"video_frame":0,"sim_step":0}]}]}
    elif op == "evaluate":
        assert payload == {"final_check":True}
        result = {"normalized_score":.5}
    elif op == "close":
        emit({"id":i, "progress":{"phase":"environment_closed"}})
        emit({"id":i, "progress":{"phase":"application_close_started","fast_shutdown":False}})
        time.sleep(0.08)
        emit({"id":i, "progress":{"phase":"application_closed"}})
        emit({"id":i, "result":{"closed":True}})
        break
    else:
        emit({"id":i, "error":"unsupported operation"})
        continue
    emit({"id":i,"result":result})
"""


class LocalNative(Native):
    def _ssh(self, command):
        return ["bash", "-c", command]


class NativeTest(unittest.TestCase):
    def test_moves_and_grips_are_generic_single_commands(self):
        ready = {"frame_id": "env_origin", "tcp_config_id": "arx-x5-link6-v1",
                 "limits_profile": "robodojo-native-v1"}
        move = ee_action(ready, "right", [0.2, 0.3, 0.7], [0, 0, 0, 1], 0)
        self.assertEqual(move["path"]["waypoints"], [
            {"right": {"position_m": [0.2, 0.3, 0.7], "quaternion_xyzw": [0, 0, 0, 1]}}
        ])
        self.assertEqual(move["gripper_events"], [{"arm": "right", "phase": "after_motion", "goal_normalized": 0}])
        grip = ee_action(ready, "left", grip=1)
        self.assertEqual(grip["path"]["kind"], "hold")
        self.assertEqual(grip["path"]["waypoints"], [])
        self.assertEqual(grip["gripper_events"][0]["arm"], "left")
        with self.assertRaises(ValueError):
            ee_action(ready, "right", [0, 0, 0], [1, 0, 0, 1])
        with self.assertRaises(ValueError):
            ee_action(ready, "left", grip=2)
        with self.assertRaises(ValueError):
            joint_action([[0.] * 13])
        with self.assertRaises(ValueError):
            joint_action([[0.] * 13 + [1.1]])

    def test_native_transport_rgb_and_fresh_worker_reset(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base with spaces"
            (base / "src" / "RoboDojo").mkdir(parents=True)
            py = base / "miniforge3" / "envs" / "RoboDojo" / "bin" / "python"
            py.parent.mkdir(parents=True)
            py.write_text(f"#!{sys.executable}\n{WORKER}", encoding="utf-8")
            py.chmod(0o755)
            parent = root / "isolated runs"
            with LocalNative(root / "local", gpu=5, seed=7, layout_id=2, base=base,
                             remote_parent=parent, startup_timeout=5, rpc_timeout=5, close_timeout=2) as native:
                self.assertEqual((native.ready["seed"], native.ready["layout"]), (7, 2))
                self.assertEqual(Path(native.remote_dir).parent, parent)
                self.assertEqual(Path(native.remote_dir).stat().st_mode & 0o777, 0o700)
                obs = native.observe()
                self.assertNotIn("object_ground_truth", obs)
                camera, = native.fetch_images(obs, camera_ids=("cam_head",)).values()
                self.assertEqual(camera.read_bytes(), b"actual observation bytes")
                outside = json.loads(json.dumps(obs))
                outside["cameras"][0]["png_path"] = "/other/observations/raw-rgb/cam_head.png"
                with self.assertRaises(ValueError):
                    native.fetch_images(outside)
                self.assertEqual(native.move("right", [0.2, 0.3, 0.7], [0, 0, 0, 1]),
                                 {"status": "completed", "executed_native_steps": 3})
                self.assertEqual(native.grip("right", 0)["status"], "completed")
                preview = native.pose_preview("left", [.2,.3,.7], [0,0,0,1])
                self.assertEqual(preview["q_rad"], [.05]*6)
                self.assertEqual(len(preview["native_rows"]), 2)
                self.assertFalse(preview["physics_advanced"])
                self.assertNotIn("privileged_debug", preview)
                self.assertEqual(native.observe()["sim_step"], 0)
                result = native.move_joints("left", [2.6,0,0,0,0,0], .8)
                self.assertEqual(result["executed_native_steps"], 52)
                self.assertEqual(result["execution_interval"],
                                 {"start_sim_step":0,"end_sim_step":520,"step_domain":"adapter_control_step"})
                arms = native.observe()["robot_state"]["arms"]
                self.assertEqual(arms["left"], {"q_rad":[2.6,0,0,0,0,0],"gripper":{"goal_normalized":.8}})
                self.assertEqual(arms["right"], {"q_rad":[.1]*6,"gripper":{"goal_normalized":.4}})
                with self.assertRaises(ValueError):
                    native.move_joints("left", [3.1,0,0,0,0,0])
                self.assertTrue(native.begin_recording()["enabled"])
                media = native.fetch_recording()
                self.assertEqual(media["cam_head"].read_bytes(), b"native video bytes")
                manifest = json.loads((media["cam_head"].parent / "video-manifest.json").read_text())
                self.assertEqual(manifest["videos"][0]["frame_mapping"][0]["sim_step"], 0)
                with self.assertRaises(WorkerError):
                    native.call("unsupported")
                self.assertEqual(native.call("evaluate", {"final_check": True}), {"normalized_score": .5})
                first = native.remote_dir
                self.assertEqual(native.reset()["sim_step"], 0)
                self.assertNotEqual(native.remote_dir, first)
                self.assertEqual((native.ready["seed"], native.ready["layout"]), (7, 2))
                # Native shutdown may be longer than the normal RPC deadline.
                native.rpc_timeout = 0.01
            self.assertEqual(len(list((root / "local").glob("worker-*.stderr.log"))), 2)
            self.assertTrue(native.close_report["native_close_acknowledged"])
            self.assertEqual(native.close_report["ssh_returncode"], 0)
            self.assertEqual([p["phase"] for p in native.close_report["progress"]],
                             ["environment_closed", "application_close_started", "application_closed"])


if __name__ == "__main__":
    unittest.main()
