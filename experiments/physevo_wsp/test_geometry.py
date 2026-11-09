import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from geometry import annotate, grasp_from_link6, marks, top_down_link6, world_on_plane


class GeometryTests(unittest.TestCase):
    def test_rgb_marks_follow_visible_positions(self):
        image = Image.new("RGB", (135, 110), (147, 92, 70))
        d = ImageDraw.Draw(image)
        d.ellipse((75, 60, 97, 82), fill=(75, 94, 219))
        d.ellipse((12, 49, 32, 69), fill=(44, 65, 212))
        d.ellipse((36, 29, 66, 56), fill=(232, 216, 186))
        # Cream-colored wall stripe must not be labeled as a circular pad.
        d.rectangle((115, 10, 123, 64), fill=(232, 216, 186))
        with TemporaryDirectory() as td:
            p = Path(td)
            image.save(p / "rgb.png")
            found = marks(p / "rgb.png")
            self.assertEqual([(m["id"], m["uv"]) for m in found["digits"]],
                             [("d0", [22, 59]), ("d1", [86, 71])])
            self.assertEqual([(m["id"], m["uv"]) for m in found["pads"]],
                             [("p0", [51, 42.5])])
            with Image.open(annotate(p / "rgb.png", found, p / "labels.png")) as labeled:
                self.assertEqual(labeled.size, image.size)

    def test_project_uses_pose_for_the_observation(self):
        # Optical +X = world +X, optical +Y = world -Y, optical +Z = world -Z.
        transform = [[1, 0, 0, 1], [0, -1, 0, 2], [0, 0, -1, 3], [0, 0, 0, 1]]
        ob = {
            "camera_calibration": {"cameras": {"cam_head": {
                "K": [[100, 0, 320], [0, 200, 240], [0, 0, 1]],
                "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }}},
            "camera_pose_sample": {"cameras": {"cam_head": {"T_world_camera": transform}}},
        }
        self.assertEqual(world_on_plane(ob, (320, 240), 1), [1, 2, 1])
        self.assertEqual(world_on_plane(ob, (370, 340), 1), [2, 1, 1])
        with self.assertRaises(ValueError):
            world_on_plane(ob, (320, 240), 4)

    def test_native_pose_round_trips_grasp_center(self):
        tip = [0.2, -0.05, 0.7725]
        for yaw in (0, math.pi / 2, -1.08, math.pi):
            geometry = {"link6_to_grasp_center_local_m": [0.1501, 0, 0]}
            xyz, q = top_down_link6(tip, yaw, geometry)
            self.assertAlmostEqual(sum(a * a for a in q), 1.0)
            self.assertAlmostEqual(xyz[0], tip[0])
            self.assertAlmostEqual(xyz[1], tip[1])
            self.assertAlmostEqual(xyz[2], tip[2] + 0.1501)
            for actual, expected in zip(grasp_from_link6(xyz, q, geometry), tip):
                self.assertAlmostEqual(actual, expected)
            # A local +Y probe points along the selected world jaw direction.
            probe = grasp_from_link6([0, 0, 0], q, {"link6_to_grasp_center_local_m": [0, 1, 0]})
            for actual, expected in zip(probe, (math.cos(yaw), math.sin(yaw), 0)):
                self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
