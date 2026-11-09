"""Fixed RGB observations and actuator geometry for the RoboDojo digit task.

No simulator object state is used. Table height and nominal thicknesses come
from the public scene configuration and object geometry metadata.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# RoboDojo env_cfg/scene/default.yml: Table center 0.740 and thickness 0.050.
TABLE_Z = 0.765
# Assets/Object/RoboDojo/Rigid/number: nominal thickness is 0.014–0.016 m.
DIGIT_TOP_Z = TABLE_Z + 0.015
DIGIT_GRASP_Z = TABLE_Z + 0.0075
# Assets/Object/RoboDojo/Geometry/cube_cushion/00001: thickness 0.005 m.
PAD_TOP_Z = TABLE_Z + 0.005
PAD_GRASP_Z = PAD_TOP_Z + 0.0075


def _components(points):
    """Return pixel islands under 8-neighbour connectivity."""
    while points:
        first = points.pop()
        stack, group = [first], [first]
        while stack:
            x, y = stack.pop()
            for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                p = (x + dx, y + dy)
                if p in points:
                    points.remove(p)
                    group.append(p)
                    stack.append(p)
        yield group


def _blobs(points, minimum, maximum, *, max_aspect=None):
    blobs = []
    for group in _components(points):
        count = len(group)
        if not minimum <= count <= maximum:
            continue
        xs, ys = zip(*group)
        box = [min(xs), min(ys), max(xs), max(ys)]
        width = box[2] - box[0] + 1
        height = box[3] - box[1] + 1
        if min(width, height) < 7 or max(width, height) > 65:
            continue
        if max_aspect is not None and max(width, height) > max_aspect * min(width, height):
            continue
        mean_x, mean_y = sum(xs) / count, sum(ys) / count
        dx, dy = [x - mean_x for x in xs], [y - mean_y for y in ys]
        angle = 0.5 * math.atan2(2 * sum(x * y for x, y in zip(dx, dy)),
                                 sum(x * x for x in dx) - sum(y * y for y in dy))
        # The asset origins lie near the centers of their bounds. The ink /
        # silhouette centroids of asymmetric shapes (notably '7') do not.
        uv = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        blobs.append({"uv": uv, "bbox": box, "pixels": count, "angle_rad": angle})
    return sorted(blobs, key=lambda b: (b["uv"][0], b["uv"][1]))


def marks(image_path):
    """Locate visible blue pieces and cream pads; return IDs ordered by pixel x.

    The classifier is fixed for the initial appearance in this layout, not an
    OCR tool or an identity tracker. Angle is the unsigned silhouette axis in
    image coordinates (+u right, +v down), not the upright direction of a digit.
    """
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    digits, pads = set(), set()
    width, height = image.size
    pixel = image.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixel[x, y]
            if b > 95 and b - r > 48 and b - g > 36:
                digits.add((x, y))
            if r > 187 and g > 175 and b > 122 and 0 <= r - g <= 32 and 12 <= g - b <= 58:
                pads.add((x, y))
    return {
        key: [{"id": f"{prefix}{i}", **b} for i, b in enumerate(blobs)]
        for key, prefix, blobs in (
            ("digits", "d", _blobs(digits, 55, 1400)),
            ("pads", "p", _blobs(pads, 250, 1800, max_aspect=1.65)),
        )
    }


def annotate(image_path, detections, output):
    """Draw pixel-detection boxes and IDs, preserving the original pixel grid."""
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    for group, color in (("digits", "#ffb57e"), ("pads", "#8df3e3")):
        for mark in detections[group]:
            x, y, _, _ = mark["bbox"]
            draw.rectangle(mark["bbox"], outline=color, width=2)
            draw.text((x, y - 20), mark["id"], fill=color, stroke_width=1,
                      stroke_fill="#101b24", font=font)
    path = Path(output)
    image.save(path)
    return path


def world_on_plane(observation, uv, z, camera_id="cam_head"):
    """Intersect a calibrated pixel ray with a world-height plane in env_origin."""
    camera = observation["camera_calibration"]["cameras"][camera_id]
    poses = observation.get("camera_pose_sample", {}).get("cameras", {})
    transform = poses[camera_id]["T_world_camera"] if camera_id in poses else camera["T_world_camera"]
    k = camera["K"]
    u, v = uv
    y = (v - k[1][2]) / k[1][1]
    x = (u - k[0][2] - k[0][1] * y) / k[0][0]
    ray = [row[0] * x + row[1] * y + row[2] for row in transform[:3]]
    if abs(ray[2]) < 1e-9:
        raise ValueError("Pixel ray is parallel to the projection plane")
    t = (z - transform[2][3]) / ray[2]
    if not math.isfinite(t) or t <= 0:
        raise ValueError("Projection plane is behind the camera")
    return [transform[0][3] + t * ray[0], transform[1][3] + t * ray[1], float(z)]


def _rotation(q):
    x, y, z, w = q
    return [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ]


def _world_offset(quaternion_xyzw, tool_geometry):
    local_offset = tool_geometry["link6_to_grasp_center_local_m"]
    return [sum(a * b for a, b in zip(row, local_offset, strict=True))
            for row in _rotation(quaternion_xyzw)]


def top_down_link6(grasp_xyz, yaw, tool_geometry):
    """Encode the requested grasp center and jaw yaw as a native link6 pose.

    ARX-X5 local +X points toward the grasp, so this rotation maps it to world
    −Z; local +Y (jaw opening) maps to (cos(yaw), sin(yaw), 0).
    """
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    q = [(c - s) / 2, (c + s) / 2, (s - c) / 2, (c + s) / 2]
    offset = _world_offset(q, tool_geometry)
    return [a - b for a, b in zip(grasp_xyz, offset, strict=True)], q


def grasp_from_link6(link6_xyz, quaternion_xyzw, tool_geometry):
    """Recover the measured grasp center from a native link6 pose."""
    offset = _world_offset(quaternion_xyzw, tool_geometry)
    return [a + b for a, b in zip(link6_xyz, offset, strict=True)]
