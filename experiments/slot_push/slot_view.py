"""Top-down rendering of exactly the telemetry observed by the slot-push agent."""
from __future__ import annotations

from functools import lru_cache
from math import atan2, ceil, degrees, floor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INK = "#182738"
MUTED = "#516174"
BACKGROUND = "#f3f7fb"
TIP = "#b54224"
BLOCK = "#4e4fa0"
GREEN = "#168065"
WALL = "#586c7f"


@lru_cache(maxsize=64)
def font(size, bold=False, chinese=False):
    if chinese:
        name = f"NotoSansCJK-{'Bold' if bold else 'Regular'}.ttc"
        try:
            return ImageFont.truetype(name, size, index=2)
        except OSError:
            pass
    try:
        return ImageFont.truetype(f"DejaVuSans{'-Bold' if bold else ''}.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def yaw_deg(state):
    c, s = state["block_angle"]
    return degrees(atan2(s, c))


def block_vertices(state, block_size):
    x, y = state["block"]
    c, s = state["block_angle"]
    a, b = (d / 2 for d in block_size)
    return [(x + c * u - s * v, y + s * u + c * v)
            for u, v in ((-a, -b), (a, -b), (a, b), (-a, b))]


def draw_world(draw, snapshot, box, *, labels=True):
    """Draw a metric, equal-scale top-down map inside a pixel rectangle."""
    scene, state = snapshot["scene"], snapshot["state"]
    xmin, xmax = scene["x"]
    ymin, ymax = scene["y"]
    left, top, right, bottom = box
    scale = min((right - left) / (xmax - xmin), (bottom - top) / (ymax - ymin))
    cx, cy = (left + right) / 2, (top + bottom) / 2
    mx, my = (xmin + xmax) / 2, (ymin + ymax) / 2

    def pixel(point):
        x, y = point
        return (cx + (x - mx) * scale, cy - (y - my) * scale)

    def rectangle(rect):
        x0, y0, x1, y1 = rect
        return (*pixel((x0, y1)), *pixel((x1, y0)))

    bounds = rectangle((xmin, ymin, xmax, ymax))
    draw.rectangle(bounds, fill="#fcfdff", outline="#abbacb", width=1)
    for cm in range(ceil(100 * xmin), floor(100 * xmax) + 1):
        x = cm / 100
        draw.line([pixel((x, ymin)), pixel((x, ymax))], fill="#e9eff6")
        if labels and cm % 2 == 0:
            xp, yp = pixel((x, ymin))
            draw.text((xp, yp + 8), str(cm), fill=MUTED, font=font(13), anchor="mt")
    for cm in range(ceil(100 * ymin), floor(100 * ymax) + 1):
        y = cm / 100
        draw.line([pixel((xmin, y)), pixel((xmax, y))], fill="#e9eff6")
        if labels and cm % 2 == 0:
            xp, yp = pixel((xmin, y))
            draw.text((xp - 8, yp), str(cm), fill=MUTED, font=font(13), anchor="rm")
    draw.rectangle(rectangle(scene["target"]), fill="#d7f2e7", outline=GREEN, width=2)
    for rail in scene["rails"]:
        draw.rectangle(rectangle(rail), fill=WALL, outline=INK, width=1)

    samples = snapshot.get("samples", [])
    if len(samples) > 1:
        draw.line([pixel(row["tip"]) for row in samples], fill="#f4a88b", width=3)
        draw.line([pixel(row["block"]) for row in samples], fill="#9fa7e1", width=3)
        prior = [pixel(point) for point in block_vertices(samples[0], scene["block_size"])]
        draw.line([*prior, prior[0]], fill="#9fa7e1", width=1)

    points = [pixel(point) for point in block_vertices(state, scene["block_size"])]
    draw.polygon(points, fill="#9e9ddd", outline=BLOCK, width=2)
    bx, by = state["block"]
    c, s = state["block_angle"]
    along = scene["block_size"][0] * 0.36
    draw.line([pixel((bx - c * along, by - s * along)), pixel((bx + c * along, by + s * along))],
              fill="#34376e", width=3)
    ax, ay = pixel((bx + c * along, by + s * along))
    draw.ellipse((ax - 3, ay - 3, ax + 3, ay + 3), fill="#34376e")

    tx, ty = pixel(state["tip"])
    radius = max(4, scale * scene["pusher_radius"])
    draw.ellipse((tx - radius, ty - radius, tx + radius, ty + radius),
                 fill="#f9c6a8", outline=TIP, width=2)
    draw.ellipse((tx - 2, ty - 2, tx + 2, ty + 2), fill=TIP)


def render_view(snapshot, path):
    """Save the model-facing observation; no waypoints or planner annotations."""
    image = Image.new("RGB", (760, 594), BACKGROUND)
    d = ImageDraw.Draw(image)
    d.text((27, 22), "SLOT PUSH  /  TOP VIEW", font=font(24, True), fill=INK)
    d.text((28, 57), "Measured position in cm  ·  +x right  ·  +y up", font=font(14), fill=MUTED)
    d.rounded_rectangle((22, 91, 486, 555), 12, fill="white", outline="#d2deea", width=1)
    draw_world(d, snapshot, (60, 113, 462, 515))
    d.rounded_rectangle((500, 91, 737, 555), 12, fill="white", outline="#d2deea", width=1)
    state = snapshot["state"]
    d.text((520, 112), "CURRENT STATE", font=font(14, True), fill=MUTED)
    for y, label, color, text in (
        (151, "Pusher", TIP, f"x {state['tip'][0] * 100:.2f} / y {state['tip'][1] * 100:.2f} cm"),
        (213, "Block", BLOCK, f"x {state['block'][0] * 100:.2f} / y {state['block'][1] * 100:.2f} cm"),
        (275, "Block yaw", BLOCK, f"{yaw_deg(state):+.1f}°"),
        (337, "Target", GREEN, "Block inside" if snapshot["success"] else "Block not inside"),
    ):
        d.ellipse((520, y + 3, 532, y + 15), fill=color)
        d.text((542, y), label, font=font(15, True), fill=INK)
        d.text((542, y + 24), text, font=font(13), fill=MUTED)
    d.line((520, 404, 716, 404), fill="#dbe5ed")
    summary = snapshot.get("contact_summary") or {}
    for i, (label, key) in enumerate((
        ("Pusher–block contact", "tip_block_ticks"), ("Block–rail contact", "block_wall_ticks"),
    )):
        value = summary.get(key, 0)
        d.text((520, 422 + 27 * i), label, font=font(12), fill=MUTED)
        d.text((714, 422 + 27 * i), f"{value} ticks", anchor="rt", font=font(12, True), fill=INK)
    if summary.get("stall"):
        d.text((520, 493), "Contact with little progress", font=font(13, True), fill=TIP)
    meters = snapshot["meters"]
    d.text((28, 567), f"Remaining: {meters['moves_left']} moves  ·  {meters['ticks_left']} ticks"
           f"       |       Controller v{snapshot['policy_version']}", font=font(13), fill=MUTED)
    image.save(Path(path))
