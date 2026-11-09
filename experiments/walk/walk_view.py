"""Render the simulator's measured side-view pose, not a planned gait."""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = "#f3f6f8"
WHITE = "#ffffff"
INK = "#1e2d36"
MUTED = "#536874"
GRID = "#e5ecf0"
TRACK = "#9daeb7"
TEAL = "#147f75"
BLUE = "#4165af"
ORANGE = "#bd6030"
RED = "#b4443c"
GREEN = "#28744c"
LEG_COLORS = (BLUE, ORANGE)


@lru_cache(maxsize=48)
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


def text(d, at, value, size, *, fill=INK, bold=False, anchor=None, chinese=False):
    d.text(at, value, font=font(size, bold, chinese), fill=fill, anchor=anchor)


def world(d, state, box, *, past=(), camera=None, grid=True, labels=True):
    """Draw recorded capsule endpoints in x/z, with a fixed metric scale."""
    left, top, right, bottom = box
    scale = min((right-left) / 2.6, (bottom-top) / 1.82)
    center = state["x"] if camera is None else camera
    x_min, x_max = center - (right-left)/(2*scale), center + (right-left)/(2*scale)
    ground = bottom - 20
    center_px = (left+right)/2

    def xy(point):
        x, z = point
        return (center_px+(x-center)*scale, ground-z*scale)

    if grid:
        d.rectangle((left, top, right, bottom), fill="#fbfdfe")
        for i in range(math.ceil(x_min * 2), math.floor(x_max * 2) + 1):
            xp, _ = xy((i/2, 0))
            d.line((xp, top, xp, ground), fill=GRID)
            if labels:
                text(d, (xp, ground+3), f"{i/2:g}", 10, fill=MUTED, anchor="mt")
        for h in (0.5, 1.0, 1.5):
            _, yp = xy((center, h))
            if yp >= top:
                d.line((left, yp, right, yp), fill=GRID)
    d.line((left, ground, right, ground), fill=TRACK, width=2)
    previous = [row for row in past if x_min <= row.get("x", math.inf) <= x_max]
    if len(previous) > 1:
        d.line([xy((r["x"], r["z"])) for r in previous], fill="#a7cfc9", width=2)
    pose = {segment["name"]: segment for segment in state["pose"]}
    # Draw the left (farther, in the side view) leg before the right one.
    for name in ("thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r", "torso"):
        segment = pose[name]
        color = ORANGE if name.endswith("_l") else BLUE if name.endswith("_r") else INK
        p0, p1 = xy(segment["start"]), xy(segment["end"])
        radius = max(2.0, segment["radius"]*scale)
        d.line((p0, p1), fill=color, width=round(2*radius))
        for x, y in (p0, p1):
            d.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    for i, (side, contact) in enumerate(zip(("r", "l"), state["contact"])):
        if not contact:
            continue
        foot = pose[f"foot_{side}"]
        x, _ = xy(tuple((a+b)/2 for a,b in zip(foot["start"], foot["end"])))
        d.ellipse((x-5, ground-5, x+5, ground+5), fill=LEG_COLORS[i], outline=WHITE, width=1)
    return xy


def status(snapshot):
    if snapshot["walk"]["fall_reason"]:
        return "Fallen", RED
    if snapshot.get("success"):
        return "Walking criterion met", GREEN
    return "Criterion not met", MUTED


def render_view(snapshot, path):
    """PNG that records the measured view and telemetry for the next decision."""
    image = Image.new("RGB", (800, 600), BG)
    d = ImageDraw.Draw(image)
    text(d, (27, 17), "PLANAR WALKER", 29, bold=True)
    text(d, (28, 56), "Measured side view  /  +x to the right  /  metres", 15, fill=MUTED)
    d.rounded_rectangle((22, 89, 528, 459), radius=14, fill=WHITE, outline="#d1dce2")
    d.rounded_rectangle((542, 89, 777, 459), radius=14, fill=WHITE, outline="#d1dce2")
    state = snapshot["state"]
    trace = snapshot.get("samples", [])
    world(d, state, (37, 106, 513, 431), past=trace)
    status_text, status_color = status(snapshot)
    text(d, (558, 108), "CURRENT STATE", 14, bold=True, fill=MUTED)
    rows = [
        ("Position", f'{state["x"]:+.3f} m'),
        ("Height", f'{state["z"]:.3f} m'),
        ("Pitch", f'{math.degrees(state["pitch"]):+.1f}°'),
        ("Forward speed", f'{state["vx"]:+.3f} m/s'),
    ]
    for j, (label, value) in enumerate(rows):
        y = 149+j*41
        text(d, (558, y), label, 12, fill=MUTED)
        text(d, (759, y), value, 15, bold=True, anchor="rt")
    d.line((558, 318, 759, 318), fill=GRID)
    text(d, (558, 331), status_text, 15, fill=status_color, bold=True)
    text(d, (558, 358), f'Foot contact R/L: {int(bool(state["contact"][0]))} / {int(bool(state["contact"][1]))}', 14)
    text(d, (558, 389), f'Alternating landings: {snapshot["walk"]["alternating_steps"]}', 14)
    text(d, (558, 415), f'Time in attempt: {state["t"]:.2f} s', 14, fill=MUTED)

    d.rounded_rectangle((22, 472, 777, 558), radius=12, fill=WHITE, outline="#d1dce2")
    text(d, (38, 488), "Recent contacts", 13, fill=MUTED, bold=True)
    ticks = snapshot.get("trace") or trace
    if len(ticks) > 1:
        n = ticks[-1]["tick"]
        for leg in range(2):
            y = 519 + leg*17
            d.rounded_rectangle((190, y-3, 752, y+3), radius=3, fill=GRID)
            for i, row in enumerate(ticks[1:], 1):
                if row["contact"][leg]:
                    x0 = 190 + ticks[i-1]["tick"]/n*562
                    x1 = 190 + row["tick"]/n*562
                    d.rectangle((x0, y-3, x1, y+3), fill=LEG_COLORS[leg])
        text(d, (752, 492), f'{n * snapshot["dt"]:.2f} s', 11, fill=MUTED, anchor="rt")
    else:
        text(d, (190, 524), "No motion segment", 12, fill=MUTED)
    for i, color in enumerate(LEG_COLORS):
        text(d, (38, 512+i*17), f"{('Right','Left')[i]} foot", 12, fill=color)
    meters = snapshot["meters"]
    text(d, (28, 574), f'Attempt {snapshot["attempt"]}  ·  Controller v{snapshot["policy_version"]}'
         f'  ·  {meters["moves_left"]} motions / {meters["ticks_left"]} ticks left', 13, fill=MUTED)
    image.save(Path(path))
