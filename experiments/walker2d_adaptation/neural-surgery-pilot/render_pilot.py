"""Render synchronized measured MuJoCo poses before and after a weight edit."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw

WALK_SRC = Path.home() / "文档/ChatGPT/embodied-weight-growth/experiments/walk"
W, H, FPS = 1080, 482, 30
SPEED = 0.5
HOLD = 1.25
FALL = {"torso touched the floor": "躯干触地", "torso height dropped below 0.8 m": "躯干低于 0.8 m",
        "torso tilted beyond 1.0 rad": "躯干倾角超过 1.0 rad"}


def load_traces(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    sides = [data[k] for k in ("before", "after")]
    timelines = []
    for phase, side in zip(("before", "after"), sides):
        states = side["trace"]
        if not states:
            raise ValueError(f"No recorded motion trace for {phase}")
        if any(not row.get("pose") for row in states):
            raise ValueError(f"Missing measured MuJoCo pose for {phase}")
        times = [row["t"] - states[0]["t"] for row in states]
        if any(b <= a for a,b in zip(times, times[1:])):
            raise ValueError(f"Non-increasing MuJoCo trace time for {phase}")
        timelines.append(times)
    return sides, timelines


def render(source, output, still, *, walk_src=WALK_SRC, seed=271):
    sys.path.insert(0, str(walk_src))
    from walk_view import BG, BLUE, GREEN, GRID, INK, MUTED, ORANGE, RED, WHITE, text, world

    sides, times = load_traces(source)
    last = max(t[-1] for t in times)
    pitch = sides[0].get("scene", {}).get("initial_pitch_rad", sides[0]["trace"][0]["pitch"])
    all_x = [r["x"] for side in sides for r in side["trace"]]
    lo, hi = min(all_x), max(all_x)
    fixed_camera = (lo+hi)/2 if hi-lo <= 2.25 else None
    camera_label = "同一坐标视角" if fixed_camera is not None else "分别跟踪，米制比例相同"

    def draw(t):
        image = Image.new("RGB", (W,H), BG)
        d = ImageDraw.Draw(image)
        def txt(at, s, size, **kwargs):
            text(d, at, s, size, chinese=True, **kwargs)
        txt((23, 13), "行走控制器 · 权重改动前后", 28, bold=True)
        txt((24, 52), f"MuJoCo 实测侧视   ·   seed {seed}   ·   初始俯仰 {pitch:g} rad   ·   半速回放", 15, fill=MUTED)
        for x, color, label in ((832, BLUE, "右腿"), (931, ORANGE, "左腿")):
            d.rounded_rectangle((x, 55, x+21, 64), radius=4, fill=color)
            txt((x+29, 50), label, 15, fill=MUTED)
        for i, (name, side, timeline) in enumerate(zip(("改动前", "改动后"), sides, times)):
            x = 22 + i*524
            trace = side["trace"]
            j = min(len(timeline)-1, bisect_right(timeline, t+1e-9)-1)
            state = trace[j]
            stopped = j == len(timeline)-1
            fall = state.get("fall_reason") or (side.get("walk", {}).get("fall_reason") if stopped else None)
            d.rounded_rectangle((x, 85, x+510, 433), radius=11, fill=WHITE, outline="#d6e1e6")
            txt((x+15, 94), name, 21, bold=True)
            verdict, color = "实测轨迹", MUTED
            if stopped:
                if fall:
                    verdict, color = "摔倒 · 终态停留", RED
                elif side.get("success"):
                    verdict, color = "达到判据 · 终态停留", GREEN
                else:
                    verdict, color = "片段结束 · 终态停留", MUTED
            txt((x+492, 100), verdict, 15, fill=color, bold=stopped, anchor="rt")
            stride = max(1, len(trace)//145)
            world(d, state, (x+15, 131, x+494, 357), past=trace[:j+1:stride], camera=fixed_camera)
            dx = state["x"]-trace[0]["x"]
            count = state.get("alternating_steps")
            telemetry = f'{timeline[j]:.2f} s    位移 {dx:+.2f} m' + (f'    交替落脚 {count}' if count is not None else '')
            txt((x+16, 373), telemetry, 16, bold=True)
            if fall:
                txt((x+16, 402), FALL.get(fall, fall), 13, fill=RED)
            elif stopped and side.get("error"):
                txt((x+16, 402), "执行提前停止", 13, fill=RED)
        progress = min(1, t/last) if last > 0 else 1
        d.rounded_rectangle((24, 447, 1054, 451), radius=2, fill=GRID)
        d.rounded_rectangle((24, 447, 24+1030*progress, 451), radius=2, fill="#4c827a")
        txt((25, 458), f"同步仿真时钟 {min(t,last):.2f} s  /  {last:.2f} s", 12, fill=MUTED)
        txt((1053, 458), f"{camera_label}   ·   +x 向右", 12, fill=MUTED, anchor="rt")
        return image

    output.parent.mkdir(parents=True, exist_ok=True)
    end_at = math.ceil(last * FPS / SPEED)
    pending = output.with_name(f".{output.stem}.part.mp4")
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{W}x{H}", "-framerate", str(FPS),
           "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(pending)]
    try:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as p:
            with p.stdin:
                for frame in range(end_at):
                    p.stdin.write(draw(frame * SPEED / FPS).tobytes())
                terminal = draw(last)
                pixels = terminal.tobytes()
                for _ in range(round(HOLD * FPS)):
                    p.stdin.write(pixels)
            if p.wait():
                raise RuntimeError("FFmpeg could not encode the pilot replay")
        pending.replace(output)
        if still:
            terminal.save(still)
    finally:
        pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--still", type=Path)
    parser.add_argument("--walk-src", type=Path, default=WALK_SRC)
    parser.add_argument("--seed", type=int, default=271)
    args = parser.parse_args()
    render(args.trace, args.output, args.still, walk_src=args.walk_src, seed=args.seed)
    print(args.output)


if __name__ == "__main__":
    main()
