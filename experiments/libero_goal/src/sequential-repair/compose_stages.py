"""Compose four matched SmolVLA rollout directories into a synchronized MP4."""

import argparse
import json
from fractions import Fraction
from pathlib import Path
import subprocess
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont


FPS, PANEL, HEADER, INFO, FOOTER = 20, 384, 77, 100, 53
SIZE = (2 * PANEL, HEADER + 2 * (PANEL + INFO) + FOOTER)
KEYS = ("stock", "bottle", "bowl", "joint")
LABELS = ("原版 SmolVLA", "第 1 轮 · 修酒瓶", "第 2 轮 · 接着修碗", "第 3 轮 · 联合修订")
SUBTITLE = "同一起姿、同一指令；各轮权重从开局固定"
ACCENTS = ("#b1cbe0", "#f4c092", "#9edbc9", "#d1baff")
OBJECTS = {"bottle": "酒瓶", "bowl": "碗"}
BG, FG, MUTED, RULE = "#101a25", "#f0f4fb", "#aabbd0", "#34485e"
GOOD, WARN, BAD = "#8cddb5", "#ffd092", "#ffa497"


def load(directory):
    path = directory / "rollout.mp4"
    result = json.loads((directory / "result.json").read_text())
    trace = json.loads((directory / "trace.json").read_text())
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", str(path),
    ])
    stream = json.loads(probe)["streams"][0]
    steps = result["steps"]
    if steps != len(trace) or [row["step"] for row in trace] != list(range(1, steps + 1)):
        raise ValueError(f"Non-contiguous trace: {directory}")
    if int(stream["nb_frames"]) != steps + 1 or Fraction(stream["r_frame_rate"]) != FPS:
        raise ValueError(f"Expected one initial frame and one frame per 20 Hz step: {path}")
    if stream["width"] != stream["height"]:
        raise ValueError(f"Expected a square camera image: {path}")
    with ZipFile(directory / "initial-public.npz") as state:
        reset = {entry: state.read(entry) for entry in state.namelist()}
    return path, result, trace, stream["width"], reset


def decode(path, steps):
    return subprocess.Popen([
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
        "-i", str(path), "-frames:v", str(steps + 1), "-threads", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], stdout=subprocess.PIPE)


def placement(result, row, step):
    if step == 0:
        return "同一起姿", MUTED
    if row["native"]:
        stable = result.get("stable_released_at")
        if stable is not None and step >= stable:
            return "曾连续 2 秒稳定放置", GOOD
        return "当前在柜顶，仍夹持" if result["goal"] in row["held"] else "当前在柜顶，无双指夹持", GOOD
    first = result.get("first_native_goal")
    if first is not None and step >= first:
        return f"第 {first} 步曾达标，当前未达标", WARN
    if step == result["steps"]:
        return "此局未达标", BAD
    return "尚未到达柜顶", MUTED


def main(argv=None, *, keys=KEYS, labels=LABELS, subtitle=SUBTITLE):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in keys:
        parser.add_argument(f"--{key}", required=True, type=Path, help="directory of one scored rollout")
    parser.add_argument("--out", required=True, type=Path, help="MP4 output; writes a PNG poster beside it")
    parser.add_argument("--title", help="short heading; defaults to task name")
    parser.add_argument("--end", type=int, help="last visible control step; defaults to the longest rollout")
    args = parser.parse_args(argv)
    records = [load(getattr(args, key.replace("-", "_"))) for key in keys]
    result0 = records[0][1]
    matched = ("id", "seed", "init_index", "layout", "goal", "instruction", "first_prompt", "prompt_mode")
    if any(any(r.get(key) != result0.get(key) for key in matched) or reset != records[0][4]
           for _, r, _, _, reset in records[1:]):
        raise ValueError("The four rollouts do not share the same initial state or instruction schedule")
    goal = result0["goal"]
    title = args.title or f"把{OBJECTS[goal]}放上柜顶"
    end = max(r["steps"] for _, r, *_ in records)
    if args.end is not None:
        if not 1 <= args.end <= end:
            parser.error(f"--end must be between 1 and {end}")
        end = args.end
    # Show placement or the failure at the cabinet in the companion still, not an arbitrary end frame.
    poster_step = min(end, max(
        r.get("stable_released_at") or r.get("first_native_goal") or r.get("first_cabinet_contact_after_lift") or 0
        for _, r, *_ in records
    ) or end)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fonts = {size: ImageFont.truetype(font, size, index=2) for size in (14, 15, 16, 18, 22, 26)}

    def label(draw, xy, value, size, color=FG, **kw):
        draw.text(xy, value, font=fonts[size], fill=color, **kw)

    enc = subprocess.Popen([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out),
    ], stdin=subprocess.PIPE)
    decoders = []
    try:
        decoders = [decode(path, min(r["steps"], end)) for path, r, *_ in records]
        frames = [None] * len(records)
        for step in range(end + 1):
            canvas = Image.new("RGB", SIZE, BG)
            draw = ImageDraw.Draw(canvas)
            label(draw, (16, 8), title, 26)
            label(draw, (16, 46), subtitle, 15, MUTED)
            label(draw, (SIZE[0] - 16, 20), f"{step / FPS:05.2f} 秒  /  第 {step} 步", 18, anchor="rt")
            for i, (_, result, trace, native, _) in enumerate(records):
                last = result["steps"]
                s = min(last, step)
                if s == step:
                    pixels = decoders[i].stdout.read(native * native * 3)
                    if len(pixels) != native * native * 3:
                        raise ValueError(f"Short video read: {keys[i]} at step {step}")
                    frames[i] = Image.frombytes("RGB", (native, native), pixels).resize(
                        (PANEL, PANEL), Image.Resampling.LANCZOS
                    )
                x, y = i % 2 * PANEL, HEADER + i // 2 * (PANEL + INFO)
                canvas.paste(frames[i], (x, y))
                if s < step:
                    draw.rounded_rectangle((x + 12, y + 12, x + 178, y + 38), radius=5, fill=BG)
                    label(draw, (x + 20, y + 13), f"第 {last} 步结束 · 定格", 14)
                row = trace[s - 1] if s else None
                base = y + PANEL
                draw.rectangle((x + 16, base + 12, x + 19, base + 33), fill=ACCENTS[i])
                label(draw, (x + 28, base + 7), labels[i], 22, ACCENTS[i])
                lift = result.get("first_lift_step")
                if lift is not None and s >= lift:
                    obj = result["first_lift"]
                    lift_info = f"首抬：{OBJECTS.get(obj, obj)} · 第 {lift} 步"
                    lift_color = FG if obj == goal else BAD
                else:
                    lift_info = "无持续双指抬起记录" if s == last else "尚无持续双指抬起记录"
                    lift_color = MUTED
                label(draw, (x + 28, base + 43), lift_info, 15, lift_color)
                if row and goal in row["held"]:
                    label(draw, (x + PANEL - 17, base + 43), "双指夹持中", 14, MUTED, anchor="rt")
                status, tone = placement(result, row, s)
                label(draw, (x + 28, base + 68), status, 16, tone)
            draw.line((PANEL, HEADER, PANEL, SIZE[1] - FOOTER), fill=RULE, width=2)
            draw.line((0, HEADER + PANEL + INFO - 1, SIZE[0], HEADER + PANEL + INFO - 1), fill=RULE)
            draw.line((16, SIZE[1] - FOOTER, SIZE[0] - 16, SIZE[1] - FOOTER), fill=RULE)
            label(draw, (16, SIZE[1] - 39), "稳定放置：目标连续 40 步在柜顶，且未被双指夹持。", 15, MUTED)
            if step == poster_step:
                canvas.save(args.out.with_suffix(".png"))
            enc.stdin.write(canvas.tobytes())
        enc.stdin.close()
        if any(dec.wait() != 0 for dec in decoders) or enc.wait() != 0:
            raise RuntimeError("ffmpeg did not finish successfully")
    finally:
        for process in [*decoders, enc]:
            if process.poll() is None:
                process.terminate()
                process.wait()
    print(args.out)


if __name__ == "__main__":
    main()
