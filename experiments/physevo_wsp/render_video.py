"""Recover a sampled camera replay from a retained Auriga observation archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from report import OP, REASON, attempts


WIDTH, HEIGHT, FPS = 1200, 760, 25
LABEL = {"placed": "看图：已放上目标垫", "not_placed": "看图：没有放上目标垫",
         "uncertain": "看图：未能确认"}


def camera_schedule(run, source):
    """Rebuild capture steps from RPC boundaries and validate each named outcome anchor."""
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    groups = attempts(events)
    orders = json.loads((source / "source-order.json").read_text())
    if orders != sorted(orders) or len({t for t, _ in orders}) != len(orders):
        raise ValueError("source order must contain distinct, increasing file timestamps")
    captures = sorted(run.glob("transfer-*-try-*/observation.json"))
    if len(groups) != len(captures) or any(not g[-1] for g in groups):
        raise ValueError("each attempt must have a named outcome observation")
    schedule = []

    def add(step, kind, i=-1, row=None, anchor=None):
        if len(schedule) >= len(orders):
            raise ValueError("source frames end before the expected capture schedule")
        t, raw_id = orders[len(schedule)]
        if anchor:
            observed = json.loads(anchor.read_text())
            if (raw_id, step) != (observed["raw_observation_id"], observed["sim_step"]):
                raise ValueError(f"capture schedule diverges from {anchor}")
        if schedule and step < schedule[-1]["sim_step"]:
            raise ValueError("capture schedule runs backwards")
        schedule.append(dict(raw_observation_id=raw_id, mtime_ns=t, sim_step=step,
                             kind=kind, attempt_index=i, row=row))

    step = 0
    add(step, "initial", anchor=run / "initial/observation.json")
    for i, (_, _, rows, _) in enumerate(groups):
        for row in rows:
            receipt = row.get("receipt")
            if not receipt:
                continue
            interval = receipt["execution_interval"]
            count = receipt["executed_steps"]
            start, end = interval["start_sim_step"], interval["end_sim_step"]
            if start != step or end - start != count * 10:
                raise ValueError("unexpected native-to-adapter step mapping")
            add(start, "before_action", i, row)
            if count:
                add(start + 10, "executing", i, row)
                for n in range(5, count + 1, 5):
                    add(start + n * 10, "executing", i, row)
            add(end, "action_receipt", i, row)
            add(end, "actor_observe", i, row)
            step = end
        add(step, "outcome", i, anchor=captures[i])
    if len(schedule) != len(orders):
        raise ValueError("unmatched camera frames remain")
    return schedule, groups


def face(size, style="Regular"):
    path = Path("/usr/share/fonts/opentype/noto") / f"NotoSansCJK-{style}.ttc"
    return ImageFont.truetype(str(path), size, index=2)


def render(run, source):
    schedule, groups = camera_schedule(run, source)
    score = json.loads((run / "native-score.json").read_text())
    plan = json.loads((run / "semantic_plan.json").read_text())
    goal = "".join(str(t["digit"]) for t in sorted(plan["transfers"], key=lambda t: int(t["pad"][1:])))
    sim_dt = json.loads((run / "initial/observation.json").read_text())["control_dt_s"]
    if abs(sim_dt * 10 * FPS - 1) > 1e-6:
        raise ValueError("this replay clock expects one native step per video tick")
    full = schedule[-1]["sim_step"]
    normalized = score["normalized_score"] * 100
    report = run / "report"
    report.mkdir(exist_ok=True)
    path = report / "replay.mp4"
    pending = report / "replay.part.mp4"
    fonts = {size: face(size) for size in (17, 18, 20, 22, 25, 30)}
    bold = {size: face(size, "Bold") for size in (22, 30, 40)}
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
           "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(pending)]
    clock = 0
    mapping = []
    try:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            for i, entry in enumerate(schedule):
                step, kind, raw_id = entry["sim_step"], entry["kind"], entry["raw_observation_id"]
                following = schedule[i + 1]["sim_step"] if i + 1 < len(schedule) else step
                # One native step is 40 ms. Preserve that time and add labeled
                # holds for outcomes; same-step duplicate captures need no hold.
                ticks = (following - step) // 10
                if kind == "outcome":
                    ticks += FPS if i + 1 < len(schedule) else 2 * FPS
                elif kind == "initial":
                    ticks += FPS
                row = entry["row"]
                index = entry["attempt_index"]
                trans, version, _, outcome = groups[index] if index >= 0 else ({}, 0, [], {})
                mapping.append({k: v for k, v in entry.items() if k != "row"} |
                               {"video_frame": clock, "display_frames": ticks,
                                "operation": row["operation"] if row else None})
                if not ticks:
                    continue
                canvas = Image.new("RGB", (WIDTH, HEIGHT), "#101c20")
                d = ImageDraw.Draw(canvas)
                def label(x, y, value, size=20, color="#d6e6df", strong=False):
                    d.text((x, y), value, font=(bold if strong else fonts)[size], fill=color)

                label(24, 13, "RoboDojo  /  排列最大数字", 30, "#f0f8f4", True)
                label(910, 21, "首轮原始相机 · 采样回放", 18, "#abc7ba")
                with Image.open(source / "observations" / raw_id / "cam_head.png") as camera:
                    camera = camera.convert("RGB").resize((864, 648), Image.Resampling.LANCZOS)
                    canvas.paste(camera, (24, 64))

                label(914, 78, f"目标  {goal}", 30, "#f0f8f4", True)
                d.line((914, 137, 1176, 137), fill="#38524b", width=1)
                if kind == "initial":
                    label(914, 164, "开场观测", 30, "#f0f8f4", True)
                    label(914, 221, "四个数字仍散在桌上", 20)
                else:
                    label(914, 159, f"数字 {trans['digit']}", 40, "#f0f8f4", True)
                    arm = "左臂" if trans["arm"] == "left" else "右臂"
                    label(914, 224, f"第 {outcome['try']} 次 · {arm} · 权重 v{version}", 20)
                    if kind == "outcome":
                        result = outcome["status"]
                        label(914, 295, LABEL[result], 22, "#b6edce" if result == "placed" else "#ffd59c", True)
                        failure = next((r["receipt"].get("reason") for r in reversed(groups[index][2])
                                        if r.get("receipt", {}).get("status") == "failed"), None)
                        if failure:
                            label(914, 341, REASON.get(failure, failure), 20)
                        elif result == "not_placed":
                            label(914, 341, "动作走完，目标垫仍是空的", 18)
                    elif row:
                        label(914, 295, OP[row["operation"]], 22, "#b6edce", True)
                        label(914, 336, f"模块阶段 {row['stage']}", 18)

                if i + 1 == len(schedule):
                    label(914, 487, f"原生评分  {normalized:g}/100", 25, "#ffd59c")
                    label(914, 527, "任务未通过" if not score["task_success"] else "任务通过", 20)
                label(914, 636, f"仿真时间  {step * sim_dt:05.2f} 秒", 20)
                label(914, 673, f"原生动作  {step // 10} / {full // 10}", 18, "#abc7ba")
                d.rectangle((24, 732, 1176, 739), fill="#38524b")
                d.rectangle((24, 732, 24 + int(1152 * step / full), 739), fill="#9edbbb")
                frame = canvas.tobytes()
                for _ in range(ticks):
                    encoder.stdin.write(frame)
                clock += ticks
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("ffmpeg replay encoding failed")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    (report / "replay-timeline.json").write_text(json.dumps({
        "source": "original head RGB captures with annotations; no motion interpolation",
        "remote_run": json.loads((run / "protocol.json").read_text())["remote_run"],
        "fps": FPS, "duration_s": clock / FPS, "frames": mapping,
    }, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path, help="directory with observations and source-order.json")
    args = parser.parse_args()
    print(render(args.run, args.source))
