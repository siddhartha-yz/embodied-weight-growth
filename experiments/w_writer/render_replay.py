"""Render only the motor traces and ink that actually appear in the run logs."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from score_w import ink_paths, load_log, points, score_log

PAPER = "#f3f6fa"
CARD = "#ffffff"
INK = "#17263a"
MUTED = "#4a6075"
FAINT = "#dfe8f0"
GRID = "#e9eef4"
CYAN = "#176982"
GREEN = "#176647"
WARN = "#9a5214"
ACCENTS = ("#265fb1", "#8a4261")
FONT_DIR = Path("/usr/share/fonts/opentype/noto")
FALLBACK_BOARD = {"x": [0.09, 0.245], "y": [-0.10, 0.105]}


@lru_cache
def font(size: int, bold: bool = False):
    path = FONT_DIR / ("NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc")
    if path.exists():
        return ImageFont.truetype(str(path), size, index=2)
    path = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def text(d, at, s, size, color=INK, bold=False, anchor=None):
    d.text(at, str(s), font=font(size, bold), fill=color, anchor=anchor)


def wrapped(d, value: str, width: int, size: int, limit: int = 2) -> list[str]:
    value = " ".join(value.split())
    lines: list[str] = []
    while value and len(lines) < limit:
        lo, hi = 1, len(value)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if d.textlength(value[:mid] + "…", font=font(size)) <= width:
                lo = mid
            else:
                hi = mid - 1
        end = lo
        if end < len(value):
            space = value.rfind(" ", 0, end + 1)
            if space > end * 0.70:
                end = space
        lines.append(value[:end].strip())
        value = value[end:].strip()
    if value and lines:
        lines[-1] += "…"
    return lines


def valid_board(turns: list[dict]) -> dict:
    for turn in turns:
        board = turn["result"].get("board")
        if isinstance(board, dict):
            x, y = board.get("x"), board.get("y")
            if (isinstance(x, list) and isinstance(y, list) and len(x) == len(y) == 2
                    and all(isinstance(t, (float, int)) and math.isfinite(t) for t in x + y)
                    and x[0] < x[1] and y[0] < y[1]):
                return {"x": x, "y": y}
    return FALLBACK_BOARD


def _visible_segments(path, x0, x1, y0, y1):
    for p, q in zip(path[:-1], path[1:]):
        dx, dy = q - p
        lo, hi = 0.0, 1.0
        for a, b in ((-dx, p[0]-x0), (dx, x1-p[0]), (-dy, p[1]-y0), (dy, y1-p[1])):
            if abs(a) < 1e-12:
                if b < 0:
                    break
            elif a < 0:
                lo = max(lo, b / a)
            else:
                hi = min(hi, b / a)
            if lo > hi:
                break
        else:
            yield p + lo * (q-p), p + hi * (q-p)


def draw_canvas(d, rect, board, result, color, *, replay: bool, moved: bool = True):
    left, top, right, bottom = rect
    x0, x1 = board["x"]
    y0, y1 = board["y"]
    scale = min((right - left) / (x1 - x0), (bottom - top) / (y1 - y0))
    xpad = ((right - left) - (x1 - x0) * scale) / 2
    ypad = ((bottom - top) - (y1 - y0) * scale) / 2

    def coord(x, y):
        return (left + xpad + (x - x0) * scale, bottom - ypad - (y - y0) * scale)

    a, b = coord(x0, y1), coord(x1, y0)
    d.rectangle([a, b], fill="#fcfdff", outline=FAINT, width=2)
    step = 0.02
    for n in range(math.ceil(x0 / step), math.floor(x1 / step) + 1):
        x = n * step
        d.line((coord(x, y0), coord(x, y1)), fill=GRID, width=1)
    for n in range(math.ceil(y0 / step), math.floor(y1 / step) + 1):
        y = n * step
        d.line((coord(x0, y), coord(x1, y)), fill=GRID, width=1)

    def plot(path, stroke_color, width):
        if len(path) < 2:
            return
        if ((path >= [x0, y0]) & (path <= [x1, y1])).all():
            d.line([coord(*p) for p in path], fill=stroke_color, width=width, joint="curve")
        else:
            for p, q in _visible_segments(path, x0, x1, y0, y1):
                d.line((coord(*p), coord(*q)), fill=stroke_color, width=width)

    if replay and moved:
        for path in result.get("traces") or [result.get("trace", [])]:
            plot(points(path), "#80b7cb", 4)
    for stroke in ink_paths(result.get("strokes") or []):
        plot(stroke, color, 6)
        for pt in (stroke[0], stroke[-1]):
            if x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1:
                u, v = coord(*pt)
                d.ellipse([u-3, v-3, u+3, v+3], fill=color)

    if replay:
        state = result.get("state") or {}
        tip = points([state.get("tip", [])])
        if len(tip) and ((tip[0] >= [x0, y0]) & (tip[0] <= [x1, y1])).all():
            u, v = coord(*tip[0])
            d.ellipse([u-5, v-5, u+5, v+5], outline=INK, fill=CARD, width=2)


def new_artboard(title: str, subtitle: str):
    image = Image.new("RGB", (1240, 786), PAPER)
    d = ImageDraw.Draw(image)
    text(d, (44, 26), title, 38, bold=True)
    text(d, (46, 82), subtitle, 17, MUTED)
    return image, d


OP = {
    "observe": "观察", "status": "观察", "raw": "手动试探", "act": "手动试探",
    "pulse": "手动试探", "move": "执行运笔", "goto": "执行运笔", "run": "执行运笔",
    "drive": "手动试探", "use": "调用接口运笔", "install": "安装接口", "reset": "清空重试", "clear": "清空重试",
    "finish": "结束", "done": "结束", "pen": "调整笔状态", "set_pen": "调整笔状态",
}


def action_name(turn: dict) -> str:
    req = turn.get("request") or {}
    op = str(req.get("op", ""))
    if op.startswith(("write", "load", "compile", "commit")):
        return "生成接口"
    if "install" in op:
        return "安装接口"
    return OP.get(op, op or "反馈")


def meter(result: dict) -> str:
    m = result.get("meters") or {}
    moves, left, ticks = m.get("moves_used"), m.get("moves_left"), m.get("ticks_used")
    stats = []
    if isinstance(moves, int):
        stats.append(f"移动 {moves}/{moves + left}" if isinstance(left, int) else f"移动 {moves}")
    if isinstance(ticks, int):
        stats.append(f"仿真步 {ticks}")
    return "   ·   ".join(stats)


def panel(d, i: int, label: str, board: dict, turn: dict | None,
          *, summary: dict | None = None, replay: bool = False, frozen: bool = False):
    x = 40 + 590 * i
    y = 125
    color = ACCENTS[i]
    d.rounded_rectangle((x, y, x+570, 731), radius=23, fill=CARD, outline=FAINT, width=2)
    text(d, (x+27, y+16), label, 25, color, bold=True)
    if turn is None:
        text(d, (x+27, y+60), "没有非零墨迹", 16, WARN)
        draw_canvas(d, (x+32, 236, x+538, 642), board, {}, color, replay=False)
        return

    r = turn["result"]
    attempt = r.get("attempt", "—")
    ordinal = turn.get("turn", "—")
    if replay:
        name = action_name(turn)
        caption = f"反馈 {ordinal} · 画布 {attempt} · {name}"
        if frozen:
            caption += " · 本组已无后续反馈"
    else:
        caption = f"最佳完成画布 {attempt} · 最后反馈 {ordinal}"
    text(d, (x+27, y+60), caption, 17, MUTED)

    if replay and r.get("ok") is False:
        text(d, (x+27, y+90), "该请求未执行", 14, WARN)
    elif replay:
        text(d, (x+27, y+90), meter(r), 14, MUTED)
    elif summary is not None:
        passed = summary["passed"]
        status = "通过形状检查" if passed else "未通过形状检查"
        dim = f'{summary["width_mm"]:.0f} × {summary["height_mm"]:.0f} mm'
        text(d, (x+27, y+90), f'{status}   ·   {summary["matched_legs"]}/4 段斜笔   ·   {dim}', 15,
             GREEN if passed else WARN)

    draw_canvas(d, (x+30, 245, x+540, 638), board, r, color, replay=replay, moved=not frozen)
    if replay:
        note = str(turn.get("note") or "")
        if r.get("ok") is False:
            note = str(r.get("error") or r.get("message") or note)
        if note:
            for j, line in enumerate(wrapped(d, note, 512, 14)):
                text(d, (x+27, 656 + j*21), line, 14, MUTED)
    elif summary is not None:
        m = meter(r)
        if m:
            text(d, (x+27, 656), m, 15, MUTED)
        text(d, (x+27, 681), f'有效墨迹 {summary["ink_strokes"]} 笔', 15, MUTED)


def select_turn(turns: list[dict], summary: dict | None) -> dict | None:
    if summary is None:
        return None
    for turn in reversed(turns):
        if turn.get("turn") == summary["turn"] and ink_paths(turn["result"].get("strokes") or []):
            return turn
    return None


def draw_process(output: Path, runs: list[list[dict]], boards: list[dict], stats: list[dict]) -> None:
    """A printable storyboard of representative motion feedback."""
    image, d = new_artboard("从试电机到写完 W", "两组独立行动。左列是抬笔试探，之后按实际反馈顺序显示累积的墨迹。")
    for group, (run, board, stat) in enumerate(zip(runs, boards, stats)):
        y = 118 + group * 302
        color = ACCENTS[group]
        label = ("权重组", "代码组")[group]
        text(d, (43, y+5), label, 23, color, bold=True)
        raw = [t for t in run if (t.get("request") or {}).get("op") in {"drive", "raw", "pulse", "act"}
               and len(points(t["result"].get("trace", []))) > 1]
        best = stat["best"]
        draws = [
            t for t in run if best is not None and t["result"].get("attempt") == best["attempt"]
            and t["result"].get("pen") is True and len(points(t["result"].get("trace", []))) > 1
            and ink_paths(t["result"].get("strokes", []))
        ]
        choice = list(range(len(draws)))
        if len(choice) > 4:
            choice = list(np.linspace(0, len(draws)-1, 4, dtype=int))
        stages = [("抬笔试探", None)]
        if raw:
            stages[0] = (f"抬笔试探 · {len(raw)} 次",
                         {**raw[-1]["result"], "strokes": [],
                          "traces": [r["result"].get("trace", []) for r in raw]})
        stages += [(f"第 {k+1} 次落笔移动", draws[k]["result"]) for k in choice]
        for j, (name, result) in enumerate(stages):
            x = 41 + j * 234
            d.rounded_rectangle((x, y+43, x+220, y+286), radius=14, fill=CARD, outline=FAINT, width=2)
            text(d, (x+13, y+53), name, 16, MUTED)
            draw_canvas(d, (x+12, y+84, x+208, y+277), board, result or {}, color,
                        replay=j == 0)
    text(d, (44, 750), "淡蓝是抬笔时的位置变化；彩色是落笔后实际留下的线。画布使用统一比例。", 15, MUTED)
    image.save(output / "process.png")


def generate(logs: tuple[Path, Path], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    runs = [load_log(p) for p in logs]
    labels = ("权重组", "代码组")
    boards = [valid_board(run) for run in runs]
    stats = [score_log(run) for run in runs]
    report = {
        name: {
            **stat,
            "feedback_turns": len(run),
            "final_meters": run[-1]["result"].get("meters"),
            "last_operation": (run[-1].get("request") or {}).get("op"),
            "feedback": [
                {
                    "turn": row.get("turn"),
                    "attempt": row["result"].get("attempt"),
                    "op": (row.get("request") or {}).get("op"),
                    "ok": row["result"].get("ok"),
                    "pen": row["result"].get("pen"),
                    "policy_version": row["result"].get("policy_version"),
                    "moves_used": (row["result"].get("meters") or {}).get("moves_used"),
                    "ink_strokes": len(ink_paths(row["result"].get("strokes") or [])),
                    "tip_m": (row["result"].get("state") or {}).get("tip"),
                    "note": row.get("note"),
                }
                for row in run
            ],
        }
        for name, stat, run in zip(("weights", "code"), stats, runs)
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")

    title = "Astra 写 W：两种运笔接口"
    image, d = new_artboard(title, "每组展示评分最高的一次画布；没有描边底图，也没有平滑或补线。")
    for i, (label, board, run, stat) in enumerate(zip(labels, boards, runs, stats)):
        panel(d, i, label, board, select_turn(run, stat["best"]), summary=stat["best"])
    text(d, (44, 750), "网格边长 2 cm。判分检查连续墨迹、尺寸和四段交替斜笔；形状可直接目测。", 15, MUTED)
    image.save(output / "comparison.png")
    draw_process(output, runs, boards, stats)

    max_len = max(map(len, runs))
    frames, delays = [], []
    for index in range(max_len):
        frame, d = new_artboard(title, f"第 {index+1}/{max_len} 帧 · 按反馈顺序回放，两组不按物理时间同步。")
        for i, (label, board, run) in enumerate(zip(labels, boards, runs)):
            t = run[min(index, len(run)-1)]
            panel(d, i, label, board, t, replay=True, frozen=index >= len(run))
        text(d, (44, 750), "淡蓝是刚才的移动，彩色是画布墨迹，空心圆是当前笔尖。", 15, MUTED)
        # Restrict the adaptive GIF palette so saturated strokes stay crisp.
        frames.append(frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        delays.append(1100 if index + 1 < max_len else 3600)
    frames[0].save(output / "replay.gif", save_all=True, append_images=frames[1:], optimize=False,
                   duration=delays, loop=0, disposal=2)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    report = generate((args.weights, args.code), args.out_dir)
    for name, arm in report.items():
        print(name, json.dumps({"feedback_turns": arm["feedback_turns"], "best": arm["best"]},
                               ensure_ascii=False))


if __name__ == "__main__":
    main()
