"""Build a self-contained Chinese replay from recorded slot-push decisions."""
from __future__ import annotations

import argparse
from base64 import b64encode
from dataclasses import dataclass
from difflib import unified_diff
from html import escape
import json
from math import hypot
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from slot_view import BACKGROUND, GREEN, INK, MUTED, TIP, draw_world, font, render_view, yaw_deg

MOTION = {"drive", "use"}
OP_LABEL = {"drive": "原始电机", "use": "运行模块", "install": "安装模块", "reset": "重置场景", "finish": "结束"}
CONTACT_LABELS = (
    ("tip_block_ticks", "推杆–方块"), ("block_wall_ticks", "方块–槽壁"), ("tip_wall_ticks", "推杆–槽壁"),
)
CSS = """
:root{font-family:system-ui,"Noto Sans CJK SC","Microsoft Yahei",sans-serif;color:#182738;
background:#f3f7fb;line-height:1.6;font-size:16px}
*{box-sizing:border-box}
body{margin:0}
main{width:min(1190px,100% - 34px);margin:0 auto 80px}
h1,h2,h3,p{margin:0}
header{padding:52px 0 24px;border-bottom:1px solid #ccd9e6}
header p{max-width:66ch;color:#516174;margin-top:9px}
.eyebrow{font-size:.78rem;font-weight:800;letter-spacing:.17em;color:#445982}
h1{font-size:clamp(2.2rem,6vw,4.3rem);line-height:1.1;letter-spacing:-.035em}
h2{font-size:clamp(1.5rem,4vw,2.25rem);line-height:1.25}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0}
nav a{padding:6px 13px;background:white;border:1px solid #ccd9e6;border-radius:20px;color:#395587;text-decoration:none;font-weight:650}
nav a:hover,nav a:focus-visible{outline:2px solid #536db4}
.legend{margin:16px 0 24px;color:#516174;font-size:.91rem}
.overview{display:grid;grid-template-columns:repeat(auto-fit,minmax(214px,1fr));gap:12px}
.summary,.turn{border:1px solid #d3dfeb;background:white;border-radius:13px;padding:16px}
.summary h3{font-size:1rem;margin-bottom:3px}
.summary p{color:#516174;font-size:.9rem}
.summary .status{font-size:1.15rem;font-weight:800;color:#273b52}
.status.pass{color:#157055}
.run{margin:46px 0 0;scroll-margin:14px}
.run-head{display:flex;justify-content:space-between;gap:12px;align-items:start;padding-bottom:14px}
.run-head p{color:#516174;margin-top:4px;font-size:.91rem}
.turn{margin:10px 0;padding:0;overflow:hidden}
.turn summary{cursor:pointer;padding:14px 17px;display:flex;gap:8px 13px;align-items:center;flex-wrap:wrap}
.turn summary:before{content:"▸";color:#536d8d}
.turn[open] summary:before{content:"▾"}
.turn[open] summary{border-bottom:1px solid #e1eaf3}
.turn summary::-webkit-details-marker{display:none}
.turn summary strong{margin-right:4px}
.badge{display:inline-block;border-radius:16px;background:#eef2fb;color:#4b5b8a;padding:1px 8px;font-size:.8rem;font-weight:700;white-space:nowrap}
.badge.contact{background:#f2f0fd;color:#534493}
.badge.stall{background:#fff2e3;color:#894410}
.badge.pass{background:#e4f5ed;color:#14644d}
.turn.stall{border-left:4px solid #c07a37}
.turn .metric{color:#516174;font-size:.87rem;margin-left:auto}
.body{padding:16px}
.note{padding:9px 12px;margin-bottom:12px;border-radius:7px;background:#f3f7fb;overflow-wrap:anywhere}
.note span{font-size:.78rem;color:#516174;font-weight:750;margin-right:7px}
.next{margin:13px 0 0;padding:11px 13px;background:#edf5f1;border-left:3px solid #168065;border-radius:4px;
overflow-wrap:anywhere}.next strong{display:block;color:#226652;margin-bottom:3px;font-size:.87rem}
.facts{display:flex;flex-wrap:wrap;gap:5px 16px;color:#516174;font-size:.9rem;margin-bottom:13px}
.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}
figure{margin:0}
figcaption{font-size:.86rem;color:#516174;margin:4px 2px}
img{width:100%;height:auto;display:block;border:1px solid #dbe5ef;border-radius:8px}
details.inner{margin-top:12px;background:#f6f9fd;border-radius:8px;padding:9px 12px}
details.inner summary{display:list-item;padding:0;border:0;color:#40567a;font-weight:650;font-size:.9rem}
details.inner summary:before{content:none}
pre{margin:7px 0 0;white-space:pre-wrap;overflow-wrap:anywhere;max-height:380px;overflow:auto;
color:#23324a;font:13px/1.55 ui-monospace,monospace}
@media(max-width:620px){header{padding-top:28px}.views{grid-template-columns:1fr}.turn .metric{width:100%;margin-left:22px}.run-head{display:block}}
"""


@dataclass
class Run:
    label: str
    path: Path
    initial: dict
    events: list[dict]
    complete: bool

    @property
    def final(self):
        return self.events[-1]["result"] if self.events else self.initial


def read_run(value):
    label, path = value.split("=", 1)
    path = Path(path)
    initial = json.loads((path / "initial.json").read_text(encoding="utf-8"))
    with (path / "events.jsonl").open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    return Run(label, path, initial, events, (path / "final.json").is_file())


def movement(before, after):
    c0, c1 = before["state"]["block"], after["state"]["block"]
    mm = hypot(c1[0] - c0[0], c1[1] - c0[1]) * 1000
    a0, a1 = yaw_deg(before["state"]), yaw_deg(after["state"])
    delta = (a1 - a0 + 180) % 360 - 180
    return mm, a0, a1, delta


def run_stats(run):
    motions = [event for event in run.events if event["request"]["op"] in MOTION]
    contacts = [event for event in motions if event["result"].get("contact_summary", {}).get("tip_block_ticks", 0)]
    stalls = [event for event in motions if event["result"].get("contact_summary", {}).get("stall")]
    actual_installs = sum(
        event["request"]["op"] == "install" and event["result"]["ok"] for event in run.events
    )
    meters = run.final["meters"]
    return {"motions": meters["moves_used"], "ticks": meters["ticks_used"], "contacts": len(contacts),
            "stalls": len(stalls), "installs": actual_installs}


def image_markup(path, text, snapshot, work):
    if path.is_file():
        origin = "送入 Astra 的原图"
        payload = path.read_bytes()
    else:
        origin = "从记录重绘"
        render_view(snapshot, work / "view.png")
        payload = (work / "view.png").read_bytes()
    return (
        '<figure><img loading="lazy" src="data:image/png;base64,'
        + b64encode(payload).decode("ascii")
        + f'" alt="{escape(text)}"><figcaption>{escape(text)} · {origin}</figcaption></figure>'
    )


def format_json(payload):
    return escape(json.dumps(payload, ensure_ascii=False, indent=2))


def detail(title, content):
    return f'<details class="inner"><summary>{escape(title)}</summary><pre>{content}</pre></details>'


def policy_diff(before, after):
    if isinstance(before, str) and isinstance(after, str):
        lines = list(unified_diff(before.splitlines(), after.splitlines(), fromfile="旧版", tofile="新版",
                                  lineterm=""))
        return "\n".join(lines) if lines else "控制器代码完全相同"
    if isinstance(before, dict) and isinstance(after, dict):
        sizes = lambda layers: [len(layer["weight"]) for layer in layers]
        old, new = before["layers"], after["layers"]
        if sizes(old) != sizes(new):
            return f"层输出宽度：{sizes(old)} → {sizes(new)}（完整数值见本步指令）"
        changes = []
        for layer, (a, b) in enumerate(zip(old, new), 1):
            for i, (r0, r1) in enumerate(zip(a["weight"], b["weight"])):
                changes.extend((f"第 {layer} 层 W[{i},{j}]", x, y)
                               for j, (x, y) in enumerate(zip(r0, r1)) if x != y)
            changes.extend((f"第 {layer} 层 b[{i}]", x, y)
                           for i, (x, y) in enumerate(zip(a["bias"], b["bias"])) if x != y)
        if not changes:
            return "控制器权重完全相同"
        changes.sort(key=lambda item: abs(item[1] - item[2]), reverse=True)
        return f"{len(changes)} 个数值发生变化；以下按变化幅度列出前 12 项：\n" + "\n".join(
            f"{key}：{x:g} → {y:g}" for key, x, y in changes[:12]
        )
    return "首次安装，完整控制器见本步指令。"


def render_turn(run, event, next_event, before, previous_spec, work, *, first_contact, first_success):
    result, request, turn = event["result"], event["request"], event["turn"]
    op = request["op"]
    s = result.get("contact_summary") or {}
    active = s.get("tip_block_ticks", 0) or s.get("block_wall_ticks", 0)
    stall = s.get("stall")
    install = op == "install" and result["ok"]
    mm, a0, a1, da = movement(before, result)
    metric, tags, facts = "", [], []
    if op in MOTION:
        metric = f"方块净移动 {mm:.1f} mm · 转角 {da:+.1f}°"
        if active:
            tags.append(("有接触", "contact"))
        if stall:
            tags.append(("触壁且进展小", "stall"))
        facts.append(f"方块朝向 {a0:+.1f}° → {a1:+.1f}°")
        facts.append(" · ".join(f"{label} {s.get(key, 0)} tick" for key, label in CONTACT_LABELS))
        facts.append(f"请求 {request.get('ticks', 0)} tick")
        if op == "use":
            x, y = request["goal"]
            tx, ty = result["state"]["tip"]
            facts.append(f"推杆目标 ({x * 100:.2f}, {y * 100:.2f}) cm；到达 ({tx * 100:.2f}, {ty * 100:.2f}) cm")
        else:
            u0, u1 = request["u"]
            facts.append(f"两路电机 {u0:+.3f} / {u1:+.3f}")
    if install:
        tags.append((f"模块 v{before['policy_version']} → v{result['policy_version']}", "contact"))
    if not result["ok"]:
        tags.append(("指令未完整执行", "stall"))
        facts.append(result.get("error", ""))
    if first_contact:
        tags.append(("首次触块", "contact"))
    if first_success:
        tags.append(("首次到达目标", "pass"))
    is_open = stall or install or first_contact or first_success
    bits = [
        f'<details class="turn {"stall" if stall else ""}" {"open" if is_open else ""}><summary>',
        f"<strong>{turn:02d} · {escape(OP_LABEL.get(op, op))}</strong>",
        "".join(f'<span class="badge {style}">{escape(label)}</span>' for label, style in tags),
        f'<span class="metric">{escape(metric)}</span></summary><div class="body">',
    ]
    if event.get("note"):
        bits.append(f'<p class="note"><span>ASTRA 的说明</span>{escape(event["note"])}</p>')
    if facts:
        bits.append('<div class="facts">' + "".join(f"<span>{escape(fact)}</span>" for fact in facts) + "</div>")
    if op in MOTION or op == "reset":
        bits.append('<div class="views">')
        bits.append(image_markup(run.path / "views" / f"{turn:02d}.png", "下指令前看到的画面", before, work))
        bits.append(image_markup(run.path / "views" / f"{turn + 1:02d}.png", "执行后的反馈画面", result, work))
        bits.append("</div>")
        if next_event and next_event.get("note"):
            next_op = next_event["request"]["op"]
            reaction = f"随后第 {next_event['turn']} 步 · {OP_LABEL.get(next_op, next_op)}"
            bits.append(
                f'<aside class="next"><strong>{escape(reaction)}</strong>{escape(next_event["note"])}</aside>'
            )
    if install:
        bits.append(detail("控制器变了什么", escape(policy_diff(previous_spec, request["spec"]))))
    bits.append(detail("本步实际指令", format_json(request)))
    payload = run.path / "views" / f"{turn:02d}.json"
    if payload.is_file():
        bits.append(detail("下指令前送入 Astra 的数值反馈", escape(payload.read_text(encoding="utf-8"))))
    feedback = run.path / "views" / f"{turn + 1:02d}.json"
    if feedback.is_file():
        bits.append(detail("下一轮送入 Astra 的数值反馈", escape(feedback.read_text(encoding="utf-8"))))
    else:
        visible = {k: v for k, v in result.items() if k != "trace"}
        bits.append(detail("本步结束时的数值反馈（原始记录）", format_json(visible)))
    bits.append("</div></details>")
    return "".join(bits)


def summary_card(run):
    stats = run_stats(run)
    passed = run.final["success"]
    status = "已入槽" if passed else "未入槽" if run.complete else "当前未入槽"
    start_mode = " · 模块冻结" if run.initial.get("policy_frozen") else ""
    return (
        f'<article class="summary"><h3>{escape(run.label)}</h3><p class="status {"pass" if passed else ""}">{status}</p>'
        f'<p>初始朝向 {yaw_deg(run.initial["state"]):+.1f}°{start_mode}</p>'
        f'<p>{stats["motions"]} 段运动 / {stats["ticks"]} tick</p>'
        f'<p>{stats["contacts"]} 段碰到方块 · {stats["stalls"]} 段触壁少动 · {stats["installs"]} 次安装</p></article>'
    )


def render_html(runs, out):
    parts = [f'<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" '
             f'content="width=device-width,initial-scale=1"><title>Astra 推方块 · 逐步回放</title><style>{CSS}</style>',
             '<main><header><div class="eyebrow">EMBODIED WEIGHT GROWTH / RUN REPLAY</div><h1>Astra 推方块</h1>'
             '<p>点击任一步展开指令前后的画面、数值反馈和 Astra 的说明。'
             '首次触块、触壁且进展小的片段、控制器安装、首次入槽默认展开。</p></header><nav aria-label="跳到一组实验">']
    parts.extend(f'<a href="#run-{i}">{escape(run.label)}</a>' for i, run in enumerate(runs))
    parts.append('</nav><div class="overview">')
    parts.extend(summary_card(run) for run in runs)
    parts.append(
        '</div><p class="legend">图中紫色是方块，橙色是推杆，绿色是目标区，深灰色是槽壁。'
        '淡线来自上一段实际运动采样，不是规划路线。接触按仿真 tick 计，'
        '「触壁且进展小」是接触与位移/转角的观测标签，不表示一定无法继续。</p>'
    )
    with TemporaryDirectory() as temp:
        work = Path(temp)
        for i, run in enumerate(runs):
            parts.append(
                f'<section class="run" id="run-{i}"><div class="run-head"><div><h2>{escape(run.label)}</h2>'
                f'<p>{len(run.events)} 次决策 · 初始模块 v{run.initial["policy_version"]} · '
                f'结束模块 v{run.final["policy_version"]}</p></div></div>'
            )
            before, previous_spec = run.initial, None
            has_contact, has_success = False, bool(run.initial["success"])
            for j, event in enumerate(run.events):
                result = event["result"]
                touches = bool((result.get("contact_summary") or {}).get("tip_block_ticks"))
                passes = bool(result["success"])
                next_event = run.events[j + 1] if j + 1 < len(run.events) else None
                parts.append(render_turn(
                    run, event, next_event, before, previous_spec, work,
                    first_contact=touches and not has_contact, first_success=passes and not has_success,
                ))
                has_contact |= touches
                has_success |= passes
                if event["request"]["op"] == "install" and event["result"]["ok"]:
                    previous_spec = event["request"]["spec"]
                before = event["result"]
            parts.append("</section>")
    parts.append("</main></html>")
    out.write_text("\n".join(parts), encoding="utf-8")


def frame_choices(run):
    final_turn = run.events[-1]["turn"] if run.events else 0
    movement_events = [event for event in run.events if event["request"]["op"] in MOTION]
    stalls = [event for event in movement_events if event["result"].get("contact_summary", {}).get("stall")]
    contact = [event for event in movement_events if event["result"].get("contact_summary", {}).get("tip_block_ticks", 0)]
    picks = [("初态", run.initial)]
    chosen = set()
    if contact:
        event = contact[0]
        picks.append((f"第 {event['turn']} 步 · 初次触块", event["result"]))
        chosen.add(event["turn"])
    if stalls:
        event = stalls[0]
        if event["turn"] in chosen:
            picks[-1] = (f"第 {event['turn']} 步 · 触壁少动", event["result"])
        else:
            picks.append((f"第 {event['turn']} 步 · 触壁少动", event["result"]))
    else:
        same_pose = lambda state: all(state[key] == run.final["state"][key]
                                      for key in ("block", "block_angle", "tip"))
        turns = [event for event in contact[1:] if not same_pose(event["result"]["state"])]
        if turns:
            delta = lambda event: abs(event["result"]["contact_summary"]["block_rotation_deg"])
            event = max(turns, key=delta)
            label = "改变朝向" if delta(event) >= 5 else "继续推进"
            picks.append((f"第 {event['turn']} 步 · {label}", event["result"]))
    if not picks[-1][0].startswith(f"第 {final_turn} 步 ·"):
        picks.append((f"第 {final_turn} 步 · 末态" if final_turn else "末态", run.final))
    else:
        picks[-1] = (f"第 {final_turn} 步 · 末态", run.final)
    return picks


def render_overview(runs, out):
    width, tile, step_h = 1200, 224, 236
    image = Image.new("RGB", (width, 156 + step_h * len(runs)), BACKGROUND)
    d = ImageDraw.Draw(image)
    d.text((28, 17), "把方块推入窄槽", fill=INK, font=font(35, True, chinese=True))
    d.text((29, 68), "紫色方块   ·   橙色推杆   ·   深灰槽壁   ·   绿色目标区    |    实际观察的关键帧（完整过程见逐步回放）",
           fill=MUTED, font=font(16, chinese=True))
    for i, run in enumerate(runs):
        y = 112 + i * step_h
        d.rounded_rectangle((26, y, width - 26, y + step_h - 12), 12,
                            fill="white", outline="#d4e0ec", width=1)
        stats = run_stats(run)
        d.text((43, y + 25), run.label, fill=INK, font=font(19, True, chinese=True))
        status = "已入槽" if run.final["success"] else "未入槽" if run.complete else "当前未入槽"
        d.text((44, y + 64), status, fill=GREEN if run.final["success"] else TIP,
               font=font(20, True, chinese=True))
        for j, txt in enumerate((
            f"{stats['motions']} 段运动 · {stats['ticks']} tick",
            f"{stats['contacts']} 段触块 · {stats['stalls']} 段触壁少动",
            (f"沿用冻结模块 v{run.initial['policy_version']}" if run.initial.get("policy_frozen")
             else f"模块 v{run.initial['policy_version']} → v{run.final['policy_version']}"),
        )):
            d.text((44, y + 110 + j * 25), txt, fill=MUTED, font=font(14, chinese=True))
        for j, (label, frame) in enumerate(frame_choices(run)):
            x = 245 + j * tile
            d.text((x + tile / 2, y + 17), label, fill=INK, anchor="mt",
                   font=font(15, True, chinese=True))
            draw_world(d, frame, (x + 12, y + 46, x + tile - 10, y + 190), labels=False)
            d.text((x + tile / 2, y + 197), f"{yaw_deg(frame['state']):+.1f}°", fill=MUTED,
                   anchor="mt", font=font(13))
    image.save(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", metavar="LABEL=DIR", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = [read_run(value) for value in args.run]
    args.out.mkdir(parents=True, exist_ok=True)
    render_html(runs, args.out / "index.html")
    render_overview(runs, args.out / "overview.png")


if __name__ == "__main__":
    main()
