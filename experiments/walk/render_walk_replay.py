"""Render walking progress and exact model feedback from recorded run directories."""
from __future__ import annotations

import argparse
from base64 import b64encode
from dataclasses import dataclass
from difflib import unified_diff
from html import escape
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from walk_view import BG, BLUE, GREEN, GRID, INK, MUTED, RED, WHITE, render_view, text, world

MOTION = {"drive", "run"}
OP = {"drive": "试电机", "run": "运行模块", "install": "安装模块", "reset": "重新站起", "finish": "结束"}
FALL = {"torso touched the floor":"躯干触地", "torso height dropped below 0.8 m":"躯干高度低于 0.8 m",
        "torso tilted beyond 1.0 rad":"躯干倾角超过 1.0 rad"}

CSS = """
:root{font:16px/1.6 system-ui,"Noto Sans CJK SC","Microsoft Yahei",sans-serif;color:#1e2d36;background:#f3f6f8}
*{box-sizing:border-box}
body{margin:0}main{width:min(1160px,100% - 36px);margin:0 auto 65px}
h1,h2,h3,p,figure{margin:0}h1{font-size:clamp(2.4rem,7vw,4.4rem);line-height:1.15;letter-spacing:-.045em}
h2{font-size:clamp(1.55rem,4vw,2.45rem);line-height:1.2}h3{font-size:1.04rem}
header{padding:48px 0 23px;border-bottom:1px solid #cddae0}
header p{color:#536874;margin-top:10px;max-width:74ch}.eyebrow{letter-spacing:.15em;font-size:.8rem;font-weight:800;color:#4165af}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0}a{color:#275390;text-underline-offset:.19em}
nav a{background:white;border:1px solid #ccdbe3;border-radius:18px;padding:5px 12px;text-decoration:none;font-weight:650}
nav a:focus-visible,nav a:hover{outline:2px solid #4165af}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:12px}
.card,.turn{background:white;border:1px solid #d1dce2;border-radius:12px}
.card{padding:14px 16px}.card p,.subtle{font-size:.9rem;color:#536874}
.verdict{font-size:1.18rem;font-weight:800;color:#795940}.pass{color:#28744c}
.phase{margin-top:42px;scroll-margin:12px}.phase .subtle{margin:5px 0 12px}
.turn{margin:9px 0;overflow:hidden}.turn.fall{border-left:4px solid #b4443c}
.turn>summary{padding:13px 16px;cursor:pointer;display:flex;gap:6px 13px;align-items:center;flex-wrap:wrap}
.turn>summary:before{content:"▸";color:#536874}.turn[open]>summary:before{content:"▾"}
.turn>summary::-webkit-details-marker{display:none}.turn[open]>summary{border-bottom:1px solid #e5ecf0}
.metric{margin-left:auto;color:#536874;font-size:.88rem}.badge{font-size:.8rem;font-weight:750;padding:1px 8px;border-radius:12px;background:#e8eff9;color:#365b91}
.badge.red{background:#fbeae7;color:#8e372e}.badge.green{background:#e3f2e9;color:#286a46}
.body{padding:14px 16px}.note{background:#f0f5f9;border-radius:8px;padding:9px 11px;overflow-wrap:anywhere;margin-bottom:11px}
.note strong,.next strong{display:block;font-size:.8rem;color:#536874}.facts{display:flex;flex-wrap:wrap;gap:4px 15px;color:#536874;margin-bottom:10px;font-size:.92rem}
.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{display:block;width:100%;height:auto;border:1px solid #e1e9ed;border-radius:8px}
figcaption{font-size:.84rem;color:#536874;margin:3px 2px}
.next{background:#e9f4f2;border-left:3px solid #147f75;padding:8px 10px;margin-top:10px;overflow-wrap:anywhere}
.inner{margin-top:10px;background:#f4f7fa;padding:7px 11px;border-radius:8px}
.inner summary{cursor:pointer;font-size:.88rem;font-weight:700;color:#365b91}pre{margin:5px 0 0;max-height:380px;overflow:auto;
white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.55 ui-monospace,monospace}
.tablebox{overflow-x:auto;border-radius:10px;background:white;border:1px solid #d1dce2}
table{width:100%;border-collapse:collapse;white-space:nowrap;text-align:left;font-size:.92rem}
th,td{padding:9px 12px;border-bottom:1px solid #e5ecf0}th{background:#eaf1f5;color:#415661}
tbody tr:last-child td{border-bottom:0}
@media(max-width:620px){header{padding-top:26px}.views{grid-template-columns:1fr}.metric{width:100%;margin-left:19px}}
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

    def best(self):
        # Choose one observed snapshot; do not combine maxima from separate attempts.
        rows = [(0, self.initial)] + [(e["turn"], e["result"]) for e in self.events
                                     if e["request"]["op"] in MOTION]
        return max(rows, key=lambda row: (
            row[1]["success"], row[1]["walk"].get("first_success_s") is not None,
            row[1]["walk"]["alternating_steps"],
            row[1]["walk"]["distance_m"], row[1]["walk"]["elapsed_s"],
        ))

    def attempts(self):
        return {e["result"]["attempt"] for e in self.events} | {self.initial["attempt"]}

    def falls(self):
        return {e["result"]["attempt"] for e in self.events if e["result"]["walk"]["fall_reason"]}

    def installs(self):
        return [e for e in self.events if e["request"]["op"] == "install" and e["result"]["ok"]]


def load_run(value):
    label, path = value.split("=", 1)
    path = Path(path)
    initial = json.loads((path / "initial.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    return Run(label, path, initial, events, (path / "final.json").is_file())


def slug(index):
    return f"phase-{index+1}"


def verdict(run):
    best = run.best()[1]
    if best["success"]:
        return "达到行走判据"
    if best["walk"].get("first_success_s") is not None:
        return "曾达标，动作结束时未达标"
    return "未达行走判据" if run.complete else "仍在试跑"


def statline(run):
    meters = run.final["meters"]
    suffix = " · 控制器冻结" if run.initial.get("policy_frozen") else ""
    return (f'{len(run.events)} 次决策 · {meters["moves_used"]} 段运动 · {meters["ticks_used"]} 个控制步'
            f' · {len(run.falls())} 次摔倒 · {len(run.installs())} 次安装{suffix}')


def image_src(payload, format_="png"):
    return f'data:image/{format_};base64,{b64encode(payload).decode("ascii")}'


def figure(path, title, snapshot, work):
    if path.is_file():
        origin, payload = "送入 Astra 的原图", path.read_bytes()
    else:
        img = work / "snapshot.png"
        render_view(snapshot, img)
        origin, payload = "由实测记录重绘", img.read_bytes()
    title = escape(title)
    return f'<figure><img loading="lazy" src="{image_src(payload)}" alt="{title}"><figcaption>{title} · {origin}</figcaption></figure>'


def detail(title, data, *, formatted=False):
    value = data if formatted else json.dumps(data, ensure_ascii=False, indent=2)
    return f'<details class="inner"><summary>{title}</summary><pre>{escape(value)}</pre></details>'


def spec_delta(before, after):
    if before is None:
        return "首次安装"
    if before == after:
        return "原样重装"
    if isinstance(before, str) and isinstance(after, str):
        return "\n".join(unified_diff(before.splitlines(), after.splitlines(), "旧版", "新版", lineterm=""))
    if isinstance(before, dict) and isinstance(after, dict):
        old, new = before["layers"], after["layers"]
        old_n, new_n = [len(layer["rows"]) for layer in old], [len(layer["rows"]) for layer in new]
        if old_n != new_n:
            return f"各层输出宽度：{old_n} → {new_n}"
        changes = []
        for layer, (a, b) in enumerate(zip(old, new), 1):
            for i, (r0, r1) in enumerate(zip(a["rows"], b["rows"])):
                w0 = {t["feature"]: t["coef"] for t in r0["terms"]}
                w1 = {t["feature"]: t["coef"] for t in r1["terms"]}
                for j in sorted(w0.keys() | w1.keys()):
                    x, y = w0.get(j, 0), w1.get(j, 0)
                    if x != y:
                        changes.append(f"L{layer} W[{i},{j}]: {x} → {y}")
                if r0["bias"] != r1["bias"]:
                    changes.append(f'L{layer} b[{i}]: {r0["bias"]} → {r1["bias"]}')
        return "\n".join(changes) or "数值未变"
    return "模块形式改变"


def turn_html(run, index, work, previous_spec):
    e = run.events[index]
    n, before, after = e["turn"], run.initial if index == 0 else run.events[index-1]["result"], e["result"]
    request = e["request"]
    op = request["op"]
    prior_walk, walk = before["walk"], after["walk"]
    a, b = before["state"], after["state"]
    new_fall = walk["fall_reason"] and not prior_walk["fall_reason"]
    metrics = (
        f'位移 {b["x"]-a["x"]:+.3f} m · 交替落脚 {walk["alternating_steps"]}'
        if op in MOTION else f'尝试 {after["attempt"]} · 模块 v{after["policy_version"]}'
    )
    badges = "".join(f'<span class="badge {color}">{label}</span>' for flag,color,label in (
        (new_fall,"red","摔倒"), (after["success"],"green","达标"), (not after["ok"],"red","动作被拒绝"),
        (not after["success"] and walk.get("first_success_s") is not None
         and prior_walk.get("first_success_s") is None,"green","本段曾达标"),
    ) if flag)
    summary = f'<strong>第 {n:02} 步 · {OP.get(op,op)}</strong>{badges}<span class="metric">{metrics}</span>'
    body = f'<div class="note"><strong>这一步为何这样做（Astra 原文）</strong>{escape(e["note"])}</div>'
    if after.get("error"):
        body += f'<div class="note"><strong>执行反馈</strong>{escape(after["error"])}</div>'
    if op in MOTION:
        pitch = lambda s: math.degrees(s["pitch"])
        facts = (f'速度 {a["vx"]:+.2f} → {b["vx"]:+.2f} m/s', f'高度 {a["z"]:.2f} → {b["z"]:.2f} m',
                 f'俯仰 {pitch(a):+.1f}° → {pitch(b):+.1f}°',
                 f'交替落脚 {prior_walk["alternating_steps"]} → {walk["alternating_steps"]}')
        if new_fall:
            reason = walk["fall_reason"]
            facts = (*facts, f'终止：{FALL.get(reason,reason)}')
        for landing in after.get("op_summary", {}).get("new_touchdowns", []):
            side = {"R":"右脚","L":"左脚"}[landing["foot"]]
            facts = (*facts, f'{side}有效落脚：向前 {landing["advance_m"]*1000:.0f} mm'
                            f'（{landing["time_s"]:.2f} s）')
        body += '<div class="facts">' + "".join(f'<span>{escape(x)}</span>' for x in facts) + '</div>'
    if op in MOTION | {"reset"}:
        view = run.path / "views"
        next_e = run.events[index+1] if index+1 < len(run.events) else None
        next_id = next_e["turn"] if next_e else None
        after_view = view / f"{next_id:02d}.png" if next_id else work / "no-original"
        body += '<div class="views">' + figure(view / f"{n:02d}.png", "动作前", before, work)
        body += figure(after_view, "动作后", after, work) + '</div>'
        if next_e:
            next_note = escape(next_e["note"])
            body += f'<div class="next"><strong>看到这个结果后，Astra 的下一步</strong>{next_note}</div>'
    if op == "install" and after["ok"]:
        body += detail("模块改动", spec_delta(previous_spec, request["spec"]), formatted=True)
    view_json = run.path / "views" / f"{n:02d}.json"
    if view_json.is_file():
        body += detail("这一步送给 Astra 的观测与文字", json.loads(view_json.read_text(encoding="utf-8")))
    out = {k: after[k] for k in ("state","walk","op_summary","meters") if k in after}
    body += detail("动作与执行结果", {"request": request, **out})
    kind = "turn fall" if new_fall else "turn"
    return f'<details class="{kind}"><summary>{summary}</summary><div class="body">{body}</div></details>'


def attempt_states(run):
    """Extract one decisive attempt, retaining only actual simulator frames."""
    _, best = run.best()
    attempt = best["attempt"]
    first = run.initial["state"] if run.initial["attempt"] == attempt else next(
        e["result"]["state"] for e in run.events if e["result"]["attempt"] == attempt
    )
    trace = [first]
    terminal = best
    for e in run.events:
        snap = e["result"]
        if snap["attempt"] != attempt:
            continue
        terminal = snap
        if e["request"]["op"] in MOTION:
            trace.extend(snap["trace"][1:])
    return trace, terminal


def animate(trace, title, terminal, dt, path):
    if len(trace) < 2:
        return None
    stride = max(1, round(.09 / dt), math.ceil(len(trace)/120))
    indices = sorted(set([*range(0,len(trace),stride), len(trace)-1]))
    peak = max(row["x"] for row in trace)
    far = min(row["x"] for row in trace)
    fixed_camera = (peak+far)/2 if peak-far < 1.6 else None
    frames, durations = [], []
    for i, j in enumerate(indices):
        state = trace[j]
        image = Image.new("RGB", (520, 330), BG)
        d = ImageDraw.Draw(image)
        text(d, (15, 13), title, 21, chinese=True, bold=True)
        suffix = " · 摔倒" if j == len(trace)-1 and terminal["walk"]["fall_reason"] else ""
        text(d, (15, 45), f'{state["t"]:.2f} 秒 · x {state["x"]:+.2f} m{suffix}', 14, chinese=True, fill=MUTED)
        world(d, state, (15, 78, 505, 310), past=trace[:j+1:stride], camera=fixed_camera)
        frames.append(image)
        if i+1 < len(indices):
            durations.append(max(40, round((trace[indices[i+1]]["t"]-state["t"])*1000)))
        else:
            durations.append(700)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return path


def gif(run, path):
    trace, terminal = attempt_states(run)
    return animate(trace, run.label, terminal, run.initial["dt"], path)


def overview(runs, path):
    row_h = 215
    image = Image.new("RGB", (1150, 106+row_h*len(runs)), BG)
    d = ImageDraw.Draw(image)
    text(d, (26, 18), "平面双足 · 实测行走", 33, chinese=True, bold=True)
    text(d, (27, 63), "蓝：右腿　橙：左腿　每行取该组表现最好的一次尝试", 17, chinese=True, fill=MUTED)
    for i, run in enumerate(runs):
        top = 99+row_h*i
        d.rounded_rectangle((20, top, 1130, top+201), radius=13, fill=WHITE, outline="#d1dce2")
        turn, best = run.best()
        trace, end = attempt_states(run)
        state = best["state"]
        best_label = "最佳结果 · 摔倒" if best["walk"]["fall_reason"] else "最佳动作结果"
        end_label = "该次尝试摔倒时" if end["walk"]["fall_reason"] else "该次尝试终态"
        poses = [(trace[0], "起始姿态"), (state, best_label), (end["state"], end_label)]
        if poses[-1][0] == poses[1][0]:
            poses[-1] = (None, "")
        color = GREEN if best["success"] else RED
        text(d, (37, top+15), run.label, 23, chinese=True, bold=True)
        text(d, (37, top+52), verdict(run), 19, chinese=True, bold=True, fill=color)
        text(d, (37, top+93), f'前进 {best["walk"]["distance_m"]:+.2f} m', 16, chinese=True)
        text(d, (37, top+120), f'交替落脚 {best["walk"]["alternating_steps"]} · 摔倒 {len(run.falls())}', 16, chinese=True)
        op = next((e["request"]["op"] for e in run.events if e["turn"] == turn), None)
        source = "原始探测" if op == "drive" else f'该段 v{best["policy_version"]}'
        text(d, (37, top+150), f'{run.final["meters"]["moves_used"]} 段运动 · {source}', 15, chinese=True, fill=MUTED)
        if poses[-1][0] is None and len(trace) > 2:
            poses[1] = (trace[len(trace)//2], "尝试中段")
            poses[2] = (state, best_label)
        for j,(s,title) in enumerate(poses):
            if s is None:
                continue
            left = 302+j*270
            text(d, (left+127, top+13), title, 15, chinese=True, bold=True, anchor="mt")
            world(d, s, (left, top+38, left+253, top+166), grid=True, labels=False)
            text(d, (left+126, top+173), f'{s["t"]:.2f} s  ·  x {s["x"]:+.2f} m', 13, fill=MUTED, anchor="mt")
    image.save(path)


def frozen_table(audit, run_root, out):
    if not audit or "frozen" not in audit:
        return ""
    rows = []
    for arm, data in audit["frozen"].items():
        label = {"weights":"权重组","code":"代码组"}[arm]
        for scene, trials in data.get("trials", {}).items():
            scene_label = {"train":"原姿态", "transfer":"新初始倾角"}.get(scene,scene)
            for trial in trials:
                outcome = "达标" if trial["success"] else "摔倒" if trial["fall_reason"] else "未达标"
                row = [label, scene_label, f'v{trial["version"]}', f'{trial["distance_m"]:+.3f} m',
                       str(trial["alternating_steps"]), f'{trial["elapsed_s"]:.2f} s', outcome]
                record = json.loads((run_root / trial["record"]).read_text(encoding="utf-8"))
                name = f'frozen-{arm}-{scene}-v{trial["version"]:02}.gif'
                payload = animate(record["result"]["trace"], f'{label} · {scene_label} · v{trial["version"]}',
                                  record["result"], record["initial"]["dt"], out / name)
                link = f'<a href="{name}">实测回放</a>' if payload else "没有运动记录"
                rows.append("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in row) + f"<td>{link}</td></tr>")
    if not rows:
        return ""
    header = ("组别","起始姿态","模块","前进距离","交替落脚","时长","结果","动作")
    cells = "".join(f"<th>{v}</th>" for v in header)
    return ('<section class="phase" id="frozen"><h2>只运行保存下来的模块</h2>'
            '<p class="subtle">从相同起始状态重放，过程没有 Astra 再选动作。第二种姿态只改变起始俯仰角。</p>'
            f'<div class="tablebox"><table><thead><tr>{cells}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')


def report(runs, out, audit_path=None):
    out.mkdir(parents=True, exist_ok=True)
    overview(runs, out / "overview.png")
    gifs = [gif(run, out / f"{slug(i)}.gif") for i, run in enumerate(runs)]
    with TemporaryDirectory(prefix="walk-report-") as tmp:
        work = Path(tmp)
        frozen = frozen_table(json.loads(audit_path.read_text(encoding="utf-8")), audit_path.parent, out) if audit_path else ""
        nav = '<nav>' + "".join(f'<a href="#{slug(i)}">{escape(r.label)}</a>' for i,r in enumerate(runs))
        nav += ('<a href="#frozen">冻结模块复测</a>' if frozen else '') + '</nav>'
        cards, phases = [], []
        for i, run in enumerate(runs):
            turn, best = run.best()
            style = " pass" if best["success"] else ""
            fell = " · 摔倒" if best["walk"]["fall_reason"] else ""
            cards.append(f'<div class="card"><h3>{escape(run.label)}</h3><div class="verdict{style}">{verdict(run)}</div>'
                         f'<p>最佳动作结果：前进 {best["walk"]["distance_m"]:+.3f} m · 交替落脚 {best["walk"]["alternating_steps"]}'
                         f'（第 {turn} 步{fell}）</p><p>{statline(run)}</p></div>')
            anim = ""
            if gifs[i]:
                name = gifs[i].name
                anim = f'<p class="subtle"><a href="{name}">播放这次尝试的实测动作 GIF</a></p>'
            rows = []
            prior = None
            for j,e in enumerate(run.events):
                rows.append(turn_html(run, j, work, prior))
                if e["request"]["op"] == "install" and e["result"]["ok"]:
                    prior = e["request"]["spec"]
            phases.append(f'<section id="{slug(i)}" class="phase"><h2>{escape(run.label)}</h2>'
                          f'<p class="subtle">{statline(run)}</p>{anim}{"".join(rows)}</section>')
        scene = runs[0].initial["scene"]
        criteria = (f'判定：保持直立、至少 {scene["minimum_time_s"]:g} 秒、向前至少 {scene["minimum_distance_m"]:g} 米、'
                    f'至少 {scene["minimum_alternating_steps"]} 次有效交替落脚。')
        html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>平面双足 · 从站立到迈步</title><style>{CSS}</style><body><main>
<header><p class="eyebrow">PLANAR WALKER / 决策回放</p><h1>从站立到迈步</h1>
<p>前进距离、交替落脚和摔倒分开看：向前滑动并不等于走路。{criteria}展开动作可看 Astra 收到的画面、状态和它随后怎么试。</p></header>
{nav}<div class="cards">{"".join(cards)}</div>{frozen}{"".join(phases)}</main></body></html>"""
        (out / "index.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--audit", type=Path, help="include frozen-module results from audit_walk.py")
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    args = ap.parse_args()
    report([load_run(v) for v in args.run], args.out, args.audit)


if __name__ == "__main__":
    main()
