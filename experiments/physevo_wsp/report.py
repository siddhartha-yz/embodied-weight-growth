"""Render a Chinese replay of a native digit-arrangement run."""

from __future__ import annotations

import argparse
from html import escape
from itertools import zip_longest
import json
import math
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps


OP = {"move_tcp": "移动夹爪", "grip": "开合夹爪", "done": "结束本次搬运",
      "joint": "移动关节", "pose_preview": "预检目标位姿",
      "fail_reobserve": "停止并重新观察", "segment_budget": "达到本段预算"}
VISUAL = {"placed": "看起来已放上", "not_placed": "未放上", "uncertain": "看不确定"}
RECEIPT = {
    "completed": "执行完成", "success": "执行完成", "succeeded": "执行完成", "ok": "执行完成",
    "failed": "执行失败", "rejected": "安全边界拒绝", "partial": "仅部分执行",
    "timed_out": "超时", "timeout": "超时", "planned": "可规划（未执行）",
}
REASON = {"tracking_error": "位姿跟踪未达标", "path_infeasible": "未找到可行轨迹"}
OK = {"completed", "success", "succeeded", "ok", "planned"}

CSS = """
:root{font:16px/1.62 system-ui,"Noto Sans CJK SC","Microsoft Yahei",sans-serif;color:#19262a;background:#f4f7f6}
*{box-sizing:border-box}body{margin:0}main{width:min(1100px,100% - 32px);margin:0 auto 65px}
header{padding:40px 0 20px}h1,h2,h3,p,figure{margin:0}h1{font-size:clamp(2.1rem,5vw,3.6rem);line-height:1.16;letter-spacing:-.03em}
h2{font-size:1.42rem;margin-bottom:8px}h3{font-size:1.04rem}.eyebrow{font-size:.78rem;font-weight:750;letter-spacing:.13em;color:#366357}
.muted{color:#4d6761}.subtitle{margin-top:8px}.meta{display:flex;flex-wrap:wrap;gap:8px;margin:13px 0}
.tag{background:#e4eeea;color:#31574c;padding:3px 9px;border-radius:6px;font-size:.83rem;font-weight:650;display:inline-block}
.good{background:#e0f2e8;color:#216343}.warn{background:#faedda;color:#80531f}.neutral{background:#ebeff3;color:#445967}
.overview{width:100%;border:1px solid #cfdbd8;border-radius:10px;background:#fff;display:block}
section{margin-top:24px}.links a{color:#285e52;margin-right:12px}
.video{margin:4px 0 22px}.video video{display:block;width:100%;border-radius:10px;background:#101c20;aspect-ratio:1200/760}
.video-recording video{aspect-ratio:4/3;max-height:68vh;object-fit:contain}
.video figcaption{margin-top:8px}.video figcaption a{color:#285e52}
.install{border-left:3px solid #619f8a;padding:8px 12px;margin:8px 0;background:#eaf2ef;border-radius:3px;overflow-wrap:anywhere}
details.attempt{border:1px solid #d6e1dd;background:#fff;border-radius:9px;margin:9px 0;overflow:hidden}
summary{cursor:pointer;padding:9px 12px}.attempt-body{padding:4px 12px 14px}
.attempt-body p{margin:7px 0;overflow-wrap:anywhere}
.scroll{overflow:auto}table{border-collapse:collapse;min-width:630px;width:100%;font-size:.88rem}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid #e0eae6;vertical-align:top}
td small{display:block;color:#566c65;overflow-wrap:anywhere;min-width:140px}
.run-error{margin-top:13px;padding:8px 12px;background:#faedda;color:#654218;border-radius:6px;overflow-wrap:anywhere}
code{font-family:ui-monospace,monospace;font-size:.89em}
"""


def load(path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def relative_link(run, path):
    """Report lives in run/report; keep links portable with the run bundle."""
    return "../" + quote(path.relative_to(run).as_posix(), safe="/")


def attempts(events):
    """Group native actions with the next visual outcome, preserving an unfinished try."""
    current, version, groups = [], None, []
    for event in events:
        kind = event.get("event")
        if kind == "install":
            version = event["version"]
        elif kind == "policy_step":
            current.append(event)
        elif kind == "outcome_observed":
            groups.append((event["transfer"], event.get("version", version), current, event))
            current = []
    if current:
        groups.append((current[-1]["transfer"], version, current, None))
    return groups


def camera_photo(directory, camera):
    for suffix in (".png", ".jpg", ".jpeg"):
        path = directory / f"{camera}{suffix}"
        if path.is_file():
            return path
    return None


def photo(directory):
    raw = camera_photo(directory, "cam_head")
    marked = directory / "head-marked.png"
    return raw or (marked if marked.is_file() else None)


def observed_frames(run):
    first = photo(run / "initial")
    last_path = next(
        (p for p in (photo(d) for d in sorted(run.glob("transfer-*-try-*"), reverse=True)) if p),
        None,
    )
    last = photo(run / "final") or last_path
    return first, last


def font(size, bold=False):
    style = "Bold" if bold else "Regular"
    file = Path("/usr/share/fonts/opentype/noto") / f"NotoSansCJK-{style}.ttc"
    return ImageFont.truetype(str(file), size, index=2) if file.is_file() else ImageFont.load_default(size)


def make_overview(path, first, last, status, summary):
    canvas = Image.new("RGB", (1040, 520), "#f0f5f3")
    d = ImageDraw.Draw(canvas)
    d.text((22, 13), "RoboDojo · 数字排列", font=font(27, True), fill="#1b322b")
    d.text((22, 53), f"{status}   {summary}", font=font(16), fill="#3e6054")
    for x, caption, source in ((22, "开场头部相机", first), (530, "最后记录的头部相机", last)):
        d.rounded_rectangle((x, 92, x + 488, 502), radius=10, fill="#fff", outline="#cfdfd9")
        d.text((x + 13, 101), caption, font=font(16, True), fill="#233f37")
        if source:
            with Image.open(source) as im:
                frame = ImageOps.contain(im.convert("RGB"), (466, 354), Image.Resampling.LANCZOS)
            canvas.paste(frame, (x + (488 - frame.width)//2, 135 + (354 - frame.height)//2))
        else:
            d.text((x + 15, 144), "暂无相机记录", font=font(15), fill="#506d62")
    canvas.save(path)


def is_number(value):
    return type(value) in (int, float)


def vector_label(values, count, precision, unit):
    if not isinstance(values, list) or len(values) != count or not all(is_number(v) for v in values):
        return "无效数值"
    return ", ".join(f"{v:.{precision}f}" for v in values) + f" {unit}"


def row_html(row, has_arm):
    op = row["operation"]
    if op in {"move_tcp", "pose_preview"} and row.get("xyz") is not None:
        command = vector_label(row["xyz"], 3, 3, "m")
        if is_number(row.get("yaw")):
            command += f"；偏航 {row['yaw']:.2f} rad"
    elif op == "grip" and "jaw" in row:
        command = f"开合指令 {row['jaw']:.2f}" if is_number(row['jaw']) else "无效开合指令"
    elif op == "joint" and row.get("q") is not None:
        command = vector_label(row["q"], 6, 2, "rad")
    elif op in {"move_tcp", "grip", "joint", "pose_preview"}:
        command = "未记录动作目标"
    else:
        command = "未发物理动作"
    receipt = row.get("receipt") or {}
    status = receipt.get("status")
    result = RECEIPT.get(status, status or "—")
    reason = receipt.get("reason") or receipt.get("failure_code") or receipt.get("error") or row.get("reason")
    status_html = escape(str(result))
    if reason and status not in OK:
        status_html += f"<small>{escape(str(reason))}</small>"
    fb = row.get("feedback") or {}
    if op == "move_tcp" and all(k in fb for k in ("tcp_x", "tcp_y", "tcp_z")):
        status_html += f"<small>实测 {vector_label([fb[k] for k in ('tcp_x', 'tcp_y', 'tcp_z')], 3, 3, 'm')}</small>"
    if op == "grip" and "jaw" in fb:
        status_html += f"<small>实测开度 {fb['jaw']:.2f}</small>"
    if op == "joint" and "q" in fb:
        status_html += f"<small>实测 {vector_label(fb['q'], 6, 2, 'rad')}</small>"
    if op == "pose_preview" and receipt.get("planned_native_steps") is not None:
        status_html += f"<small>规划 {receipt['planned_native_steps']} 步</small>"
    stage = row.get("stage")
    if stage is None or stage == "—":
        stage = (row.get("state") or {}).get("phase", "—")
    stage = str(stage)
    if row.get("next_stage") is not None:
        stage += f" → {row['next_stage']}"
    arm = {"left": "左", "right": "右"}.get(row.get("command_arm"), "—")
    arm_cell = f"<td>{arm}</td>" if has_arm else ""
    return (f"<tr><td>{escape(str(row['step']))}</td><td>{escape(stage)}</td>{arm_cell}<td>{escape(OP.get(op,op))}</td>"
            f"<td><code>{escape(command)}</code></td><td>{status_html}</td>"
            f"<td>{escape(str(row.get('native_steps','—')))}</td></tr>")


def render_steps(rows):
    if not rows:
        return ""
    has_arm = any(r.get("command_arm") for r in rows)
    arm_header = "<th>手臂</th>" if has_arm else ""
    return (f"<div class=scroll><table><thead><tr><th>步</th><th>状态</th>{arm_header}<th>模块输出</th><th>目标"
            "</th><th>执行回报</th><th>累计步</th></tr></thead><tbody>"
            + "".join(row_html(row, has_arm) for row in rows) + "</tbody></table></div>")


def render_attempt(run, index, group, strategy):
    transfer, version, rows, outcome = group
    visual = VISUAL.get(outcome.get("status"), "没有结束观测") if outcome else "没有结束观测"
    kind = "good" if outcome and outcome.get("status") == "placed" else "neutral"
    title = f"数字 {transfer['digit']} → {transfer['pad']} · {'左' if transfer['arm']=='left' else '右'}臂"
    label = f"v{version}" if version is not None else f"{strategy}版本未知"
    evidence = ""
    if outcome:
        prior_transfer = [d for d in sorted(run.glob("transfer-*-try-*")) if d.is_dir()]
        capture_dir = prior_transfer[index] if index < len(prior_transfer) else None
        links = []
        if capture_dir:
            links = [(camera_photo(capture_dir, cam), label) for cam, label in (
                ("cam_head", "头部相机"), ("cam_left_wrist", "左腕相机"), ("cam_right_wrist", "右腕相机"))]
            links.append((capture_dir / "head-marked.png", "带标记画面"))
        ref = " ".join(f'<a href="{relative_link(run,p)}">{label}</a>' for p, label in links if p and p.is_file())
        stop = OP.get(outcome["controller_stop"], outcome["controller_stop"])
        note = escape(outcome.get("note", ""))
        failures = [r['receipt'] for r in rows if r.get('receipt') and r['receipt'].get('status') not in OK]
        reason = str(failures[-1]['reason']) if failures and failures[-1].get('reason') else None
        cause = f'；{escape(REASON.get(reason, reason))}' if reason else ''
        last_jaw = next((r for r in reversed(rows) if r['operation'] == 'grip' and is_number(r.get('jaw'))), None)
        release = ""
        if last_jaw and last_jaw['jaw'] <= .2:
            release = ('已发出闭爪，之后没有松爪指令。' if strategy == "权重" else
                       '已发出闭爪，之后没有单独松爪指令。')
        evidence = (f'<p>模块停止方式：{escape(stop)}{cause}。{release} {ref}</p>'
                    f'<p class="muted">Astra 看图：{note}</p>')
        title += f" · 第 {outcome['try']} 次"
    return (f'<details class="attempt"><summary><b>{escape(title)}</b> · {escape(label)} · {len(rows)} 次模块决策 '
            f'<span class="tag {kind}">{escape(visual)}</span></summary>'
            f'<div class="attempt-body">{evidence}{render_steps(rows)}</div></details>')


def render_homing(run, rows, event):
    if not rows and not event:
        return ""
    home_path = run / "home.json"
    data = load(home_path, {})
    errors = []
    for arm, label in (("left", "左臂"), ("right", "右臂")):
        a = data.get("initial", {}).get(arm, {}).get("tcp")
        b = data.get("final", {}).get(arm, {}).get("tcp")
        if a is not None and b is not None:
            errors.append(f"{label} {1000 * math.dist(a, b):.1f} mm")
    stop = event.get("controller_stop") if event else None
    status = ("模块执行结束" if stop == "done" else OP.get(stop, stop or "未记录结束状态"))
    lines = [f"归位状态：{escape(status)}。"]
    if errors:
        lines.append(f"末端距初始位置：{'；'.join(errors)}。")
    if home_path.is_file():
        lines.append(f'<a href="{relative_link(run, home_path)}">归位原始记录</a>')
    return (f'<section><h2>机械臂归位</h2><p class="muted">{" ".join(lines)}</p>'
            f'<details class="attempt"><summary>展开 {len(rows)} 次归位决策</summary>'
            f'<div class="attempt-body">{render_steps(rows)}</div></details></section>')


def video_html(run, report, first):
    recording, sampled = report / "cam_head.mp4", report / "replay.mp4"
    if recording.is_file() and recording.stat().st_size > 0:
        path, figure_class = recording, "video video-recording"
        label = "头部相机实录"
        caption = "仿真运行中采样的头部相机画面。"
        poster = f' poster="{relative_link(run, first)}"' if first else ""
        manifest = load(report / "video-manifest.json")
        mapping = ' · <a href="video-manifest.json">帧与仿真步对应表</a>' if manifest else ""
        head = next((v for v in (manifest or {}).get("videos", []) if v.get("camera_id") == "cam_head"), {})
        frames, fps = head.get("frames"), head.get("fps")
        if frames == 1:
            caption = "本次录像只有开场一帧。"
        elif is_number(frames) and is_number(fps):
            caption = f"运行中采样 {frames} 帧，以 {fps:g} fps 回放；帧间的仿真步长可不同。"
    elif sampled.is_file() and sampled.stat().st_size > 0:
        path, figure_class = sampled, "video"
        label = "头部相机采样回放"
        caption = "真实相机采样帧，按仿真时间排列；结果处稍作停留。"
        poster = ""
        mapping = ""
    else:
        return ""
    return (f'<figure class="{figure_class}"><video aria-label="{label}"{poster} controls playsinline preload="metadata">'
            f'<source src="{path.name}" type="video/mp4"></video>'
            f'<figcaption class="muted">{caption} <a href="{path.name}">打开 MP4</a>{mapping}</figcaption></figure>')


def render(run):
    protocol = load(run / "protocol.json", {})
    is_code = protocol.get("policy") == "code"
    strategy = "代码" if is_code else "权重"
    eyebrow = "CODE SKILL" if is_code else "WSP"
    title = "让生成的代码搬数字" if is_code else "让权重搬数字"
    rules = ("动作与恢复规则由 Astra 生成的 Python 模块决定；摄像头、执行器与原生评分器固定。" if is_code else
             "动作与恢复规则由数值网络决定；摄像头、执行器与原生评分器固定。")
    score = load(run / "native-score.json")
    error = load(run / "runner-error.json") or load(run / "score-error.json")
    log = run / "events.jsonl"
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()] if log.is_file() else []
    groups = attempts(events)
    home = [g for g in groups if g[0]["digit"] == "归位" and g[0]["pad"] == "起点"]
    groups = [g for g in groups if g not in home]
    home_result = next((e for e in reversed(events) if e.get("event") == "home_result"), None)
    installs = [e for e in events if e.get("event") == "install"]
    plan = load(run / "semantic_plan.json", {})
    first, last = observed_frames(run)
    report = run / "report"
    report.mkdir(exist_ok=True)
    status, color, score_label = "未拿到原生评分", "neutral", "原生得分 —"
    if score:
        status = ("无效回合" if not score.get("valid_episode") else
                  "任务通过" if score.get("task_success") else "任务未通过")
        color = "good" if status == "任务通过" else "warn"
        value = score.get("normalized_score")
        if value is not None:
            score_label = f"原生得分 {100*value:.0f}/100"
    if error:
        status, color = "运行中断", "warn"
    terminal = score.get("episode_terminal", {}) if score else {}
    used = terminal.get("native_action_count")
    limit = terminal.get("native_step_limit", protocol.get("native_limit"))
    motion_label = f"{used}/{limit} 原生步" if used is not None and limit is not None else "原生步数未确认"
    revisions = max(0, len(installs) - 1)
    make_overview(report / "overview.png", first, last, status, f"{score_label} · {motion_label} · {revisions} 次{strategy}修订")
    goal = "".join(str(t["digit"]) for t in sorted(plan.get("transfers", []), key=lambda t: int(t["pad"][1:])))
    pose_label = f"布局 {protocol['layout']} · 种子 {protocol['seed']}" if "layout" in protocol and "seed" in protocol else "布局未知"
    last_checks = {outcome['transfer']['pad']: outcome for _, _, _, outcome in groups if outcome}
    placed = [str(t['transfer']['digit']) for _, t in sorted(last_checks.items()) if t['status'] == 'placed']
    placement_label = f"按每个目标最近一次看图，确认放上了：{'、'.join(placed)}。" if placed else (
        "各目标最近一次看图均未确认放上。" if last_checks else "暂无看图判断。")
    links = [
        f'<a href="{relative_link(run,p)}">{label}</a>'
        for p, label in ((first, "开场原图"), (last, "最后原图"), (log, "逐步原始记录"),
                         (run / "native-score.json", "评分器原始回报")) if p and p.is_file()
    ]
    history, old_stages = [], []
    for version in installs:
        n = version["version"]
        if is_code:
            raw = run / f"code-v{n:02}.py"
            ref_label = "查看生成的 Python"
            change = ""
        else:
            raw = run / f"policy-v{n:02}.json"
            ref_label = "查看数字权重"
            stages = load(raw, {}).get("stages", [])
            delta = sum(a != b for a, b in zip_longest(stages, old_stages))
            change = f" · {len(stages)} 个网络状态"
            if old_stages:
                change += f" · 相对上一版有 {delta} 个状态改变"
            old_stages = stages
        ref = f' · <a href="{relative_link(run,raw)}">{ref_label}</a>' if raw.is_file() else ""
        notes_path = run / f"skill-v{n:02}.md"
        if is_code and notes_path.is_file():
            ref += f' · <a href="{relative_link(run,notes_path)}">技能说明</a>'
        elapsed = f" · 生成用时 {version['seconds']:.1f} 秒" if 'seconds' in version else ""
        history.append(f'<div class="install"><b>v{n}</b>{change}{elapsed}{ref}'
                       f'<br>{escape(version.get("note",""))}</div>')
    attempts_html = "".join(render_attempt(run, i, a, strategy) for i, a in enumerate(groups))
    interrupted = f'<p class="run-error">{escape(error["type"])}：{escape(error["error"])}</p>' if error else ""
    video = video_html(run, report, first)
    if not video and (video_error := load(run / "video-error.json")):
        video = f'<p class="run-error">原生录像导出失败：{escape(video_error["error"])}</p>'
    homing = render_homing(run, [r for _, _, rows, _ in home for r in rows], home_result)
    history_html = ''.join(history) if history else f'<p>暂无已安装的{strategy}。</p>'
    result = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoboDojo 数字排列 · {strategy}试跑</title><style>{CSS}</style>
<main><header><p class="eyebrow">ROBODOJO SIM / {eyebrow}</p><h1>{title}</h1>
<p class="muted subtitle">{escape(pose_label)}{f' · Astra 识别并计划的目标：{escape(goal)}' if goal else ''}</p>
<div class="meta"><span class="tag {color}">{status}</span><span class="tag">{score_label}</span>
<span class="tag">{motion_label}</span><span class="tag">{len(installs)} 个{strategy}版本</span></div>{interrupted}</header>
{video}
<figure><img class="overview" src="overview.png" alt="RoboDojo 开场和最后记录的头部相机画面"></figure>
<p class="links subtitle">{''.join(links)}</p>
<section><h2>{strategy}改了什么</h2><p class="muted">{rules}</p>
{history_html}</section>
<section><h2>每次搬运</h2><p class="muted">{placement_label} 展开查看模块输出、实测状态与动作回报。</p>
{attempts_html if attempts_html else '<p>暂无搬运记录。</p>'}</section>
{homing}
</main></html>"""
    (report / "index.html").write_text(result, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error(f"not a run directory: {args.run_dir}")
    print(render(args.run_dir.resolve()) / "index.html")


if __name__ == "__main__":
    main()
