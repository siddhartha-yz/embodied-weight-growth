"""Plot frozen Walker2d checkpoint evaluations from the lab's results.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARMS = {
    "bp": ("直接 BP", "#24698c", "o"),
    "astra": ("Astra 调度 + BP", "#bb6231", "D"),
}
INK = "#203243"
MUTED = "#536779"
GRID = "#e5ebef"
PAPER = "#fafbfd"


def draw(result, output):
    arms = result["arms"]
    points = {
        key: [(point["trial_budget_used"], point["pass"], point["n"]) for point in arms[key]["curve"]]
        for key in ARMS
    }
    ticks = sorted({x for curve in points.values() for x, _, _ in curve})
    counts = {n for curve in points.values() for _, _, n in curve}
    if len(counts) != 1:
        raise ValueError("a pass-count axis requires the same number of audit starts per checkpoint")
    n = counts.pop()
    old = {key: arms[key]["return_to_old_ground"] for key in ARMS}
    old_counts = {group["n"] for group in old.values()}
    if len(old_counts) != 1:
        raise ValueError("old-ground counts require the same number of audit starts per group")
    n_old = old_counts.pop()

    plt.rcParams.update({
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "path",
    })
    fig = plt.figure(figsize=(12.0, 5.35), facecolor=PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.72, 1], left=.073, right=.956, bottom=.203, top=.695, wspace=.36)
    learn, retain = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    fig.text(.073, .928, "Walker2d  ·  在线适应对照", size=11.5, color=MUTED, weight="normal")
    fig.text(.073, .846, "新地面学得多快，回到旧地面还能不能走", size=22, weight="bold")
    fig.text(.073, .785, f"冻结检查点复测：新地面每点 {n} 个未见起姿；旧地面比较学习前后", size=11, color=MUTED)
    learn.set_title("01   新地面通过数", loc="left", pad=13, fontsize=14, weight="bold")
    retain.set_title("02   换回原地面", loc="left", pad=13, fontsize=14, weight="bold")

    for ax in (learn, retain):
        ax.set_facecolor("white")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(length=0, labelsize=10, pad=7)

    if n == 12:
        learn.axhline(8, color="#93a3af", linewidth=1.15, linestyle=(0, (4, 4)), zorder=1)
    learn.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    learn.set_axisbelow(True)
    span = max(1, ticks[-1] - ticks[0])
    learn.set(xlim=(ticks[0] - span * .035, ticks[-1] + span * .094), ylim=(-n*.05, n*1.065))
    learn.set_xlabel("已用在线试跑次数", labelpad=8, fontsize=10.5)
    learn.set_ylabel(f"通过数（满分 {n}）", labelpad=8, fontsize=10.5)
    learn.set_xticks(ticks)
    learn.set_yticks(range(0, n + 1, max(1, n // 6)))
    last_close = abs(points["bp"][-1][1] - points["astra"][-1][1]) < n*.075
    for i, (key, (label, color, marker)) in enumerate(ARMS.items()):
        x, y, _ = zip(*points[key])
        learn.plot(x, y, label=label, color=color, linewidth=2, marker=marker, markersize=6.5,
                   markeredgewidth=1.5, markerfacecolor=color if key == "bp" else "white",
                   linestyle="-" if key == "bp" else (0, (5, 2)), zorder=3+i, clip_on=False)
        y_offset = (7 if key == "bp" else -7) if last_close else 0
        learn.annotate(str(y[-1]), (x[-1], y[-1]), xytext=(8, y_offset), textcoords="offset points",
                       va="center", color=color, weight="bold", fontsize=10, clip_on=False)
    handles, labels = learn.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.073, .087), ncol=2,
                        frameon=False, borderaxespad=0, fontsize=10, handlelength=2.45, columnspacing=1.55)
    for text in legend.get_texts():
        text.set_color(INK)
    if n == 12:
        fig.text(.365, .062, "虚线为 8/12；连续两次达到才算通过适应判据。", color=MUTED, size=9.5)

    retain.set_xlim(0, n_old*1.18)
    retain.set_ylim(-.51, 3.47)
    retain.set_yticks([])
    retain.set_xticks([0, n_old])
    retain.xaxis.set_ticks_position("bottom")
    retain.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    retain.set_axisbelow(True)
    for i, (key, (label, color, _)) in enumerate(ARMS.items()):
        y = 2.76 - i * 1.75
        retain.text(0, y+.43, label, size=10.5, weight="bold")
        for yj, stat, bar_color in ((y, "before", "#b4c2ce"), (y-.46, "after", color)):
            value = old[key][stat]
            retain.barh(yj, n_old, height=.275, color="#f0f3f6", zorder=1)
            retain.barh(yj, value, height=.275, color=bar_color, zorder=2)
            retain.text(value+n_old*.032, yj, f"{value}/{n_old}", va="center", ha="left", fontsize=10,
                        weight="bold" if stat == "after" else "normal", color=INK, clip_on=False)
    retain.text(0, -.25, "灰色：学习前　彩色：学习后", color=MUTED, size=9.5)

    fig.savefig(output, dpi=165, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.results.open(encoding="utf-8") as handle:
        draw(json.load(handle), args.output)


if __name__ == "__main__":
    main()
