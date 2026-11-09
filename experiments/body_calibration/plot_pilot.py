"""Render end-effector trajectories from completed lab evaluation JSON."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("linear", type=Path)
    p.add_argument("backlash", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    reports = [json.loads(x.read_text()) for x in (args.linear, args.backlash)]
    labels = ["Reach A", "Reach B", "Trace a loop"]
    colors = {"selected_code": "#15736e", "wsp": "#c15a30"}
    names = {"selected_code": "Numerical calibration", "wsp": "Astra weight write"}
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.1))
    fig.patch.set_facecolor("#fbfbf8")
    fig.subplots_adjust(left=.095, right=.96, top=.79, bottom=.12, hspace=.53, wspace=.34)
    for row, (report, axis_row) in enumerate(zip(reports, axes)):
        controllers = report["controllers"]
        for col, ax in enumerate(axis_row):
            ax.set_facecolor("#fbfbf8")
            trial = controllers["selected_code"]["trials"][col]
            target = np.asarray(trial["trajectory"]["target_tip"]) * 100
            ax.plot(*target.T, lw=2.8, color="#343f49", alpha=.48, label="Reference", zorder=1)
            ax.scatter(*target[0], marker="o", s=13, color="#343f49", zorder=4)
            for key, color in colors.items():
                tr = controllers[key]["trials"][col]
                actual = np.asarray(tr["trajectory"]["actual_tip"]) * 100
                ax.plot(*actual.T, lw=1.45, color=color, label=names[key],
                        linestyle="--" if key == "wsp" else "-", zorder=3)
            if row == 0:
                ax.set_title(labels[col], loc="left", color="#263645", fontsize=10, weight="bold")
            values = [1000 * controllers[k]["trials"][col]["motion_tip_rmse_m"] for k in colors]
            ax.set_xlabel(f"Motion RMSE: {values[0]:.1f} / {values[1]:.1f} mm", fontsize=8, labelpad=4)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(color="#e4e7e6", linewidth=.6)
            ax.tick_params(labelsize=7, colors="#42515b", length=2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#c9d0ce")
            if col == 0:
                label = "Linear mix" if row == 0 else "With backlash"
                ax.set_ylabel(f"{label}\nY position (cm)", fontsize=9)
    fig.text(.095, .93, "One arm, three unseen motions", fontsize=20, weight="bold", color="#223441")
    fig.text(.095, .885, "Same calibration traces and trajectory tracker; only the frozen actuator model changes.",
             fontsize=10, color="#4e5a5c")
    lines, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(lines, legends, loc="upper left", bbox_to_anchor=(.09, .864), ncol=3,
               frameon=False, fontsize=8.5, borderaxespad=0)
    fig.text(.095, .041, "MuJoCo pilot  ·  body seed 271  ·  horizontal coordinates in cm",
             fontsize=8, color="#556563")
    fig.savefig(args.output, dpi=170, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
