"""Summarize the matched arms without choosing or changing any trial."""
import argparse
import json
from pathlib import Path
from statistics import median


def upright(row):
    m = row["metrics"]
    return m["completed_horizon"] and not m["physically_down"]


def shortest_angle(x):
    return abs((x + 180) % 360 - 180)


def paired(data):
    cases, nominal = data["push"], data["nominal"]
    active = [r for r in cases if r["trigger_step"] is not None]
    arms = ("base", "wsp", "code")
    out = {
        "push_cases": len(cases), "triggered": len(active),
        "triggers_before_push": [r["case"] for r in active if r["trigger_step"] < r["scenario"]["push_step"]],
        "nominal_cases": len(nominal),
        "nominal_triggered": [r["case"] for r in nominal if r["trigger_step"] is not None],
        "all_prefix_identical": all(r["prefix_identical"] for r in cases + nominal),
        "arms": {},
    }
    for k in arms:
        rows = [r["arms"][k] for r in cases]
        a = [r["arms"][k]["activity_stats"] for r in active]
        out["arms"][k] = {
            "recovered": [r["case"] for r in cases if r["arms"][k]["metrics"]["recovered_and_finished"]],
            "physically_down": [r["case"] for r in cases if r["arms"][k]["metrics"]["physically_down"]],
            "upright": sum(upright(r) for r in rows),
            "nominal_upright": sum(upright(r["arms"][k]) for r in nominal),
            "active_median_rms_rad": median(x["rms_l2_rad"] for x in a) if a else None,
            "active_median_peak_rad": median(x["peak_l2_rad"] for x in a) if a else None,
            "active_cap_hits": sum(x["cap_hits"] for x in a),
            "active_actor_ticks": sum(x["ticks"] for x in a),
        }
    out["larger_rms_cases"] = [r["case"] for r in active
        if r["arms"]["wsp"]["activity_stats"]["rms_l2_rad"] > r["arms"]["code"]["activity_stats"]["rms_l2_rad"]]
    yaw = []
    for r in active:
        if not all(upright(r["arms"][k]) for k in arms):
            continue
        angles = {k:r["arms"][k]["metrics"]["matched_recovery"].get("yaw_error_deg") for k in arms}
        if any(v is None for v in angles.values()):
            continue
        yaw.append({"case":r["case"], "error_deg":angles,
                    "wsp_shift_deg":shortest_angle(angles["wsp"]-angles["base"]),
                    "code_shift_deg":shortest_angle(angles["code"]-angles["base"])})
    out["yaw_active_all_upright"] = yaw
    out["larger_wsp_yaw_shift_cases"] = [r["case"] for r in yaw if r["wsp_shift_deg"] > r["code_shift_deg"]]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("summary",type=Path)
    args=p.parse_args()
    print(json.dumps(paired(json.loads(args.summary.read_text())), indent=2))
