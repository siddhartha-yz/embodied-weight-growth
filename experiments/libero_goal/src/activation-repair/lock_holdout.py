"""Select the first supported, non-colliding official resets per preregistered group."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "multitask-repair")]
from layouts import BowlOffsetEnv

CAPS = {"bowl": 4, "bottle-crossed": 2, "bottle-native": 2}

def group(case):
    if case["goal"] == "bowl":
        return "bowl"
    return "bottle-crossed" if case.get("prompt_mode") == "historical_bottle" else "bottle-native"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidates", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("evidence", type=Path)
    args = p.parse_args()
    cases = json.loads(args.candidates.read_text())
    selected = []
    counts = dict.fromkeys(CAPS, 0)
    tested = []
    with BowlOffsetEnv(pair="objects", render=False) as env:
        a, b = [env._init_states[k] for k in ("bowl", "bottle")]
        overlap = int(np.equal(a[:, None, :], b[None, :, :]).all(axis=-1).any(axis=0).sum())
        if overlap:
            raise ValueError("Official initialization banks overlap")
        for case in cases:
            g = group(case)
            if counts[g] == CAPS[g]:
                continue
            result = dict(id=case["id"], group=g, valid=False)
            try:
                env.reset(goal=case["goal"], init_index=case["init_index"], seed=case["seed"],
                          init_source=case["init_source"], layout=case["layout"])
                if env.status()["goal_achieved"]:
                    raise ValueError("Already achieved target at reset")
                result["valid"] = True
                selected.append(case)
                counts[g] += 1
            except (RuntimeError, ValueError) as ex:
                result["reason"] = str(ex)
            tested.append(result)
    evidence = {"official_bank_sizes": {"bowl": len(a), "bottle": len(b)},
                "identical_raw_rows_between_banks": overlap, "tested": tested,
                "accepted_counts": counts}
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    if counts != CAPS:
        raise RuntimeError(f"Insufficient geometry-valid holdout rows: {counts}")
    args.output.write_text(json.dumps(selected, indent=2) + "\n")

if __name__ == "__main__":
    main()
