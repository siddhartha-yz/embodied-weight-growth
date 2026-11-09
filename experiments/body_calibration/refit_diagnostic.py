"""Post-hoc control: fit the nonzero terms Astra chose, using the same probes."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_pilot import Adapter, calibration_score, design, samples, tasks, track
from plant import DT, Plant, tip_from_q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--condition", choices=("linear", "backlash"), required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    a = p.parse_args()
    data = samples(json.loads(a.records.read_text()), DT)
    model = Adapter.from_dict(json.loads(a.adapter.read_text()))
    x = design(data, 2, model.deadzone)
    coeff = np.concatenate((
        model.mix.T, model.bias[None, :], model.state.T, model.history.T, model.history2.T,
    ), axis=0)
    refit = np.zeros_like(coeff)
    for j in range(2):
        use = coeff[:, j] != 0
        sub = x[:, use]
        scale = np.maximum(np.sqrt(np.mean(sub**2, axis=0)), 0.05)
        z = sub / scale
        reg = 0.004 * len(x) * np.eye(z.shape[1])
        for i, feature in enumerate(np.flatnonzero(use)):
            if feature == 2:
                reg[i, i] = 0.0
        refit[use, j] = np.linalg.solve(z.T @ z + reg, z.T @ data[:, 10 + j]) / scale
    fitted = Adapter.from_dict(dict(
        mix=refit[:2].T, bias=refit[2], state=refit[3:7].T, history=refit[7:9].T,
        history2=refit[9:11].T, deadzone=model.deadzone,
    ))
    plant = Plant(a.seed, a.condition)
    trials = [track(plant, fitted, t, DT) for t in tasks(DT, tip_from_q)]
    print(json.dumps({
        "name": "posthoc_refit_astra_support",
        "seed": a.seed,
        "condition": a.condition,
        "n_success": sum(t["success"] for t in trials),
        "mean_motion_tip_rmse_m": float(np.mean([t["motion_tip_rmse_m"] for t in trials])),
        "mean_final_hold_tip_error_m": float(np.mean([t["final_hold_tip_error_m"] for t in trials])),
        "probe_qdd_rmse_rad_s2": calibration_score(fitted, data),
        "frozen_adapter": fitted.as_dict(),
        "trials": trials,
    }, allow_nan=False))


if __name__ == "__main__":
    main()
