"""Fit the recorded code v8 as an unchanged-runtime sparse tanh policy."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from walk_policy import FEATURES, compile_policy, inputs
from walk_server import Session


USE = np.array([2, 3, 5, *range(6, 18), 20, 21, 22])
STATE_SIGMA = np.array([0.07, 0.20, 0.35] + [0.07] * 6 + [0.38] * 6)


def save(path, data):
    path.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def teacher_torques(x):
    """Vectorize this one fixed teacher, checked against the original interpreter."""
    p, vx, dp = x[:, 2], x[:, 3], x[:, 5]
    q, dq, t = x[:, 6:12], x[:, 12:18], x[:, 22]
    s = np.sin(7 * t)
    a = np.minimum(1, t / .5)
    b = -.7 * (p + .35 * a) - .025 * dp
    motor = np.zeros_like(q)
    for j, swinging, phase in ((0, s > 0, s), (3, s <= 0, -s)):
        h_self = .7 * (np.minimum(0, p + .3) - q[:, j]) - .018 * dq[:, j]
        other = (j + 3) % 6
        h_other = .7 * (np.minimum(0, p + .3) - q[:, other]) - .018 * dq[:, other]
        stance = (.4 * np.sin(q[:, j] - p)
                  + .4 * np.sin(q[:, j] - p + q[:, j + 1]) - .3 * vx + .1)
        swing = .3 * (p - q[:, j] - q[:, j + 1] - q[:, j + 2])
        motor[:, j] = np.where(swinging, h_self, 2 * b - h_other)
        motor[:, j + 1] = .8 * (-.08 - .9 * np.maximum(0, phase) * a - q[:, j + 1]) - .022 * dq[:, j + 1]
        motor[:, j + 2] = np.where(swinging, swing, stance) - .01 * dq[:, j + 2]
    return np.clip(motor, -1, 1)


def states_array(states):
    return np.array([[d[k] for k in FEATURES] for d in (inputs(s, .5) for s in states)])


def augment(bases, rng, repeats, sigma):
    data = np.repeat(bases, repeats, axis=0)
    data[:, USE[:15]] += rng.normal(size=(len(data), 15)) * STATE_SIGMA * sigma
    return np.concatenate((bases, data))


def build_network(center, scale, width):
    net = torch.nn.Sequential(
        torch.nn.Linear(len(USE), width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), torch.nn.Tanh(),
        torch.nn.Linear(width, 6),
    )
    with torch.no_grad():
        # Seed a representation for the teacher's alternating mode changes.
        # Only t is an input: these gates are ordinary first-layer tanh units.
        times = np.arange(12) * math.pi / 7
        for i, t in enumerate(times[times <= 5.01]):
            net[0].weight[i, :].zero_()
            net[0].weight[i, -1] = 38.0 * scale[-1]
            net[0].bias[i] = 38.0 * (center[-1] - t)
    return net


def serialize(net, center, scale):
    layers = []
    for depth, module in enumerate(net[::2]):
        w = module.weight.detach().numpy().astype(float)
        b = module.bias.detach().numpy().astype(float)
        if depth == 0:
            w = w / scale
            b -= w @ center
        indexes = USE if depth == 0 else range(w.shape[1])
        layers.append({"rows": [
            {"terms": [{"feature": int(i), "coef": float(c)} for i, c in zip(indexes, row) if c],
             "bias": float(bias)} for row, bias in zip(w, b)
        ]})
    return {"layers": layers}


def rollout(spec, scene, seed):
    session = Session("weights", scene=scene, seed=seed, frozen_spec=spec)
    initial = session.snapshot()
    result = session.request({"op": "run", "ticks": 250})
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return {"initial": initial, "result": result, "scene": scene, "seed": seed}


def score(r):
    w = r["result"]["walk"]
    return (r["result"]["success"] and w["fall_reason"] is None,
            w["alternating_steps"], w["fall_reason"] is None, w["elapsed_s"], w["distance_m"])


def phase_probe(policy, states):
    """Ankle torque responses to speed and ankle-angle changes in each phase."""
    derivatives = {name: [] for name in ("stance_vx", "swing_vx", "stance_ankle_q", "swing_ankle_q")}
    delta = .02
    for state in states:
        phase = math.sin(7 * state["t"])
        if abs(phase) <= .35:
            continue
        u_v = [policy({**state, "vx": state["vx"] + d}, .5) for d in (-delta, delta)]
        for ankle, swinging in ((2, phase > 0), (5, phase <= 0)):
            label = "swing" if swinging else "stance"
            u_q = []
            for d in (-delta, delta):
                q = state["q"].copy()
                q[ankle] += d
                u_q.append(policy({**state, "q": q}, .5)[ankle])
            derivatives[label + "_vx"].append((u_v[1][ankle] - u_v[0][ankle]) / (2 * delta))
            derivatives[label + "_ankle_q"].append((u_q[1] - u_q[0]) / (2 * delta))
    return {key: {"median": float(np.median(d)), "n": len(d)} for key, d in derivatives.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--nominal-trace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--width", type=int, choices=(16, 20, 24), default=24)
    ap.add_argument("--seed", type=int, default=1118)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.teacher.read_text())["spec"]
    teacher = compile_policy("code", source)
    trace = json.loads(args.nominal_trace.read_text())
    world_seed = trace["seed"]
    if trace["scene"] != "train" or trace["artifact"]["spec"] != source:
        raise ValueError("Expected the nominal recorded rollout of the provided teacher")
    raw = trace["result"]["trace"][:-1]
    bases = states_array(raw)
    np.testing.assert_allclose(teacher_torques(bases), np.array([teacher(r, .5) for r in raw]),
                               atol=1e-12, rtol=1e-12)
    # Extra weight at first contact, with independent physical sensor changes
    # throughout. The transfer pitch is not consulted until a candidate is frozen.
    anchor = np.concatenate((bases, bases[:30], bases[:30]))
    train = np.concatenate((augment(anchor, rng, 25, .75), augment(anchor, rng, 10, 1.6)))
    val = augment(bases, rng, 4, 1.0)
    center, scale = train[:, USE].mean(axis=0), np.maximum(train[:, USE].std(axis=0), .05)
    normalize = lambda arr: torch.as_tensor((arr[:, USE] - center) / scale, dtype=torch.float32)
    tx = normalize(train)
    ty = torch.as_tensor(teacher_torques(train), dtype=torch.float32)
    vx = normalize(val)
    vy = teacher_torques(val)
    net = build_network(center, scale, args.width)
    opt = torch.optim.Adam(net.parameters(), lr=.003)
    channels = torch.tensor([1., 1., 2., 1., 1., 2.])
    trial_log = []
    best = None
    best_record = None
    for i in range(1, args.steps + 1):
        idx = torch.randint(len(tx), (512,))
        pred = net(tx[idx])
        loss = ((pred.clamp(-1, 1) - ty[idx]).square() * channels).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 400 and i != args.steps:
            continue
        with torch.no_grad():
            error = net(vx).clamp(-1, 1).numpy() - vy
        spec = serialize(net, center, scale)
        nominal = rollout(spec, "train", world_seed)
        record = {"step": i, "validation_mae": float(np.abs(error).mean()),
                  "validation_rmse": float(np.mean(error ** 2) ** .5),
                  "train_examples": len(tx), "score": score(nominal),
                  "walk": nominal["result"]["walk"]}
        trial_log.append(record)
        if best is None or score(nominal) > score(best):
            best, best_record = nominal, record
            save(args.out / "candidate.json", spec)
            save(args.out / "nominal.json", nominal)
        print(f"step {i}: validation MAE={record['validation_mae']:.5f}, "
              f"nominal={record['score']}", flush=True)
        save(args.out / "fit.json", {
            "method": "supervised teacher-action regression; nominal simulator corrections only",
            "distillation_seed": args.seed, "simulator_seed": world_seed,
            "architecture": [25, args.width, args.width, 6],
            "nominal_states": len(bases), "baseline_augmented_examples": len(train),
            "perturbation_sigma_rad_or_si": STATE_SIGMA.tolist(), "validation_examples": len(val),
            "selected": best_record, "checkpoints": trial_log,
        })
        if record["score"][0]:
            break
        if i >= 800 and i % 800 == 0:
            visited = states_array(nominal["result"]["trace"][:-1])
            if len(visited):
                extras = augment(visited, rng, 6, .33)
                tx = torch.cat((tx, normalize(extras)))
                ty = torch.cat((ty, torch.as_tensor(teacher_torques(extras), dtype=torch.float32)))
        for group in opt.param_groups:
            group["lr"] = max(.00035, .003 * math.exp(-i / 4800))
    selected = json.loads((args.out / "candidate.json").read_text())
    holdout = rollout(selected, "transfer", world_seed)
    save(args.out / "transfer.json", holdout)
    save(args.out / "phase-response.json", {
        "teacher": phase_probe(teacher, raw), "student": phase_probe(compile_policy("weights", selected), raw),
    })
    print(f"frozen transfer={score(holdout)}", flush=True)


if __name__ == "__main__":
    main()
