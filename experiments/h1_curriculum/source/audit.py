"""Audit or reconstruct the published numeric record without running MuJoCo."""
import argparse
import gzip
import json
from pathlib import Path

from adaptive_probe import validate_plan
from weight_edit import apply_proposal
from weight_policy import Policy, read_json

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
RESULTS = ARCHIVE / "results"


def flattened(checkpoint):
    return {(layer, row, source): value
            for layer in ("hidden", "outputs")
            for row, terms in checkpoint[layer].items()
            for source, value in [("bias", terms.get("bias", 0)), *terms.get("weights", {}).items()]
            if value != 0}


def changed_keys(before, after):
    return {key for key in before.keys() | after.keys() if before.get(key, 0) != after.get(key, 0)}


def audit():
    meta = read_json(HERE / "plant-describe.json")
    sources, actions = meta["feature_names"], meta["action_names"]
    frozen = read_json(RESULTS / "checkpoints.json")
    weights = {"zero": read_json(ARCHIVE / frozen["zero"])}
    edits = {}
    files = ("initial-rounds.jsonl", "fixed-rounds.jsonl", "adaptive-rounds.jsonl")
    expected = (29, 30, 450)
    for name, count in zip(files, expected, strict=True):
        rows = [json.loads(line) for line in (RESULTS / name).read_text().splitlines()]
        assert len(rows) == count, name
        for i, row in enumerate(rows, 1):
            label, parent = row["id"], row["parent"]
            assert label not in weights
            if name != files[0]:
                lane = "adaptive" if name == files[-1] else "fixed"
                assert label == f"{lane}/r{i:02d}"
                assert parent == (f"{lane}/r{i-1:02d}" if i > 1 else "standing/r05")
            weights[label], changed = apply_proposal(weights[parent], row["proposal"], sources, actions)
            assert changed == row["changed_coefficients"], label
            edits[label] = row
            if "plan" in row:
                plan = validate_plan(row["plan"])
                assert [t["seed"] for t in plan["trials"]] == [c["seed"] for c in row["development"]], label
    for label, path in frozen.items():
        assert read_json(ARCHIVE / path) == weights[label], label

    repair = read_json(RESULTS / "plan-repair.json")
    for attempt in repair["old_runner"]["attempts"]:
        try:
            validate_plan(attempt["plan"])
        except ValueError as error:
            assert str(error) == attempt["error"].removeprefix("ValueError: ")
        else:
            raise AssertionError("Former invalid plan passed")
    repaired = repair["revised_runner"]
    row = edits[repaired["accepted_revision"]]
    assert row["plan_retries"] == 1 and row["proposal"]["rows"] == repaired["locked_weight_rows"]
    assert repaired["repair_response"] == {"plan": row["plan"]} and row["parent"] == repaired["parent"]
    try:
        validate_plan(repaired["initial_plan"])
    except ValueError:
        pass
    else:
        raise AssertionError("r345 should require a plan-only repair")

    snapshots = [weights[f"adaptive/r{i}"] for i in range(150, 451)]
    params = [flattened(w) for w in snapshots]
    touched = [changed_keys(a, b) for a, b in zip(params, params[1:])]
    print(f"Checked {sum(expected)} accepted edits and {len(frozen)} frozen weight files.")
    print(f"r150–r450: {len(set.union(*touched))} distinct coefficients; "
          f"{sum(map(len, touched))} edits including repeated coefficients; "
          f"{len(changed_keys(params[0], params[-1]))} endpoint differences.")
    hidden = {tuple(sorted(w["hidden"])) for w in snapshots}
    print("Active hidden units across r150–r450:", sorted(hidden))
    return weights, meta


def compare_ankles(weights, meta):
    """Evaluate all three policies on the same stored r150 states."""
    import numpy as np

    revs = ("r150", "r270", "r450")
    names = meta["action_names"]
    ankles = [names.index(side + "_ankle") for side in ("left", "right")]
    policies = {rev: Policy.compile(weights["adaptive/" + rev], meta["feature_names"], names) for rev in revs}
    for seed in (2801, 2803):
        with gzip.open(RESULTS / f"r150-walk-{seed}.trace.json.gz", "rt") as stream:
            trace = json.load(stream)
        features = []
        for state, next_state in zip(trace, trace[1:]):
            feet = np.asarray(state["foot_xyz_m"]).copy()
            feet[:, :2] -= np.asarray(state["qpos"][:2])
            features.append(np.r_[state["qpos"][2:], state["qvel"], next_state["target_vx"],
                                  state["root_rpy"], np.asarray(state["foot_contact"], dtype=float), feet.ravel()])
        record = np.asarray([state["action"] for state in trace[1:]])
        fractions = []
        for rev, policy in policies.items():
            predicted = np.asarray([policy.act(state) for state in features])
            if rev == "r150":
                np.testing.assert_allclose(predicted, record, rtol=0, atol=1e-10)
            both = (np.abs(predicted[:, ankles]) >= 1.0 - 1e-12).all(axis=1)
            fractions.append(f"{rev}={100*both.mean():.1f}%")
        print(f"r150 states, seed {seed}: both ankle targets saturated; " + ", ".join(fractions))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--export", nargs=2, metavar=("CHECKPOINT_ID", "OUTPUT_JSON"))
    cli.add_argument("--ankles", action="store_true", help="reproduce offline ankle saturation using NumPy")
    args = cli.parse_args()
    weights, meta = audit()
    if args.export:
        label, filename = args.export
        if label not in weights:
            cli.error(f"Unknown checkpoint {label}")
        path = Path(filename)
        with path.open("x") as stream:
            json.dump(weights[label], stream, indent=2)
            stream.write("\n")
        print(f"Reconstructed {label} → {path}")
    if args.ankles:
        compare_ankles(weights, meta)


if __name__ == "__main__":
    main()
