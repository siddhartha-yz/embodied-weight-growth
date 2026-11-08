"""Replay recorded decisions in MuJoCo and verify the resulting observations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from score_w import load_log, score_log
from w_server import Session


def check(actual, expected, path):
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        for key, value in expected.items():
            check(actual[key], value, f"{path}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        for i, value in enumerate(expected):
            check(actual[i], value, f"{path}[{i}]")
    elif type(actual) in (int, float) and type(expected) in (int, float):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError(f"{path}: expected {expected}, got {actual}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path}: structure or value differs from the recorded result")


def verify_run(directory):
    def read(filename):
        return json.loads((directory / filename).read_text(encoding="utf-8"))

    protocol = read("protocol.json")
    session = Session(protocol["arm"], seed=protocol["seed"], condition=protocol["condition"])
    check(session.snapshot(), read("initial.json"), f"{directory.name}.initial")
    replayed = []
    for event in load_log(directory / "events.jsonl"):
        result = session.request(event["request"])
        check(result, event["result"], f"{directory.name}.turn{event['turn']}")
        replayed.append({**event, "result": result})
    check(replayed[-1]["result"], read("final.json")["state"], f"{directory.name}.final")
    score = score_log(replayed)
    check(score, read("shape.json"), f"{directory.name}.shape")
    ticks = replayed[-1]["result"]["meters"]["ticks_used"]
    print(f"{directory.name}: {len(replayed)} decisions, {ticks} ticks; recorded observations and score matched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="+", help="one or more recorded arm directories")
    for run in parser.parse_args().run:
        verify_run(run)
