"""Inspect a recorded W from ink coordinates, independently of the agent.

The score is a shape screen, not a comparison with prescribed waypoints.
Coordinates are metres in the simulator's Cartesian frame (+y is up).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def points(value: object) -> np.ndarray:
    """Return finite planar observations, or an empty path."""
    try:
        xy = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.empty((0, 2))
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        return np.empty((0, 2))
    return xy


def length(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def ink_paths(strokes: Iterable[object]) -> list[np.ndarray]:
    """Treat stationary dots or movement shorter than 1 mm as no drawn stroke."""
    return [xy for s in strokes if length(xy := points(s)) >= 0.001]


def _sample_path(xy: np.ndarray, count: int = 96) -> np.ndarray:
    dist = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    keep = np.r_[True, np.diff(dist) > 1e-9]
    grid = np.linspace(0.0, dist[-1], count)
    return np.column_stack([np.interp(grid, dist[keep], dim[keep]) for dim in xy.T])


def _fit_legs(xy: np.ndarray) -> tuple[int, list[int], float]:
    """Find descending, ascending, descending, ascending legs in path order.

    Each leg must advance horizontally by at least 8% of the glyph width and
    change height by at least 22% of its height. Its directed vertical change
    must be at least 55% of its total vertical travel (allows curved corners).
    """
    width, height = np.ptp(xy, axis=0)
    if width < 1e-9 or height < 1e-9:
        return 0, [], 0.0
    x, y = xy.T
    n = len(xy)
    ydist = np.r_[0.0, np.abs(np.diff(y)).cumsum()]
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    vert = ydist[None, :] - ydist[:, None]
    forward = (dx >= 0.08 * width) & np.triu(np.ones((n, n), dtype=bool), k=1)
    cost = np.full(n, -np.inf)
    cost[0] = 0.0
    parents: list[np.ndarray] = []
    partial = 0
    for k, sign in enumerate((-1, 1, -1, 1), 1):
        directed = dy * sign
        legal = forward & (directed >= 0.22 * height) & (directed >= 0.55 * vert)
        clean = np.divide(directed, vert, out=np.zeros_like(directed), where=vert > 0.0)
        matrix = np.where(legal, cost[:, None] + clean, -np.inf)
        prev = np.argmax(matrix, axis=0)
        cost = matrix[prev, np.arange(n)]
        parents.append(prev)
        if np.isfinite(cost).any():
            partial = k
    if not np.isfinite(cost[-1]):
        # A partial prefix is useful for comparing failed attempts, but a full
        # fit must account for the actual last point of the entire stroke.
        return min(partial, 3), [], 0.0
    idx = [n - 1]
    for parent in reversed(parents):
        idx.append(int(parent[idx[-1]]))
    return 4, idx[::-1], float(cost[-1] / 4)


def assess(strokes: Iterable[object], min_width: float = 0.030, min_height: float = 0.018) -> dict:
    """A successful shape is a single, nontrivial stroke with four clear legs."""
    ink = ink_paths(strokes)
    if not ink:
        return dict(passed=False, score=0.0, ink_strokes=0, width_mm=0.0,
                    height_mm=0.0, horizontal_progress=0.0, matched_legs=0,
                    raised_ends=False, failed_checks=["no ink"], vertices=[])
    primary = max(ink, key=length)
    direction = "left-to-right" if primary[-1, 0] >= primary[0, 0] else "right-to-left"
    if direction == "right-to-left":
        primary = primary[::-1]
    width, height = np.ptp(primary, axis=0)
    progress = float((primary[-1, 0] - primary[0, 0])
                     / max(1e-9, np.abs(np.diff(primary[:, 0])).sum()))
    extent = float((primary[-1, 0] - primary[0, 0]) / max(1e-9, width))
    sampled = _sample_path(primary)
    legs, idx, consistency = _fit_legs(sampled)
    raised = bool(height > 0.0 and min(primary[0, 1], primary[-1, 1])
                  >= primary[:, 1].min() + 0.50 * height)
    checks = {
        "more than one ink stroke": len(ink) == 1,
        "width below minimum": width >= min_width,
        "height below minimum": height >= min_height,
        "too much horizontal reversal": progress >= 0.68 and extent >= 0.75,
        "fewer than four clear alternating legs": legs == 4,
        "stroke ends not both raised": raised,
    }
    raw_score = (40 * legs / 4 + 10 * consistency + 10 * min(1.0, width / min_width)
                 + 10 * min(1.0, height / min_height) + 15 * min(progress, extent)
                 + 10 * raised + 5 * (len(ink) == 1))
    return dict(
        passed=bool(all(checks.values())), score=round(raw_score, 2),
        ink_strokes=len(ink), width_mm=round(float(width) * 1000, 2),
        height_mm=round(float(height) * 1000, 2),
        horizontal_progress=round(progress, 3), matched_legs=legs, raised_ends=raised,
        direction=direction, leg_monotonicity=round(consistency, 3),
        failed_checks=[why for why, ok in checks.items() if not ok],
        vertices=sampled[idx].tolist(),
    )


def load_log(path: Path) -> list[dict]:
    turns = []
    for lineno, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        if not isinstance(row, dict) or not isinstance(row.get("result"), dict):
            raise ValueError(f"{path}:{lineno}: expected a turn object with a result")
        turns.append(row)
    if not turns:
        raise ValueError(f"{path}: empty run")
    return turns


def _attempt_key(turn: dict) -> object:
    return turn["result"].get("attempt", turn.get("attempt", 1))


def completed_canvases(turns: list[dict]) -> list[dict]:
    """Use the final ink snapshot of each attempt, not a flattering prefix."""
    canvases: list[dict] = []
    last_ink = None
    previous = None
    for turn in turns:
        attempt = _attempt_key(turn)
        strokes = turn["result"].get("strokes")
        if previous is not None and attempt != previous:
            if last_ink is not None:
                canvases.append(last_ink)
            last_ink = None
        elif last_ink is not None and isinstance(strokes, list) and not ink_paths(strokes):
            # Some runners clear the canvas without changing attempt numbers.
            # Only treat a successful reset/clear operation as such a boundary.
            op = str((turn.get("request") or {}).get("op", ""))
            if turn["result"].get("ok", True) and ("reset" in op or "clear" in op):
                canvases.append(last_ink)
                last_ink = None
        if isinstance(strokes, list) and ink_paths(strokes):
            last_ink = turn
        previous = attempt
    if last_ink is not None:
        canvases.append(last_ink)
    return canvases


def score_log(turns: list[dict], min_width: float = 0.030, min_height: float = 0.018) -> dict:
    scored = []
    for turn in completed_canvases(turns):
        scored.append(dict(turn=turn.get("turn"), attempt=_attempt_key(turn),
                           **assess(turn["result"]["strokes"], min_width, min_height)))
    best = max(scored, key=lambda a: (a["passed"], a["score"])) if scored else None
    return dict(min_width_mm=min_width * 1000, min_height_mm=min_height * 1000,
                best=best, last=scored[-1] if scored else None, attempts=scored)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-width", type=float, default=0.030, metavar="M")
    parser.add_argument("--min-height", type=float, default=0.018, metavar="M")
    args = parser.parse_args()
    print(json.dumps(score_log(load_log(args.log), args.min_width, args.min_height),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
