#!/usr/bin/env python3
"""Write the K=3 recenter vs fixed_ref score summary from final eval.csv files."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path("/raid/ext_csh/MPI_store/recenter_fixed_ref_k3/full")
ENVS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-expert-v2",
)
TAUS = (0.4, 2.5, 10.0)
SEEDS = (0, 1, 2, 3)
NAME = re.compile(
    r"^(?P<env>.+)_tau(?P<tau>[0-9.]+)_shared_(?P<branch>recenter|fixed_ref)3_seed(?P<seed>\d+)$"
)
KST = timezone(timedelta(hours=9))


def final_row(path: Path) -> dict | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    last = None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            last = row
    if last is None or str(last.get("step")) != "1000000":
        return None
    return last


def num(row: dict | None, key: str) -> float | None:
    if row is None or row.get(key) in (None, ""):
        return None
    try:
        return float(row[key])
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-0") and float(text) == 0 else text


def collect() -> dict:
    scores = {}
    if not ROOT.is_dir():
        return scores
    for run in ROOT.iterdir():
        match = NAME.match(run.name)
        if match is None:
            continue
        row = final_row(run / "eval.csv")
        if row is None:
            continue
        key = (
            match["env"],
            float(match["tau"]),
            match["branch"],
            int(match["seed"]),
        )
        scores[key] = (num(row, "d4rl_pi3"), num(row, "d4rl_score"))
    return scores


def render(scores: dict) -> str:
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00")
    scored = len(scores)
    lines = [
        "# TD3+BC K=3 recenter versus fixed_ref",
        "",
        f"Updated {now}.",
        "",
        "Final checkpoint only. `d4rl_pi3` is the deployment actor. "
        "`d4rl_score` is the first actor. "
        "Paired difference is recenter minus fixed_ref. "
        "Pilot scores are not included.",
        "",
        f"Scored runs {scored}/216. A blank cell has no 1M eval yet. "
        "Cell means use only seeds where both branches are scored.",
        "",
        "## Deployment actor `d4rl_pi3`",
        "",
        "| env | T | n | recenter | fixed_ref | delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    deltas = []
    first_deltas = []
    for env in ENVS:
        for tau in TAUS:
            paired = []
            first = []
            for seed in SEEDS:
                left = scores.get((env, tau, "recenter", seed))
                right = scores.get((env, tau, "fixed_ref", seed))
                if left and right and None not in (left[0], right[0]):
                    paired.append((left[0], right[0]))
                if left and right and None not in (left[1], right[1]):
                    first.append((left[1], right[1]))
            cell = mean([a - b for a, b in paired])
            if cell is not None:
                deltas.append(cell)
            lines.append(
                f"| {env} | {tau:g} | {len(paired)}/4 | {fmt(mean([a for a, _ in paired]))} | "
                f"{fmt(mean([b for _, b in paired]))} | {fmt(cell)} |"
            )
    lines += [
        "",
        f"Equal-weight mean of scored cell deltas: {fmt(mean(deltas))} "
        f"over {len(deltas)}/27 cells.",
        "",
        "## First actor `d4rl_score`",
        "",
        "| env | T | n | recenter | fixed_ref | delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for env in ENVS:
        for tau in TAUS:
            paired = []
            for seed in SEEDS:
                left = scores.get((env, tau, "recenter", seed))
                right = scores.get((env, tau, "fixed_ref", seed))
                if left and right and None not in (left[1], right[1]):
                    paired.append((left[1], right[1]))
            cell = mean([a - b for a, b in paired])
            if cell is not None:
                first_deltas.append(cell)
            lines.append(
                f"| {env} | {tau:g} | {len(paired)}/4 | {fmt(mean([a for a, _ in paired]))} | "
                f"{fmt(mean([b for _, b in paired]))} | {fmt(cell)} |"
            )
    lines += [
        "",
        f"Equal-weight mean of scored first-actor cell deltas: {fmt(mean(first_deltas))} "
        f"over {len(first_deltas)}/27 cells.",
        "",
        "## Seeds, deployment `d4rl_pi3`",
        "",
        "Each seed column is `recenter / fixed_ref / delta`.",
        "",
        "| env | T | s0 | s1 | s2 | s3 |",
        "|---|---:|---|---|---|---|",
    ]
    for env in ENVS:
        for tau in TAUS:
            cols = []
            for seed in SEEDS:
                left = scores.get((env, tau, "recenter", seed))
                right = scores.get((env, tau, "fixed_ref", seed))
                lv = left[0] if left else None
                rv = right[0] if right else None
                delta = None if lv is None or rv is None else lv - rv
                cols.append(f"{fmt(lv)} / {fmt(rv)} / {fmt(delta)}")
            lines.append(f"| {env} | {tau:g} | " + " | ".join(cols) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    text = render(collect())
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text(text)
    print(args.path)


if __name__ == "__main__":
    main()
