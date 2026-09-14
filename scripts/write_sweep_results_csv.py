#!/usr/bin/env python3
"""Write seed{2,3} D4RL matrices under sweep_results/ from local results/*_s23."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sweep_results"
SEEDS = (2, 3)
ENVS = [
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-expert-v2",
]
# Base MPI 14-tau grid, plus high-τ extension matching TD3 dense tail.
TAUS = [
    0.05,
    0.1,
    0.2,
    0.4,
    0.7,
    1.5,
    2.5,
    4.0,
    7.0,
    10.0,
    12.0,
    14.0,
    17.0,
    20.0,
    24.0,
    28.0,
    34.0,
    40.0,
]

# (K, integrator_dir, results_subdir, run_tag, preferred_score_col)
SWEEPS = [
    (1, "Imp", "mpi1_s23", "mpi1", "d4rl_pi1"),
    (2, "Imp", "mpi2_s23", "mpi2", "d4rl_pi2"),
    (3, "Imp", "mpi3_s23", "mpi3", "d4rl_pi3"),
    (8, "Imp", "mpi8_s23", "mpi8", "d4rl_pi8"),
    (1, "Exp", "exp1_s23", "exp1", "d4rl_pi1"),
    (2, "Exp", "exp2_s23", "exp2", "d4rl_pi2"),
    (3, "Exp", "exp3_s23", "exp3", "d4rl_pi3"),
]


def tau_key(t: float) -> str:
    return f"{t:g}"


def fmt_score(v: float) -> str:
    return f"{v:.4f}"


def collect(root: Path, tag: str, score_key: str) -> dict[tuple[str, str, int], float]:
    """(env, tau_str, seed) -> d4rl score at step 1M only."""
    out: dict[tuple[str, str, int], float] = {}
    pat = re.compile(rf"(.+)_tau(.+)_{re.escape(tag)}_seed(\d+)$")
    if not root.is_dir():
        return out
    for path in root.glob("*/eval.csv"):
        match = pat.match(path.parent.name)
        if not match:
            continue
        seed = int(match.group(3))
        if seed not in SEEDS:
            continue
        if not (path.parent / "params_1000000.pkl").is_file():
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if not rows:
            continue
        last = rows[-1]
        if int(float(last["step"])) != 1_000_000:
            continue
        raw = last.get(score_key) or last.get("d4rl_score")
        if raw is None or raw == "":
            continue
        env, tau = match.group(1), match.group(2)
        out[(env, tau, seed)] = float(raw)
    return out


def write_seed_csv(
    path: Path, cells: dict[tuple[str, str, int], float], seed: int
) -> tuple[int, int]:
    """Write matrix CSV. Returns (filled_cells, total_cells)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    filled = 0
    total = len(TAUS) * len(ENVS)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", *ENVS])
        for t in TAUS:
            tk = tau_key(t)
            row: list[str] = [tk]
            for env in ENVS:
                v = cells.get((env, tk, seed))
                if v is None:
                    row.append("")
                else:
                    row.append(fmt_score(v))
                    filled += 1
            w.writerow(row)
    return filled, total


def main() -> None:
    stats: list[str] = []
    any_change = False
    for k, integ, subdir, tag, score_key in SWEEPS:
        cells = collect(ROOT / "results" / subdir, tag, score_key)
        for seed in SEEDS:
            path = OUT / f"K={k}" / integ / f"seed{seed}.csv"
            filled_preview = sum(
                1 for t in TAUS for env in ENVS if (env, tau_key(t), seed) in cells
            )
            if filled_preview == 0:
                # Never delete: other hosts may own/share the same path via git.
                stats.append(f"K={k}/{integ}/seed{seed}.csv skip(empty)")
                continue
            before = path.read_text(encoding="utf-8") if path.is_file() else None
            filled, total = write_seed_csv(path, cells, seed)
            after = path.read_text(encoding="utf-8")
            changed = before != after
            any_change = any_change or changed
            flag = "updated" if changed else "unchanged"
            stats.append(f"K={k}/{integ}/seed{seed}.csv {filled}/{total} {flag}")
    for line in stats:
        print(line)
    print(f"any_change={any_change}")


if __name__ == "__main__":
    main()
