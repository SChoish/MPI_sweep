#!/usr/bin/env python3
"""Completeness, config, and paired-key audit for the BAR-P3 / MCEP-P3 grids.

This does not compute manuscript-facing score estimates. It checks that both
90-run grids exist, share the locked protocol keys, and can be paired by
environment, tau, and seed; the compact paired release handles estimation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
TAUS = (4, 7, 10, 14, 20)
SEEDS = (0, 1)
SHARED = {
    "mpi_steps": 3,
    "integrator": "implicit",
    "q_scale_norm": True,
    "normalize": True,
    "max_timesteps": 1_000_000,
    "eval_episodes": 10,
    "eval_freq": 1_000_000,
    "batch_size": 256,
    "discount": 0.99,
    "polyak": 0.005,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_freq": 2,
    "lr": 0.0003,
    "n_jitted_updates": 8,
    "updates_per_dispatch": 64,
}
CPU_FINISH_PREFIX = "walker2d-expert-v2"


def portable_path(path: Path) -> str:
    repo = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar-dir", default=str(root / "results" / "bar_p3"))
    parser.add_argument("--mcep-dir", default=str(root / "results" / "mcep_p3"))
    parser.add_argument(
        "--out",
        default=str(
            root / "sweep_results" / "diagnostics" / "bar_mcep_p3_paired" / "AUDIT.json"
        ),
    )
    return parser.parse_args()


def run_dir(root: Path, env: str, tau: int, seed: int, method: str) -> Path:
    tag = "mpi3" if method == "bar" else "mcep3"
    return root / f"{env}_tau{tau}_{tag}_seed{seed}"


def final_eval(path: Path) -> tuple[float | None, int | None, str | None]:
    eval_path = path / "eval.csv"
    if not eval_path.is_file():
        return None, None, "missing eval.csv"
    rows = list(csv.DictReader(eval_path.open()))
    if not rows:
        return None, None, "empty eval.csv"
    at_budget = [row for row in rows if int(float(row["step"])) >= 1_000_000]
    row = at_budget[-1] if at_budget else rows[-1]
    step = int(float(row["step"]))
    if step < 1_000_000:
        return float(row["d4rl_score"]), step, f"eval step {step} < 1000000"
    return float(row["d4rl_score"]), step, None


def check_config(path: Path, method: str, env: str, tau: int, seed: int) -> list[str]:
    errors = []
    config_path = path / "config.json"
    if not config_path.is_file():
        return ["missing config.json"]
    cfg = json.loads(config_path.read_text())
    expected = {
        "method": method,
        "env": env,
        "tau": float(tau),
        "seed": seed,
        **SHARED,
    }
    for key, value in expected.items():
        if key not in cfg:
            errors.append(f"missing config key {key}")
            continue
        actual = cfg[key]
        if isinstance(value, float):
            if abs(float(actual) - value) > 1e-12:
                errors.append(f"{key}={actual!r} != {value}")
        elif actual != value:
            errors.append(f"{key}={actual!r} != {value}")
    return errors


def audit_grid(root: Path, method: str) -> dict:
    cells = []
    failures = []
    for env in ENVS:
        for tau in TAUS:
            for seed in SEEDS:
                path = run_dir(root, env, tau, seed, method)
                key = {"environment": env, "tau": tau, "seed": seed, "dir": str(path)}
                if not path.is_dir():
                    failures.append({**key, "error": "missing run directory"})
                    continue
                ckpt = path / "params_1000000.pkl"
                if not ckpt.is_file():
                    failures.append({**key, "error": "missing params_1000000.pkl"})
                    continue
                score, step, eval_error = final_eval(path)
                if eval_error:
                    failures.append({**key, "error": eval_error, "score": score, "step": step})
                    continue
                config_errors = check_config(path, method, env, tau, seed)
                cfg = json.loads((path / "config.json").read_text())
                cell = {
                    **key,
                    "score": score,
                    "step": step,
                    "save_interval": cfg.get("save_interval"),
                    "restore_path": cfg.get("restore_path"),
                    "cpu_finish_cell": env.startswith(CPU_FINISH_PREFIX),
                }
                if config_errors:
                    failures.append({**cell, "error": "; ".join(config_errors)})
                    continue
                cells.append(cell)
    return {
        "root": portable_path(root),
        "n_expected": 90,
        "n_ok": len(cells),
        "n_fail": len(failures),
        "failures": failures,
        "cells": cells,
    }


def main() -> int:
    args = parse_args()
    bar = audit_grid(Path(args.bar_dir), "bar")
    mcep = audit_grid(Path(args.mcep_dir), "mcep")
    bar_keys = {(c["environment"], c["tau"], c["seed"]) for c in bar["cells"]}
    mcep_keys = {(c["environment"], c["tau"], c["seed"]) for c in mcep["cells"]}
    expected = {(env, tau, seed) for env in ENVS for tau in TAUS for seed in SEEDS}
    paired = sorted(bar_keys & mcep_keys)
    report = {
        "protocol": "bar_mcep_p3_paired_five_budget",
        "n_expected_each": 90,
        "bar": {k: v for k, v in bar.items() if k != "cells"},
        "mcep": {k: v for k, v in mcep.items() if k != "cells"},
        "paired_keys": len(paired),
        "missing_bar_keys": sorted(
            f"{env}_tau{tau}_seed{seed}" for env, tau, seed in sorted(expected - bar_keys)
        ),
        "missing_mcep_keys": sorted(
            f"{env}_tau{tau}_seed{seed}" for env, tau, seed in sorted(expected - mcep_keys)
        ),
        "shared_config": SHARED,
        "pass": (
            bar["n_ok"] == 90
            and mcep["n_ok"] == 90
            and bar_keys == mcep_keys == expected
            and not bar["failures"]
            and not mcep["failures"]
        ),
        "cpu_finish_note": (
            "Eight MCEP walker2d-expert cells resumed on CPU from matching "
            "logged emergency-checkpoint steps; two tau=20 cells started and "
            "completed CPU-only."
        ),
        "manuscript_status": (
            "paired score release is governed by paired_final_scores.csv and "
            "SUMMARY.json; AUDIT.json is completeness/config only"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {**report, "bar_failures": bar["failures"], "mcep_failures": mcep["failures"]}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
