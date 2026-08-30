#!/usr/bin/env python3
"""Run and summarize paired simulator-reference audits for route shadows."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from _lab_import import REPO_ROOT as ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

from launch_route_shadow import DEFAULT_CELLS, parse_cells  # noqa: E402

SOURCES = ("shadow_mu1", "shadow_muk")
PRIMARY_KEY = "critic_symmetric_relative_error_rho_at_ak_p50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default=DEFAULT_CELLS)
    parser.add_argument("--seeds", default="0 1")
    parser.add_argument("--mpi-steps", type=int, choices=(2, 3), default=3)
    parser.add_argument("--steps", default="250000 500000 750000 1000000")
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results_route_shadow"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "audit_report_20260829" / "route_shadow_mc",
    )
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--n-states", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--smoothing-rollouts", type=int, default=4)
    parser.add_argument("--rollout-seed", type=int, default=20260829)
    parser.add_argument("--sample-seed", type=int, default=20260830)
    parser.add_argument("--audit-seed", type=int, default=20260831)
    parser.add_argument("--dead-zone", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-job", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    cells = parse_cells(args.cells)
    seeds = [int(value) for value in args.seeds.split()]
    steps = [int(value) for value in args.steps.split()]
    if not seeds or not steps:
        raise ValueError("--seeds and --steps must not be empty")
    jobs: list[dict[str, Any]] = []
    for env_name, tau in cells:
        for seed, step, source in itertools.product(seeds, steps, SOURCES):
            run_tag = (
                f"{env_name}_tau{tau}_mpi{args.mpi_steps}_seed{seed}"
                "_route_shadow"
            )
            checkpoint = args.results_dir / run_tag / f"params_{step}.pkl"
            output = (
                args.out_dir
                / env_name
                / f"tau{tau}_mpi{args.mpi_steps}_seed{seed}"
                / f"step{step}"
                / source
            )
            jobs.append(
                {
                    "env": env_name,
                    "tau": tau,
                    "seed": seed,
                    "step": step,
                    "source": source,
                    "checkpoint": checkpoint,
                    "output": output,
                }
            )
    return jobs


def audit_command(job: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("mc_value_audit.py")),
        f"--checkpoint={job['checkpoint']}",
        f"--env={job['env']}",
        f"--out-dir={job['output']}",
        f"--critic-source={job['source']}",
        f"--episodes={args.episodes}",
        f"--n-states={args.n_states}",
        f"--horizon={args.horizon}",
        f"--gamma={args.gamma}",
        f"--smoothing-rollouts={args.smoothing_rollouts}",
        f"--rollout-seed={args.rollout_seed}",
        f"--sample-seed={args.sample_seed}",
        f"--audit-seed={args.audit_seed}",
        f"--dead-zone={args.dead_zone}",
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def run_job(job: dict[str, Any], args: argparse.Namespace) -> tuple[str, bool, str]:
    label = (
        f"{job['env']} T={job['tau']} seed={job['seed']} "
        f"step={job['step']} {job['source']}"
    )
    summary_path = job["output"] / "summary.json"
    if summary_path.is_file() and not args.overwrite:
        return label, True, "skipped existing summary"
    if not job["checkpoint"].is_file():
        return label, False, f"missing checkpoint {job['checkpoint']}"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_PLATFORM_NAME"] = "cpu"
    thread_count = str(args.threads_per_job)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = thread_count
    result = subprocess.run(
        audit_command(job, args),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    job["output"].mkdir(parents=True, exist_ok=True)
    (job["output"] / "mc.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    return label, result.returncode == 0, f"return code {result.returncode}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(jobs: list[dict[str, Any]], args: argparse.Namespace) -> None:
    indexed: dict[tuple[str, str, int, int], dict[str, float]] = {}
    for job in jobs:
        summary_path = job["output"] / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        value = summary.get(PRIMARY_KEY)
        if value is None:
            raise ValueError(f"{summary_path} has no finite {PRIMARY_KEY}")
        key = (job["env"], job["tau"], job["seed"], job["step"])
        indexed.setdefault(key, {})[job["source"]] = float(value)

    checkpoint_rows: list[dict[str, Any]] = []
    for (env_name, tau, seed, step), values in sorted(indexed.items()):
        if any(source not in values for source in SOURCES):
            continue
        checkpoint_rows.append(
            {
                "env": env_name,
                "tau": tau,
                "seed": seed,
                "step": step,
                "shadow_mu1_error_p50": values["shadow_mu1"],
                "shadow_muk_error_p50": values["shadow_muk"],
                "paired_delta_muk_minus_mu1": (
                    values["shadow_muk"] - values["shadow_mu1"]
                ),
            }
        )
    write_csv(args.out_dir / "checkpoint_pairs.csv", checkpoint_rows)

    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in checkpoint_rows:
        key = (str(row["env"]), str(row["tau"]), int(row["seed"]))
        grouped.setdefault(key, []).append(float(row["paired_delta_muk_minus_mu1"]))
    expected_steps = len(set(int(value) for value in args.steps.split()))
    run_rows: list[dict[str, Any]] = []
    for (env_name, tau, seed), deltas in sorted(grouped.items()):
        if len(deltas) != expected_steps:
            raise ValueError(
                f"incomplete checkpoint pair for {env_name} T={tau} seed={seed}: "
                f"expected {expected_steps}, found {len(deltas)}"
            )
        run_rows.append(
            {
                "env": env_name,
                "tau": tau,
                "seed": seed,
                "n_checkpoints": len(deltas),
                "mean_paired_delta_muk_minus_mu1": float(np.mean(deltas)),
            }
        )
    write_csv(args.out_dir / "run_level.csv", run_rows)
    if not run_rows:
        raise ValueError("no complete route-shadow pairs were available")

    run_deltas = np.asarray(
        [float(row["mean_paired_delta_muk_minus_mu1"]) for row in run_rows],
        dtype=np.float64,
    )
    observed = abs(float(np.mean(run_deltas)))
    sign_flip_values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(run_deltas)):
        sign_flip_values.append(abs(float(np.mean(run_deltas * np.asarray(signs)))))
    permutation_p = float(
        np.mean(np.asarray(sign_flip_values) >= (observed - 1e-15))
    )
    rng = np.random.default_rng(20260829)
    bootstrap_means = np.empty(args.bootstrap_samples, dtype=np.float64)
    for index in range(args.bootstrap_samples):
        sample = rng.choice(run_deltas, size=len(run_deltas), replace=True)
        bootstrap_means[index] = np.mean(sample)
    aggregate = {
        "primary_key": PRIMARY_KEY,
        "run_estimand": (
            "mean over requested checkpoints of shadow_muk minus shadow_mu1 "
            "per-checkpoint state-median symmetric relative error against the "
            "common Q_MC^{rho_1} first-route estimand"
        ),
        "reference_estimand": (
            "simulator return under the effective first-route backup "
            "continuation rho_1"
        ),
        "interpretation_caveat": (
            "routes imply different continuation policies; this measures "
            "relative alignment with the common Q_MC^{rho_1} first-route "
            "estimand, not each shadow critic's calibration to its own "
            "continuation"
        ),
        "inference_unit": "training run",
        "n_runs": int(len(run_deltas)),
        "mean_delta": float(np.mean(run_deltas)),
        "median_delta": float(np.median(run_deltas)),
        "positive_delta_runs": int(np.sum(run_deltas > 0.0)),
        "two_sided_exact_sign_flip_p": permutation_p,
        "run_bootstrap_mean_95_interval": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "bootstrap_samples": int(args.bootstrap_samples),
        "directional_hypothesis": "mean_delta > 0",
        "decision_rule": (
            "support requires mean_delta > 0 and "
            "two_sided_exact_sign_flip_p < 0.05"
        ),
        "scope": "targeted mechanism study; not a universal task-level estimate",
    }
    (args.out_dir / "SUMMARY.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.threads_per_job < 1:
        raise ValueError("workers and threads-per-job must be positive")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    jobs = build_jobs(args)
    if args.dry_run:
        print(f"[route-shadow-mc] jobs={len(jobs)}", flush=True)
        for job in jobs:
            status = "ready" if job["checkpoint"].is_file() else "missing"
            print(
                f"[{status}] {job['checkpoint']} source={job['source']} "
                f"out={job['output']}",
                flush=True,
            )
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "paired_route_shadow_simulator_reference",
        "primary_key": PRIMARY_KEY,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job, args): job for job in jobs}
        for future in as_completed(futures):
            label, success, detail = future.result()
            print(f"[{'ok' if success else 'fail'}] {label}: {detail}", flush=True)
            if not success:
                failures.append(f"{label}: {detail}")
    if failures:
        raise RuntimeError("route-shadow MC failures:\n" + "\n".join(failures))
    summarize(jobs, args)
    print(f"[done] paired summary={args.out_dir / 'SUMMARY.json'}", flush=True)


if __name__ == "__main__":
    main()
