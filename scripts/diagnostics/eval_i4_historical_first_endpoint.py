#!/usr/bin/env python3
"""CPU re-eval of historical I4 first actor and endpoint from local ckpts.

Historical mpi4_norm eval.csv stores d4rl_pi4 at 1M but leaves d4rl_score
empty on 500/504 cells. This script does not train, does not touch GPUs, and
does not overwrite original eval.csv. Both policies are rolled out on this
host so extraction is a same-stack delta, not a mix of offrl training eval
and a later first-actor dump.

d4rl_score contract: online first actor (actor_params), not the Polyak copy.
d4rl_pi4 contract: online endpoint (actor4_params).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("EIGEN_NUM_THREADS", "1")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

from _lab_import import REPO_ROOT as _ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _ckpt_compat import normalize_checkpoint  # noqa: E402
from recover_i234_first_endpoint import (  # noqa: E402
    ENVIRONMENTS,
    FIELDS,
    SEEDS_I4_HISTORICAL,
    TAUS,
    _optional_finite,
    _tau_key,
)
from train_td3bc import Actor, evaluate, load_checkpoint  # noqa: E402

DEFAULT_LAB = Path("/home/ext_csv/mpi_sweep_lab")
ARCHIVE = _ROOT / "sweep_results/diagnostics/i234_first_endpoint"
KST = timezone(timedelta(hours=9))
RUN_PAT = re.compile(r"(.+)_tau(.+)_mpi4_seed(\d+)$")
FAMILY = "historical_i4_504_cpu_reeval"


def _infer_action_dim(payload: dict[str, Any], mean: np.ndarray) -> int:
    params = payload["actor_params"]["params"]
    kernel = np.asarray(params["Dense_0"]["kernel"])
    output_kernel = np.asarray(params["Dense_2"]["kernel"])
    if kernel.shape[0] != mean.shape[0]:
        raise ValueError("checkpoint actor input dimension does not match normalization")
    return int(output_kernel.shape[-1])


def _make_policy(actor: Actor, params: Any):
    apply = jax.jit(actor.apply)

    def policy(state: np.ndarray) -> np.ndarray:
        x = jnp.asarray(state, dtype=jnp.float32)
        if x.ndim == 1:
            x = x[None, :]
            return np.asarray(apply(params, x)[0], dtype=np.float32)
        return np.asarray(apply(params, x), dtype=np.float32)

    return policy


def _read_original(eval_path: Path) -> tuple[float | None, float | None]:
    with eval_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or rows[-1].get("step") != "1000000":
        raise ValueError(f"{eval_path} missing 1M eval row")
    last = rows[-1]
    return _optional_finite(last.get("d4rl_score")), _optional_finite(last.get("d4rl_pi4"))


def _load_done(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    done = {}
    for row in rows:
        done[(row["environment"], row["T"], row["seed"])] = row
    return done


def _append_row(path: Path, row: dict[str, str]) -> None:
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _list_cells(results_dir: Path) -> list[tuple[str, str, int, Path]]:
    cells = []
    for env in ENVIRONMENTS:
        for tau in TAUS:
            for seed in SEEDS_I4_HISTORICAL:
                tau_s = _tau_key(tau)
                run = results_dir / f"{env}_tau{tau_s}_mpi4_seed{seed}"
                cells.append((env, tau_s, seed, run))
    return cells


def eval_cell(run_dir: Path, env: str, seed: int, episodes: int) -> tuple[float, float, float, float]:
    ckpt = run_dir / "params_1000000.pkl"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    payload = normalize_checkpoint(load_checkpoint(ckpt))
    if "actor_params" not in payload or "actor4_params" not in payload:
        raise KeyError(f"{ckpt} missing actor_params or actor4_params")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    max_action = float(payload.get("max_action", 1.0))
    actor = Actor(action_dim=_infer_action_dim(payload, mean), max_action=max_action)
    first_ret, first_score = evaluate(
        _make_policy(actor, payload["actor_params"]),
        env,
        seed,
        mean,
        std,
        episodes,
    )
    deploy_ret, deploy_score = evaluate(
        _make_policy(actor, payload["actor4_params"]),
        env,
        seed,
        mean,
        std,
        episodes,
    )
    return first_ret, first_score, deploy_ret, deploy_score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", type=Path, default=DEFAULT_LAB)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="defaults to <lab-root>/results/mpi4_norm",
    )
    parser.add_argument("--output-csv", type=Path, default=ARCHIVE / "i4_historical_504_first_endpoint.csv")
    parser.add_argument("--progress-json", type=Path, default=ARCHIVE / "I4_HISTORICAL_504_REEVAL_PROGRESS.json")
    parser.add_argument("--limit", type=int, default=0, help="evaluate at most N unfinished cells; 0 = all")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"refusing non-CPU JAX backend {jax.default_backend()!r}")
    results_dir = args.results_dir or (args.lab_root / "results" / "mpi4_norm")
    cells = _list_cells(results_dir)
    if len(cells) != 504:
        raise ValueError(f"expected 504 I4 cells, listed {len(cells)}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(args.output_csv)
    remaining = [cell for cell in cells if (cell[0], cell[1], str(cell[2])) not in done]
    if args.limit > 0:
        remaining = remaining[: args.limit]
    started = datetime.now(KST)
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "n_done": len(done),
                "n_remaining_this_run": len(remaining),
                "started_at": started.isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        flush=True,
    )
    for env, tau_s, seed, run_dir in remaining:
        orig_first, orig_deploy = _read_original(run_dir / "eval.csv")
        t0 = time.time()
        first_ret, first_score, deploy_ret, deploy_score = eval_cell(
            run_dir, env, seed, args.eval_episodes
        )
        elapsed = time.time() - t0
        row = {
            "family": FAMILY,
            "K": "4",
            "integrator": "implicit",
            "environment": env,
            "T": tau_s,
            "seed": str(seed),
            "tag": run_dir.name,
            "score_column": "d4rl_pi4",
            "target_d4rl": f"{first_score}",
            "deployment_d4rl": f"{deploy_score}",
            "delta_endpoint_minus_first": f"{deploy_score - first_score}",
            "host_run_dir": str(run_dir.resolve()),
        }
        _append_row(args.output_csv, row)
        done[(env, tau_s, str(seed))] = row
        receipt = {
            "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
            "backend": "cpu",
            "n_done": len(done),
            "n_expected": 504,
            "last_cell": {
                "environment": env,
                "T": tau_s,
                "seed": seed,
                "seconds": round(elapsed, 3),
                "target_d4rl": first_score,
                "deployment_d4rl": deploy_score,
                "original_d4rl_score": orig_first,
                "original_d4rl_pi4": orig_deploy,
                "return_first": first_ret,
                "return_pi4": deploy_ret,
            },
        }
        args.progress_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{len(done)}/504] {run_dir.name} first={first_score:.2f} "
            f"pi4={deploy_score:.2f} orig_pi4={orig_deploy} {elapsed:.1f}s",
            flush=True,
        )
    if len(done) == 504:
        deltas = [float(row["delta_endpoint_minus_first"]) for row in done.values()]
        summary = {
            "built_at": datetime.now(KST).isoformat(timespec="seconds"),
            "host": "ext_csv",
            "method": "cpu_checkpoint_reeval",
            "family": FAMILY,
            "n": 504,
            "mean_endpoint_minus_first": statistics.mean(deltas),
            "n_positive": sum(delta > 0.0 for delta in deltas),
            "n_negative": sum(delta < 0.0 for delta in deltas),
            "n_zero": sum(delta == 0.0 for delta in deltas),
            "note": (
                "Same-stack CPU re-eval of online first actor and endpoint. "
                "Original mpi4_norm eval.csv is unchanged. Do not mix with P0 I4."
            ),
        }
        (ARCHIVE / "I4_HISTORICAL_504_REEVAL_SUMMARY.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
