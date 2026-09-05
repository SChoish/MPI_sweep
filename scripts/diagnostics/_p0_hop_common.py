"""Shared P0 hop-actor audit constants and inventory (no JAX)."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from _lab_import import REPO_ROOT as ROOT

KST = timezone(timedelta(hours=9))
ARCHIVE = ROOT / "sweep_results/diagnostics/p0_hop_actor_audit"
P0_ARCHIVE = ROOT / "sweep_results/diagnostics/p0_bar_p4_vs_two_actor_p4"
PROTOCOL = ROOT / "scripts/experiments/P0_SHARED_DRIVER_PROTOCOL.json"
FIRST90 = P0_ARCHIVE / "first90_final_scores.csv"
TAIL90 = P0_ARCHIVE / "tail90_final_scores.csv"

ENVIRONMENTS = (
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
HIGH_T = (10, 14, 20)
CONDITIONS = ("bar_p4", "two_actor_p4")
METHOD_LABEL = {"bar_p4": "MART4", "two_actor_p4": "two_actor_p4"}
EVAL_SEED_BASE = 10_000
NEAR_BOUND = 0.99
NEAR_ZERO_RMS = 1e-6
DATASET_N = 4096
ROLLOUT_N_EACH = 2048
DATASET_SAMPLE_SEED = 20260905
ROLLOUT_SAMPLE_SEED = 20260905
DATA_DIR = Path("/raid/ext_csv/datasets/d4rl")

EPISODE_FIELDS = (
    "task",
    "T",
    "training_seed",
    "method",
    "actor_index",
    "actor_role",
    "checkpoint_hash",
    "checkpoint_step",
    "evaluation_seed",
    "raw_return",
    "normalized_score",
    "episode_length",
    "host_run_dir",
)


def now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_roles(condition: str) -> list[dict[str, Any]]:
    if condition == "bar_p4":
        return [
            {
                "actor_index": index,
                "actor_role": f"mu{index}",
                "objective": "dataset_td3bc" if index == 1 else "jko_recenter",
            }
            for index in (1, 2, 3, 4)
        ]
    return [
        {
            "actor_index": 1,
            "actor_role": "target_value",
            "objective": "dataset_td3bc_T_over_K",
        },
        {
            "actor_index": 2,
            "actor_role": "deployment",
            "objective": "dataset_td3bc_T",
        },
    ]


def expected_cells() -> list[dict[str, Any]]:
    cells = []
    for environment in ENVIRONMENTS:
        for tau in TAUS:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    cells.append(
                        {
                            "key": f"{condition}|{environment}|T={tau}|seed={seed}",
                            "condition": condition,
                            "method": METHOD_LABEL[condition],
                            "environment": environment,
                            "T": tau,
                            "seed": seed,
                        }
                    )
    return cells


def _load_score_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        key = row.get("key") or (
            f"{row['condition']}|{row['environment']}|T={row['T']}|seed={row['seed']}"
        )
        out[key] = row
    return out


def _inspect_checkpoint(ckpt: Path) -> dict[str, Any]:
    with ckpt.open("rb") as handle:
        payload = pickle.load(handle)
    config = payload.get("config") or {}
    actors = payload.get("actors_params")
    n_actors = len(actors) if actors is not None else None
    mean = payload.get("mean")
    std = payload.get("std")
    return {
        "step": int(payload.get("step") or 0),
        "n_actors": n_actors,
        "max_action": float(payload.get("max_action", 1.0)),
        "normalize": bool(config.get("normalize", True)),
        "q_scale_norm": config.get("q_scale_norm"),
        "method_config": config.get("method"),
        "mpi_steps": config.get("mpi_steps"),
        "tau_config": config.get("tau"),
        "integrator": config.get("integrator"),
        "reference_depth": config.get("reference_depth"),
        "mean_dim": int(getattr(mean, "shape", [0])[0]) if mean is not None else None,
        "std_dim": int(getattr(std, "shape", [0])[0]) if std is not None else None,
        "has_actors_params": actors is not None,
        "has_critic_params": "critic_params" in payload,
        "has_target_actor": "target_actor_params" in payload,
    }


def build_inventory() -> dict[str, Any]:
    first = _load_score_rows(FIRST90)
    tail = _load_score_rows(TAIL90)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8")) if PROTOCOL.is_file() else {}
    rows = []
    n_local = 0
    n_evalable = 0
    for cell in expected_cells():
        key = cell["key"]
        source = first.get(key) or tail.get(key)
        record = dict(cell)
        record["actors"] = actor_roles(cell["condition"])
        record["score_table"] = "first90" if key in first else ("tail90" if key in tail else None)
        if source is None:
            record["status"] = "missing_from_score_tables"
            record["missing_reason"] = "no row in first90 or tail90"
            rows.append(record)
            continue
        run_dir = Path(source["host_run_dir"])
        ckpt = run_dir / "params_1000000.pkl"
        record["host_run_dir"] = str(run_dir)
        record["checkpoint_path"] = str(ckpt)
        record["archived_deployment_d4rl"] = source.get("deployment_d4rl")
        record["archived_target_d4rl"] = source.get("target_d4rl")
        if not ckpt.is_file():
            host_hint = str(run_dir)
            if "ext_csh" in host_hint:
                record["status"] = "missing_on_this_host"
                record["missing_reason"] = (
                    "1M checkpoint lives on ext_csh; not re-trained and not replaced"
                )
            else:
                record["status"] = "checkpoint_absent"
                record["missing_reason"] = f"params_1000000.pkl not found at {ckpt}"
            rows.append(record)
            continue
        n_local += 1
        try:
            info = _inspect_checkpoint(ckpt)
        except Exception as error:
            record["status"] = "unreadable"
            record["missing_reason"] = f"{type(error).__name__}: {error}"
            rows.append(record)
            continue
        expected_n = 4 if cell["condition"] == "bar_p4" else 2
        if info["n_actors"] != expected_n:
            record["status"] = "incompatible_actor_count"
            record["missing_reason"] = (
                f"expected {expected_n} actors, checkpoint has {info['n_actors']}"
            )
            record["checkpoint"] = info
            rows.append(record)
            continue
        record["status"] = "evalable"
        record["checkpoint_hash"] = sha256_file(ckpt)
        record["checkpoint"] = info
        n_evalable += 1
        rows.append(record)

    paired = 0
    for environment in ENVIRONMENTS:
        for tau in TAUS:
            for seed in SEEDS:
                mart = next(
                    (
                        row
                        for row in rows
                        if row["environment"] == environment
                        and row["T"] == tau
                        and row["seed"] == seed
                        and row["condition"] == "bar_p4"
                        and row["status"] == "evalable"
                    ),
                    None,
                )
                control = next(
                    (
                        row
                        for row in rows
                        if row["environment"] == environment
                        and row["T"] == tau
                        and row["seed"] == seed
                        and row["condition"] == "two_actor_p4"
                        and row["status"] == "evalable"
                    ),
                    None,
                )
                if mart and control:
                    paired += 1
                    mart["paired"] = True
                    control["paired"] = True
                else:
                    if mart:
                        mart["paired"] = False
                    if control:
                        control["paired"] = False

    n_expected = len(expected_cells())
    return {
        "built_at": now_kst(),
        "host": "ext_csv",
        "understood_as": (
            "Diagnose MART vs two-actor similarity via hop-level returns and "
            "common-state action distances on existing P0 K=4 checkpoints. "
            "No new training. Do not mix I4 first/endpoint scores. Do not "
            "substitute other sweeps. Paired analysis only where both methods exist."
        ),
        "grid": {
            "environments": list(ENVIRONMENTS),
            "T": list(TAUS),
            "seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "n_expected_runs": n_expected,
        },
        "eval_protocol": {
            "episodes": 10,
            "evaluation_seed_base": EVAL_SEED_BASE,
            "eval_env_map": "train_td3bc.EVAL_ENV gymnasium MuJoCo v4",
            "obs_norm": "checkpoint mean/std; training used eps=1e-3 kept-observation",
            "deterministic": True,
            "do_not_mix_with": [
                "i4_historical_504_first_endpoint.csv",
                "all180_first_endpoint_scores.csv archived training eval",
            ],
        },
        "coverage": {
            "n_expected_runs": n_expected,
            "n_local_checkpoints": n_local,
            "n_evalable": n_evalable,
            "n_paired_task_T_seed": paired,
            "fraction_runs_evalable": n_evalable / n_expected,
            "fraction_paired": paired / (len(ENVIRONMENTS) * len(TAUS) * len(SEEDS)),
            "missing_tasks_on_this_host": [
                env
                for env in ENVIRONMENTS
                if not any(
                    row["environment"] == env and row["status"] == "evalable" for row in rows
                )
            ],
        },
        "bellman_and_objectives": {
            "critic_target_actor": "Polyak copy of actor 1 only; downstream actors never enter backups",
            "MART_hop_tau": "T/K with K=4",
            "two_actor_target_tau": "T/K",
            "two_actor_deployment_tau": "T",
            "q_weight": "lambda = 2*tau / mean(|Q|) when q_scale_norm is true",
            "source_revision_note": (
                "Current train_td3bc.py is the eval/restore code. Checkpoint "
                "config fields are recorded per run; do not assume every "
                "historical run matches HEAD."
            ),
        },
        "protocol_status": protocol.get("status"),
        "rows": rows,
    }


def write_inventory(path: Path | None = None) -> dict[str, Any]:
    inventory = build_inventory()
    path = path or (ARCHIVE / "INVENTORY.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    compact = [{k: row.get(k) for k in (
        "key",
        "status",
        "method",
        "environment",
        "T",
        "seed",
        "checkpoint_hash",
        "missing_reason",
        "paired",
        "host_run_dir",
    )} for row in inventory["rows"]]
    with (ARCHIVE / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "key",
                "status",
                "method",
                "environment",
                "T",
                "seed",
                "checkpoint_hash",
                "missing_reason",
                "paired",
                "host_run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(compact)
    (ARCHIVE / "COVERAGE.json").write_text(
        json.dumps(inventory["coverage"], indent=2) + "\n", encoding="utf-8"
    )
    return inventory
