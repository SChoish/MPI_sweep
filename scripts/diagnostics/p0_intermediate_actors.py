"""Shared constants and pure helpers for P0 intermediate-actor diagnostics.

This module is import-safe without JAX. Environment evaluation lives in
``run_p0_intermediate_actors.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts.experiments.run_p0_bar_two_actor_p4 import (
    CONDITIONS,
    ENVIRONMENTS,
    SEEDS,
    T_VALUES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_P0_ROOT = Path("/home/ext_csh/p0_bar_p4_two_actor_p4_29fea94_gpu_v2")
DEFAULT_TAIL_CHECKPOINTS = (
    REPO_ROOT
    / "sweep_results/diagnostics/hosts/ext_csh/p0_manifest_tail90/CHECKPOINTS.json"
)
DEFAULT_TAIL_MANIFEST = (
    REPO_ROOT / "sweep_results/diagnostics/hosts/ext_csh/p0_manifest_tail90/MANIFEST.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "sweep_results/diagnostics/p0_intermediate_actors"
FROZEN_SOURCE_REVISION = "29fea94bee9d9276ac463caa2b897e3474fe1b2b"
EVAL_EPISODE_SEEDS = tuple(range(1000, 1010))
HIGH_T = (10, 14, 20)
COSINE_MIN_L2 = 1e-6
BOUND_FRAC = 0.95
N_DATASET_STATES = 2048
N_ROLLOUT_PER_POLICY = 256
DATASET_SAMPLE_SEED = 20260905
ROLLOUT_SAMPLE_SEED = 20260905
Q_SCALE_EPS = 1e-6

METHOD_LABEL = {"bar_p4": "MART4", "two_actor_p4": "two_actor_p4"}


@dataclass(frozen=True)
class ActorSpec:
    actor_index: int
    actor_id: str
    role: str
    enters_bellman: bool
    objective: str
    q_scale_ref: str
    tau_multiplier: str


MART_ACTORS = (
    ActorSpec(
        1,
        "mart_mu1",
        "first / value-learning actor",
        True,
        "dataset-anchored TD3+BC at T/K",
        "Q(s, mu1(s))",
        "T/K",
    ),
    ActorSpec(
        2,
        "mart_mu2",
        "re-centering hop 2",
        False,
        "W2-proximal to stop-grad(mu1) at T/K",
        "Q(s, mu1(s))",
        "T/K",
    ),
    ActorSpec(
        3,
        "mart_mu3",
        "re-centering hop 3",
        False,
        "W2-proximal to stop-grad(mu2) at T/K",
        "Q(s, mu2(s))",
        "T/K",
    ),
    ActorSpec(
        4,
        "mart_mu4",
        "deployment actor",
        False,
        "W2-proximal to stop-grad(mu3) at T/K",
        "Q(s, mu3(s))",
        "T/K",
    ),
)
TWO_ACTOR_ACTORS = (
    ActorSpec(
        1,
        "two_actor_target",
        "target / value-learning actor",
        True,
        "dataset-anchored TD3+BC at T/K",
        "Q(s, mu_target(s))",
        "T/K",
    ),
    ActorSpec(
        2,
        "two_actor_deploy",
        "deployment actor",
        False,
        "dataset-anchored TD3+BC at T",
        "Q(s, mu_deploy(s))",
        "T",
    ),
)
ACTORS_BY_CONDITION = {"bar_p4": MART_ACTORS, "two_actor_p4": TWO_ACTOR_ACTORS}


def run_key(condition: str, environment: str, tau: int, seed: int) -> str:
    return f"{condition}|{environment}|T={int(tau)}|seed={int(seed)}"


def pair_key(environment: str, tau: int, seed: int) -> str:
    return f"{environment}|T={int(tau)}|seed={int(seed)}"


def planned_runs() -> list[dict[str, Any]]:
    rows = []
    for environment in ENVIRONMENTS:
        for tau in T_VALUES:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    rows.append(
                        {
                            "run_key": run_key(condition, environment, tau, seed),
                            "condition": condition,
                            "method_label": METHOD_LABEL[condition],
                            "environment": environment,
                            "T": int(tau),
                            "seed": int(seed),
                            "actors": [asdict(spec) for spec in ACTORS_BY_CONDITION[condition]],
                        }
                    )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms_action_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-state RMS over action coordinates: sqrt(mean_action_dim((b-a)^2))."""
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return np.sqrt(np.mean(delta * delta, axis=-1))


def vector_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return np.linalg.norm(delta, axis=-1)


def cosine_with_near_zero_mask(
    u: np.ndarray, v: np.ndarray, min_l2: float = COSINE_MIN_L2
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u_norm = np.linalg.norm(u, axis=-1)
    v_norm = np.linalg.norm(v, axis=-1)
    valid = (u_norm >= min_l2) & (v_norm >= min_l2)
    denom = np.clip(u_norm * v_norm, min_l2, None)
    cosine = np.sum(u * v, axis=-1) / denom
    cosine = np.where(valid, cosine, np.nan)
    return cosine, valid


def bound_fraction(actions: np.ndarray, max_action: float, frac: float = BOUND_FRAC) -> float:
    actions = np.asarray(actions, dtype=np.float64)
    return float(np.mean(np.abs(actions) >= frac * float(max_action)))


def hop_tau(total_t: float, multiplier: str) -> float:
    if multiplier == "T":
        return float(total_t)
    if multiplier == "T/K":
        return float(total_t) / 4.0
    raise ValueError(f"unknown tau multiplier {multiplier}")


def coverage_stats(inventory_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    planned = len(planned_runs())
    present = [row for row in inventory_rows if row.get("eval_eligible")]
    missing = [row for row in inventory_rows if not row.get("eval_eligible")]
    by_condition = {condition: 0 for condition in CONDITIONS}
    for row in present:
        by_condition[row["condition"]] += 1
    pair_present: dict[str, set[str]] = {}
    for row in present:
        pair_present.setdefault(pair_key(row["environment"], row["T"], row["seed"]), set()).add(
            row["condition"]
        )
    paired = sum(1 for flags in pair_present.values() if flags == set(CONDITIONS))
    planned_pairs = len(ENVIRONMENTS) * len(T_VALUES) * len(SEEDS)
    planned_policies = 90 * 4 + 90 * 2
    present_policies = sum(len(row["actors"]) for row in present)
    tasks_present = sorted({row["environment"] for row in present})
    return {
        "planned_runs": planned,
        "present_runs": len(present),
        "missing_runs": len(missing),
        "run_coverage": len(present) / planned if planned else 0.0,
        "present_by_condition": by_condition,
        "planned_pairs": planned_pairs,
        "paired_cells_both_methods": paired,
        "pair_coverage": paired / planned_pairs if planned_pairs else 0.0,
        "planned_policies": planned_policies,
        "present_policies": present_policies,
        "policy_coverage": present_policies / planned_policies if planned_policies else 0.0,
        "tasks_present": tasks_present,
        "tasks_missing": [task for task in ENVIRONMENTS if task not in tasks_present],
        "note": (
            "Evaluate present cells only. Paired MART vs two-actor analysis "
            "uses cells where both methods have a 1M checkpoint. Do not "
            "substitute other sweeps."
        ),
    }


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def episode_part_path(out_dir: Path, run_key_value: str, actor_id: str) -> Path:
    safe = run_key_value.replace("|", "__").replace("=", "")
    return out_dir / "episodes" / f"{safe}__{actor_id}.jsonl"


def summarize_mean_std(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }
