#!/usr/bin/env python3
"""CPU-only recovery of tail90 first-actor scores from local eval.csv.

Reads existing 1M evaluation rows. Does not train or re-evaluate.

target_d4rl  <- d4rl_score  (online first / target actor)
deployment_d4rl <- d4rl_pi4 (BAR-P4) or d4rl_eval (two-actor-P4)

Endpoint values and eval SHA-256 must match the published compact tail shard.
This does not repair the failed P0 primary inclusion gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "sweep_results/diagnostics/p0_bar_p4_vs_two_actor_p4"
TAIL_SCORES = ROOT / "sweep_results/diagnostics/hosts/ext_csh/p0_manifest_tail90/scores.csv"
DEFAULT_RUNS = Path("/home/ext_csh/p0_bar_p4_two_actor_p4_29fea94_gpu_v2/runs")
KST = timezone(timedelta(hours=9))

HEAD_FIELDS = (
    "key",
    "condition",
    "environment",
    "T",
    "seed",
    "tag",
    "score_column",
    "deployment_d4rl",
    "target_d4rl",
    "return_deploy_col",
    "host_run_dir",
)
DEPLOY_COL = {
    "bar_p4": ("d4rl_pi4", "return_pi4"),
    "two_actor_p4": ("d4rl_eval", "return_eval"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _intish(value: str) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def _tag(condition: str, environment: str, t_value: str, seed: str) -> str:
    method = "mpi4" if condition == "bar_p4" else "mcep4"
    return f"{environment}_tau{_intish(t_value)}_{method}_seed{seed}"


def _read_final_eval(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty eval.csv: {path}")
    last = rows[-1]
    if last.get("step") != "1000000":
        raise ValueError(f"{path} final step is {last.get('step')!r}, not 1000000")
    return last


def _finite(name: str, raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}: {raw!r}")
    return value


def _condition_stats(rows: list[dict[str, str]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for condition in ("bar_p4", "two_actor_p4"):
        subset = [row for row in rows if row["condition"] == condition]
        deltas = [
            float(row["deployment_d4rl"]) - float(row["target_d4rl"]) for row in subset
        ]
        out[condition] = {
            "n": len(subset),
            "mean_endpoint_minus_first": statistics.mean(deltas) if deltas else None,
            "n_positive": sum(delta > 0.0 for delta in deltas),
            "n_negative": sum(delta < 0.0 for delta in deltas),
            "n_zero": sum(delta == 0.0 for delta in deltas),
        }
    return out


def _task_stats(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["environment"])].append(row)
    summary = []
    for (condition, environment), subset in sorted(grouped.items()):
        deltas = [
            float(row["deployment_d4rl"]) - float(row["target_d4rl"]) for row in subset
        ]
        summary.append(
            {
                "condition": condition,
                "environment": environment,
                "n": len(subset),
                "mean_endpoint_minus_first": statistics.mean(deltas),
                "n_positive": sum(delta > 0.0 for delta in deltas),
            }
        )
    return summary


def recover_tail90(runs_root: Path) -> list[dict[str, str]]:
    with TAIL_SCORES.open(newline="", encoding="utf-8") as handle:
        compact = list(csv.DictReader(handle))
    if len(compact) != 90:
        raise ValueError(f"expected 90 compact tail rows, got {len(compact)}")

    recovered = []
    for row in compact:
        condition = row["condition"]
        if condition not in DEPLOY_COL:
            raise ValueError(f"unknown condition: {condition}")
        score_col, return_col = DEPLOY_COL[condition]
        if row["score_column"] != score_col:
            raise ValueError(
                f"{row['run_key']} compact score_column {row['score_column']!r} "
                f"!= contract {score_col!r}"
            )
        tag = _tag(condition, row["environment"], row["T"], row["seed"])
        run_dir = runs_root / condition / tag
        eval_path = run_dir / "eval.csv"
        config_path = run_dir / "config.json"
        if not eval_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(run_dir)

        eval_digest = sha256_file(eval_path)
        if eval_digest != row["eval_sha256"]:
            raise ValueError(f"eval SHA-256 mismatch: {eval_path}")

        final = _read_final_eval(eval_path)
        first = _finite("d4rl_score", final["d4rl_score"])
        deploy = _finite(score_col, final[score_col])
        compact_score = _finite("compact score", row["score"])
        if not math.isclose(deploy, compact_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{row['run_key']} endpoint {deploy} != compact {compact_score}"
            )
        if return_col not in final:
            raise ValueError(f"{eval_path} missing {return_col}")

        recovered.append(
            {
                "key": row["run_key"],
                "condition": condition,
                "environment": row["environment"],
                "T": _intish(row["T"]),
                "seed": row["seed"],
                "tag": tag,
                "score_column": score_col,
                "deployment_d4rl": f"{deploy}",
                "target_d4rl": f"{first}",
                "return_deploy_col": return_col,
                "host_run_dir": str(run_dir.resolve()),
            }
        )
    keys = [row["key"] for row in recovered]
    if len(set(keys)) != 90:
        raise ValueError("recovered tail90 keys are not unique")
    return recovered


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=ARCHIVE)
    args = parser.parse_args()

    tail_rows = recover_tail90(args.runs_root)
    with (ARCHIVE / "first90_final_scores.csv").open(newline="", encoding="utf-8") as handle:
        head_rows = list(csv.DictReader(handle))
    if list(head_rows[0].keys()) != list(HEAD_FIELDS):
        raise ValueError("first90_final_scores.csv schema drift")
    if len(head_rows) != 90:
        raise ValueError(f"expected 90 first90 rows, got {len(head_rows)}")

    all_rows = head_rows + tail_rows
    all_keys = [row["key"] for row in all_rows]
    if len(set(all_keys)) != 180:
        raise ValueError("combined first/endpoint keys are not 180 unique")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tail_path = out / "tail90_final_scores.csv"
    all_path = out / "all180_first_endpoint_scores.csv"
    receipt_path = out / "FIRST_ENDPOINT_RECOVERY.json"
    _write_csv(tail_path, HEAD_FIELDS, tail_rows)
    _write_csv(all_path, HEAD_FIELDS, all_rows)

    receipt = {
        "built_at": datetime.now(KST).isoformat(timespec="seconds"),
        "method": "cpu_eval_csv_recovery",
        "no_reevaluation": True,
        "no_training": True,
        "evaluation_contract": {
            "target_d4rl": "d4rl_score (online first / target actor)",
            "bar_p4_deployment": "d4rl_pi4",
            "two_actor_p4_deployment": "d4rl_eval",
        },
        "inputs": {
            "runs_root": str(args.runs_root.resolve()),
            "compact_tail_scores": str(TAIL_SCORES),
            "first90_scores": str(ARCHIVE / "first90_final_scores.csv"),
        },
        "outputs": {
            "tail90_final_scores": str(tail_path),
            "all180_first_endpoint_scores": str(all_path),
        },
        "integrity": {
            "tail90_rows": 90,
            "all180_rows": 180,
            "endpoint_matches_compact_tail": True,
            "eval_sha256_matches_compact_tail": True,
            "manuscript_primary_promotion_allowed": False,
            "note": (
                "Recovers missing first-actor columns for the ext_csh shard. "
                "Does not repair the failed frozen P0 primary inclusion gate."
            ),
        },
        "tail90": _condition_stats(tail_rows),
        "all180": _condition_stats(all_rows),
        "tail90_by_task": _task_stats(tail_rows),
        "all180_by_task": _task_stats(all_rows),
        "hashes": {
            "tail90_final_scores_sha256": sha256_file(tail_path),
            "all180_first_endpoint_scores_sha256": sha256_file(all_path),
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["tail90"], indent=2))
    print(json.dumps(receipt["all180"], indent=2))
    print(f"wrote {tail_path}")
    print(f"wrote {all_path}")
    print(f"wrote {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
