#!/usr/bin/env python3
"""CPU recovery of I2/I3/I4 first-actor vs endpoint from existing eval.csv.

Does not train or re-evaluate. Freezes the locally available inclusion grid
before writing scores. Historical s23 runs, explicit reruns, and P0 I4 stay
in separate tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ARCHIVE = ROOT / "sweep_results/diagnostics/i234_first_endpoint"
P0_ALL180 = (
    ROOT / "sweep_results/diagnostics/p0_bar_p4_vs_two_actor_p4/all180_first_endpoint_scores.csv"
)
DEFAULT_EXT_CSV_LAB = Path("/home/ext_csv/mpi_sweep_lab")
KST = timezone(timedelta(hours=9))
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
TAUS = (
    0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4.0, 7.0, 10.0, 12.0, 14.0, 17.0, 20.0,
)
SEEDS_S23 = (2, 3)
SEEDS_S01 = (0, 1)
SEEDS_I4_HISTORICAL = (0, 1, 2, 3)
SEEDS_LOCAL = SEEDS_S23
SEEDS_ABSENT = SEEDS_S01
P0_C_K1_TAUS = (0.625, 1.0, 1.75, 3.0, 3.5, 4.25, 5.0)
PLATEAU = {0.05, 0.1, 0.2, 0.4}
TRANSITION = {0.7, 1.5, 2.5, 4.0, 7.0}
TAIL = {10.0, 12.0, 14.0, 17.0, 20.0}
FIELDS = (
    "family",
    "K",
    "integrator",
    "environment",
    "T",
    "seed",
    "tag",
    "score_column",
    "target_d4rl",
    "deployment_d4rl",
    "delta_endpoint_minus_first",
    "host_run_dir",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tau_key(value: float) -> str:
    return f"{value:g}"


def _finite(name: str, raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}: {raw!r}")
    return value


def _region(tau: float) -> str:
    if tau in PLATEAU:
        return "plateau"
    if tau in TRANSITION:
        return "transition"
    if tau in TAIL:
        return "tail"
    return "other"


def recover_grid(
    family: str,
    k: int,
    integrator: str,
    results_dir: Path,
    tag: str,
    seeds: tuple[int, ...],
    environments: tuple[str, ...] = ENVIRONMENTS,
    taus: tuple[float, ...] = TAUS,
) -> list[dict[str, str]]:
    score_col = f"d4rl_pi{k}"
    pat = re.compile(rf"(.+)_tau(.+)_{re.escape(tag)}_seed(\d+)$")
    found: dict[tuple[str, str, int], dict[str, str]] = {}
    if not results_dir.is_dir():
        raise FileNotFoundError(results_dir)
    for eval_path in sorted(results_dir.glob("*/eval.csv")):
        match = pat.match(eval_path.parent.name)
        if not match:
            continue
        env, tau_s, seed = match.group(1), match.group(2), int(match.group(3))
        if env not in environments or seed not in seeds:
            continue
        if not (eval_path.parent / "params_1000000.pkl").is_file():
            continue
        with eval_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or rows[-1].get("step") != "1000000":
            continue
        last = rows[-1]
        first = _finite("d4rl_score", last["d4rl_score"])
        deploy = _finite(score_col, last[score_col])
        found[(env, _tau_key(float(tau_s)), seed)] = {
            "family": family,
            "K": str(k),
            "integrator": integrator,
            "environment": env,
            "T": _tau_key(float(tau_s)),
            "seed": str(seed),
            "tag": eval_path.parent.name,
            "score_column": score_col,
            "target_d4rl": f"{first}",
            "deployment_d4rl": f"{deploy}",
            "delta_endpoint_minus_first": f"{deploy - first}",
            "host_run_dir": str(eval_path.parent.resolve()),
        }
    expected = [
        (env, _tau_key(tau), seed)
        for env in environments
        for tau in taus
        for seed in seeds
    ]
    missing = [key for key in expected if key not in found]
    if missing:
        raise ValueError(f"{results_dir} missing {len(missing)} cells, e.g. {missing[:4]}")
    return [found[key] for key in expected]


def recover_s23(family: str, k: int, integrator: str, results_dir: Path, tag: str) -> list[dict[str, str]]:
    if not results_dir.is_dir():
        return []
    return recover_grid(family, k, integrator, results_dir, tag, SEEDS_S23)


def recover_p0_i4() -> list[dict[str, str]]:
    if not P0_ALL180.is_file():
        raise FileNotFoundError(P0_ALL180)
    with P0_ALL180.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        if row["condition"] != "bar_p4":
            continue
        first = _finite("target_d4rl", row["target_d4rl"])
        deploy = _finite("deployment_d4rl", row["deployment_d4rl"])
        out.append(
            {
                "family": "p0_contemporaneous_i4",
                "K": "4",
                "integrator": "implicit",
                "environment": row["environment"],
                "T": row["T"],
                "seed": row["seed"],
                "tag": row["tag"],
                "score_column": row["score_column"],
                "target_d4rl": row["target_d4rl"],
                "deployment_d4rl": row["deployment_d4rl"],
                "delta_endpoint_minus_first": f"{deploy - first}",
                "host_run_dir": row["host_run_dir"],
            }
        )
    if len(out) != 90:
        raise ValueError(f"expected 90 P0 I4 rows, got {len(out)}")
    return out


def inventory_k1() -> dict[str, object]:
    root = RESULTS / "mpi1_s23"
    present = []
    if root.is_dir():
        pat = re.compile(r"(.+)_tau(.+)_mpi1_seed(\d+)$")
        for eval_path in root.glob("*/eval.csv"):
            match = pat.match(eval_path.parent.name)
            if not match:
                continue
            seed = int(match.group(3))
            if seed not in SEEDS_LOCAL:
                continue
            if not (eval_path.parent / "params_1000000.pkl").is_file():
                continue
            with eval_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or rows[-1].get("step") != "1000000":
                continue
            present.append(
                {
                    "environment": match.group(1),
                    "T": _tau_key(float(match.group(2))),
                    "seed": seed,
                }
            )
    present_taus = sorted({float(row["T"]) for row in present})
    return {
        "local_family": "historical_k1_s23",
        "n_present": len(present),
        "seeds_present": list(SEEDS_LOCAL),
        "seeds_absent": list(SEEDS_ABSENT),
        "present_taus": present_taus,
        "p0c_requested_h": list(P0_C_K1_TAUS),
        "p0c_requested_present": [tau for tau in P0_C_K1_TAUS if tau in set(present_taus)],
        "p0c_requested_missing": [tau for tau in P0_C_K1_TAUS if tau not in set(present_taus)],
        "note": (
            "K1 first actor is the deployment actor; eval.csv has d4rl_score only. "
            "Present s23 taus are the I4 total-budget grid, not the I4 h=T/4 grid."
        ),
    }


def _stats(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {"n": 0}
    deltas = [float(row["delta_endpoint_minus_first"]) for row in rows]
    return {
        "n": len(rows),
        "mean_endpoint_minus_first": statistics.mean(deltas),
        "n_positive": sum(delta > 0.0 for delta in deltas),
        "n_negative": sum(delta < 0.0 for delta in deltas),
        "n_zero": sum(delta == 0.0 for delta in deltas),
    }


def _task_region_stats(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], _region(float(row["T"])))].append(
            float(row["delta_endpoint_minus_first"])
        )
    return [
        {
            "environment": env,
            "region": region,
            "n": len(values),
            "mean_endpoint_minus_first": statistics.mean(values),
            "n_positive": sum(value > 0.0 for value in values),
        }
        for (env, region), values in sorted(grouped.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _optional_finite(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    value = float(raw)
    if not math.isfinite(value):
        return None
    return value


def inventory_historical_i4(
    results_dir: Path,
    environments: tuple[str, ...] = ENVIRONMENTS,
    taus: tuple[float, ...] = TAUS,
    seeds: tuple[int, ...] = SEEDS_I4_HISTORICAL,
) -> dict[str, object]:
    pat = re.compile(r"(.+)_tau(.+)_mpi4_seed(\d+)$")
    endpoint: dict[tuple[str, str, int], float] = {}
    first: dict[tuple[str, str, int], float] = {}
    if not results_dir.is_dir():
        raise FileNotFoundError(results_dir)
    for eval_path in sorted(results_dir.glob("*/eval.csv")):
        match = pat.match(eval_path.parent.name)
        if not match:
            continue
        env, tau_s, seed = match.group(1), match.group(2), int(match.group(3))
        if env not in environments or seed not in seeds:
            continue
        if not (eval_path.parent / "params_1000000.pkl").is_file():
            continue
        with eval_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or rows[-1].get("step") != "1000000":
            continue
        last = rows[-1]
        deploy = _finite("d4rl_pi4", last["d4rl_pi4"])
        key = (env, _tau_key(float(tau_s)), seed)
        endpoint[key] = deploy
        maybe_first = _optional_finite(last.get("d4rl_score"))
        if maybe_first is not None:
            first[key] = maybe_first
    expected = [
        (env, _tau_key(tau), seed)
        for env in environments
        for tau in taus
        for seed in seeds
    ]
    missing_endpoint = [key for key in expected if key not in endpoint]
    if missing_endpoint:
        raise ValueError(
            f"{results_dir} missing {len(missing_endpoint)} endpoints, e.g. {missing_endpoint[:4]}"
        )
    return {
        "family": "historical_i4_504",
        "results_dir": str(results_dir.resolve()),
        "n_expected": len(expected),
        "n_endpoint": len(endpoint),
        "n_first_actor": len(first),
        "n_missing_first_actor": sum(1 for key in expected if key not in first),
        "first_actor_present": [
            {"environment": env, "T": tau, "seed": seed}
            for env, tau, seed in expected
            if (env, tau, seed) in first
        ],
        "note": (
            "Historical mpi4_norm eval.csv stores d4rl_pi4 at 1M but leaves "
            "d4rl_score empty on almost every cell. First-actor recovery from "
            "eval.csv is therefore impossible; re-evaluation of both policies "
            "from params_1000000.pkl would be required. Do not fill those "
            "blanks from P0 or s23 tables."
        ),
    }


def recover_ext_csv(lab_root: Path) -> dict[str, object]:
    i2 = recover_grid(
        "historical_score_grid_s01",
        2,
        "implicit",
        lab_root / "results_mpi2",
        "mpi2",
        SEEDS_S01,
    )
    i3 = recover_grid(
        "historical_score_grid_s01",
        3,
        "implicit",
        lab_root / "results_mpi3",
        "mpi3",
        SEEDS_S01,
    )
    i4_cov = inventory_historical_i4(lab_root / "results" / "mpi4_norm")
    if len(i2) != 252 or len(i3) != 252:
        raise ValueError(f"ext_csv counts i2={len(i2)} i3={len(i3)}")
    families = {row["family"] for row in i2 + i3}
    if families != {"historical_score_grid_s01"}:
        raise ValueError(f"unexpected families: {sorted(families)}")
    return {"i2_s01": i2, "i3_s01": i3, "i4_historical_504_coverage": i4_cov}


def _write_s23_archive(out: Path) -> dict[str, object]:
    i2 = recover_s23("historical_score_grid_s23", 2, "implicit", RESULTS / "mpi2_s23", "mpi2")
    i3 = recover_s23("historical_score_grid_s23", 3, "implicit", RESULTS / "mpi3_s23", "mpi3")
    exp3 = recover_s23("explicit_s23_separate", 3, "explicit", RESULTS / "exp3_s23", "exp3")
    i4_p0 = recover_p0_i4()
    k1 = inventory_k1()

    _write_csv(out / "i2_s23_first_endpoint.csv", i2)
    _write_csv(out / "i3_s23_first_endpoint.csv", i3)
    _write_csv(out / "explicit_i3_s23_first_endpoint.csv", exp3)
    _write_csv(out / "i4_p0_first_endpoint.csv", i4_p0)

    inclusion = {
        "frozen_at": datetime.now(KST).isoformat(timespec="seconds"),
        "no_reevaluation": True,
        "no_training": True,
        "included": {
            "i2_historical_s23": "9 task x 14 T x seeds {2,3} = 252",
            "i3_historical_s23": "9 task x 14 T x seeds {2,3} = 252",
            "i4_p0_contemporaneous": "9 task x {4,7,10,14,20} x seeds {0,1} = 90 BAR-P4 only",
        },
        "excluded_or_separate": {
            "i2_i3_seeds_0_1": "not on ext_csh; live on ext_csv",
            "i4_historical_504": "no mpi4_s23 tree on this host",
            "explicit_i3_s23": "available 252, stored separately, not mixed into implicit I3",
            "two_actor_p4": "P0 control, not I-MART",
        },
        "priority_regions": {
            "plateau": sorted(PLATEAU),
            "transition": sorted(TRANSITION),
            "tail": sorted(TAIL),
        },
    }
    receipt = {
        "built_at": inclusion["frozen_at"],
        "method": "cpu_eval_csv_recovery",
        "evaluation_contract": {
            "target_d4rl": "d4rl_score (online first actor)",
            "i2_deployment": "d4rl_pi2",
            "i3_deployment": "d4rl_pi3",
            "i4_p0_deployment": "d4rl_pi4",
        },
        "inclusion": inclusion,
        "k1_inventory": k1,
        "i2": _stats(i2),
        "i3": _stats(i3),
        "explicit_i3": _stats(exp3),
        "i4_p0": _stats(i4_p0),
        "i2_by_task_region": _task_region_stats(i2),
        "i3_by_task_region": _task_region_stats(i3),
        "i4_p0_by_task_region": _task_region_stats(i4_p0),
        "hashes": {
            "i2": sha256_file(out / "i2_s23_first_endpoint.csv"),
            "i3": sha256_file(out / "i3_s23_first_endpoint.csv"),
            "explicit_i3": sha256_file(out / "explicit_i3_s23_first_endpoint.csv"),
            "i4_p0": sha256_file(out / "i4_p0_first_endpoint.csv"),
        },
        "note": (
            "Extraction gain is within-run endpoint minus first actor. "
            "It is not superiority over an independently trained deployment actor."
        ),
    }
    (out / "INCLUSION_GRID.json").write_text(
        json.dumps(inclusion, indent=2) + "\n", encoding="utf-8"
    )
    (out / "K1_INVENTORY.json").write_text(
        json.dumps(k1, indent=2) + "\n", encoding="utf-8"
    )
    (out / "SUMMARY.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def _write_ext_csv_archive(out: Path, lab_root: Path) -> dict[str, object]:
    tables = recover_ext_csv(lab_root)
    _write_csv(out / "i2_s01_first_endpoint.csv", tables["i2_s01"])
    _write_csv(out / "i3_s01_first_endpoint.csv", tables["i3_s01"])
    i4_cov = tables["i4_historical_504_coverage"]
    inclusion = {
        "host": "ext_csv",
        "frozen_at": datetime.now(KST).isoformat(timespec="seconds"),
        "no_reevaluation": True,
        "no_training": True,
        "do_not_mix_with_s23_or_p0": True,
        "not_a_four_seed_headline": True,
        "included": {
            "i2_historical_s01": "9 task x 14 T x seeds {0,1} = 252",
            "i3_historical_s01": "9 task x 14 T x seeds {0,1} = 252",
        },
        "excluded_or_separate": {
            "i2_i3_seeds_2_3": "ext_csh s23 tables; do not concatenate",
            "i4_p0_contemporaneous": "P0 BAR-P4 90-cell table; different family",
            "i4_historical_504_first_actor": (
                "eval.csv has finite d4rl_pi4 on 504/504 but finite d4rl_score "
                f"on only {i4_cov['n_first_actor']}/504; no first/endpoint table written"
            ),
            "explicit_integrator": "not recovered here",
            "two_actor_p4": "P0 control, not I-MART",
        },
        "priority_regions": {
            "plateau": sorted(PLATEAU),
            "transition": sorted(TRANSITION),
            "tail": sorted(TAIL),
        },
        "lab_root": str(lab_root.resolve()),
    }
    receipt = {
        "built_at": inclusion["frozen_at"],
        "host": "ext_csv",
        "method": "cpu_eval_csv_recovery",
        "evaluation_contract": {
            "target_d4rl": "d4rl_score (online first actor)",
            "i2_deployment": "d4rl_pi2",
            "i3_deployment": "d4rl_pi3",
            "i4_historical_deployment": "d4rl_pi4",
        },
        "inclusion": inclusion,
        "i2_s01": _stats(tables["i2_s01"]),
        "i3_s01": _stats(tables["i3_s01"]),
        "i4_historical_504_coverage": i4_cov,
        "i2_s01_by_task_region": _task_region_stats(tables["i2_s01"]),
        "i3_s01_by_task_region": _task_region_stats(tables["i3_s01"]),
        "hashes": {
            "i2_s01": sha256_file(out / "i2_s01_first_endpoint.csv"),
            "i3_s01": sha256_file(out / "i3_s01_first_endpoint.csv"),
        },
        "note": (
            "Extraction gain is within-run endpoint minus first actor. "
            "Do not mix with s23 or P0 families or treat this as a four-seed headline. "
            "Historical I4 first-actor columns are not recovered from empty eval fields."
        ),
    }
    (out / "EXT_CSV_INCLUSION_GRID.json").write_text(
        json.dumps(inclusion, indent=2) + "\n", encoding="utf-8"
    )
    (out / "I4_HISTORICAL_504_COVERAGE.json").write_text(
        json.dumps(i4_cov, indent=2) + "\n", encoding="utf-8"
    )
    (out / "EXT_CSV_SUMMARY.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARCHIVE)
    parser.add_argument(
        "--host",
        choices=("s23", "ext_csv"),
        default="s23",
        help="s23 writes ext_csh tables; ext_csv writes unmixed s01/I4-504 tables only",
    )
    parser.add_argument(
        "--lab-root",
        type=Path,
        default=DEFAULT_EXT_CSV_LAB,
        help="ext_csv results root containing results_mpi2, results_mpi3, results/mpi4_norm",
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.host == "ext_csv":
        receipt = _write_ext_csv_archive(out, args.lab_root)
        print(
            json.dumps(
                {
                    "i2_s01": receipt["i2_s01"],
                    "i3_s01": receipt["i3_s01"],
                    "i4_historical_504_coverage": {
                        k: receipt["i4_historical_504_coverage"][k]
                        for k in (
                            "n_expected",
                            "n_endpoint",
                            "n_first_actor",
                            "n_missing_first_actor",
                        )
                    },
                },
                indent=2,
            )
        )
        return 0
    receipt = _write_s23_archive(out)
    print(json.dumps({k: receipt[k] for k in ("i2", "i3", "explicit_i3", "i4_p0", "k1_inventory")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
