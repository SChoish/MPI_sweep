#!/usr/bin/env python3
"""Create a self-contained compact release from a verified P2 final-v2 bundle.

The raw audit remains the source for geometry and finite-difference replay.  This
packager copies only the frozen design, aggregate metadata, verification receipt,
and the twelve per-run cell tables needed for fresh-clone release checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping


PROTOCOL = "p2_relu_residence_final_v2"
RAW_STATUS = "analysis_complete_pending_verification"
VERIFIED_STATUS = "verified_complete"
COMPACT_SCHEMA = "p2-relu-residence-compact-v1"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ENVIRONMENTS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-expert-v2",
)
SEEDS = (0, 1)
T_VALUES = (0.025, 0.05, 0.1, 0.2)
K_VALUES = (1, 2, 4, 8, 16)
N_STATES = 512
CATEGORIES = (
    "resident",
    "relu_cross",
    "box_cross",
    "tie_cross",
    "boundary_ambiguous",
    "boundary_touch",
    "nonfinite",
)
ROOT_PAYLOADS = (
    "FINAL_DESIGN_LOCK.json",
    "MANIFEST.json",
    "STATUS.json",
    "SUMMARY.json",
    "VERIFY.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def expected_cell_paths() -> tuple[str, ...]:
    return tuple(
        f"{environment}_seed{seed}/residence_cells.csv"
        for environment in ENVIRONMENTS
        for seed in SEEDS
    )


def _require_protocol(document: Mapping[str, Any], label: str) -> None:
    if document.get("protocol") != PROTOCOL:
        raise ValueError(f"{label} is not a {PROTOCOL} artifact")


def _validate_clean_provenance(provenance: Any) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError("raw MANIFEST git_provenance is missing")
    if provenance.get("git_dirty") is not False:
        raise ValueError("P2 final-v2 packaging requires recorded git_dirty=false")
    if provenance.get("git_tracked_dirty") is not False:
        raise ValueError("P2 final-v2 packaging requires a clean tracked worktree")
    if provenance.get("head_matches_origin_main") is not True:
        raise ValueError("P2 final-v2 packaging requires HEAD == origin/main")
    if (
        provenance.get("git_status_porcelain_sha256") != EMPTY_SHA256
        or provenance.get("git_tracked_status_porcelain_sha256") != EMPTY_SHA256
        or provenance.get("source_snapshot_is_durable_provenance") is not True
    ):
        raise ValueError("P2 final-v2 clean-provenance evidence is inconsistent")
    revision = provenance.get("git_revision")
    origin = provenance.get("origin_main_revision")
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or revision != origin
    ):
        raise ValueError("P2 final-v2 Git revision provenance is inconsistent")
    return provenance


def _validate_raw_bundle(raw_dir: Path) -> dict[str, Any]:
    for relative in (*ROOT_PAYLOADS, *expected_cell_paths()):
        if not (raw_dir / relative).is_file():
            raise FileNotFoundError(raw_dir / relative)

    design = read_json(raw_dir / "FINAL_DESIGN_LOCK.json")
    manifest = read_json(raw_dir / "MANIFEST.json")
    status = read_json(raw_dir / "STATUS.json")
    summary = read_json(raw_dir / "SUMMARY.json")
    receipt = read_json(raw_dir / "VERIFY.json")
    for label, document in (
        ("FINAL_DESIGN_LOCK", design),
        ("MANIFEST", manifest),
        ("STATUS", status),
        ("SUMMARY", summary),
        ("VERIFY", receipt),
    ):
        _require_protocol(document, label)

    for label, document in (
        ("MANIFEST", manifest),
        ("STATUS", status),
        ("SUMMARY", summary),
    ):
        if document.get("status") != RAW_STATUS:
            raise ValueError(f"{label} raw status is not verification-gated")
        if document.get("scientific_admissible") is not False:
            raise ValueError(f"{label} improperly self-admits before verification")
        if document.get("scientific_admissible_when_verified") is not True:
            raise ValueError(f"{label} is missing the verification gate")

    if (
        receipt.get("status") != VERIFIED_STATUS
        or receipt.get("pass") is not True
        or receipt.get("scientific_admissible") is not True
    ):
        raise ValueError("raw P2 final-v2 VERIFY receipt did not pass")
    if (
        int(receipt.get("n_runs", -1)) != len(ENVIRONMENTS) * len(SEEDS)
        or int(receipt.get("n_states_per_run", -1)) != N_STATES
        or int(receipt.get("n_anchors_total", -1))
        != len(ENVIRONMENTS) * len(SEEDS) * N_STATES
    ):
        raise ValueError("raw P2 final-v2 VERIFY counts drifted")
    for field in (
        "independent_geometry_recompute_pass",
        "finite_difference_pass",
        "exit_bracket_pass",
        "pooled_survival_pass",
        "task_survival_pass",
        "task_equal_survival_pass",
        "aggregate_boundary_categories_pass",
        "family_survival_pass",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"raw P2 verification gate failed: {field}")

    design_without_hash = dict(design)
    claimed_design_hash = design_without_hash.pop("design_sha256", None)
    if claimed_design_hash != canonical_sha256(design_without_hash):
        raise ValueError("FINAL_DESIGN_LOCK self-hash mismatch")
    if manifest.get("design_sha256") != claimed_design_hash:
        raise ValueError("MANIFEST design hash mismatch")
    if receipt.get("design_sha256") != claimed_design_hash:
        raise ValueError("VERIFY design hash mismatch")
    if receipt.get("manifest_sha256") != sha256_file(raw_dir / "MANIFEST.json"):
        raise ValueError("VERIFY does not bind the raw MANIFEST")

    provenance = _validate_clean_provenance(manifest.get("git_provenance"))
    runtime = manifest.get("runtime", {})
    if runtime.get("backend") != "cpu" or runtime.get("jax_enable_x64") is not True:
        raise ValueError("raw P2 final-v2 runtime is not CPU float64")
    if manifest.get("counts") != {
        "runs": len(ENVIRONMENTS) * len(SEEDS),
        "states_per_run": N_STATES,
        "anchors_total": len(ENVIRONMENTS) * len(SEEDS) * N_STATES,
    }:
        raise ValueError("raw P2 MANIFEST counts drifted")

    if design.get("environments") != list(ENVIRONMENTS):
        raise ValueError("P2 final environment order drifted")
    if design.get("seeds") != list(SEEDS):
        raise ValueError("P2 final seed grid drifted")
    if design.get("grid") != {"T": list(T_VALUES), "K": list(K_VALUES)}:
        raise ValueError("P2 final T/K grid drifted")
    if (
        int(design.get("n_runs", -1)) != len(ENVIRONMENTS) * len(SEEDS)
        or int(design.get("n_states_per_run", -1)) != N_STATES
        or design.get("coverage_gate") is not None
        or design.get("learned_slope") is not None
    ):
        raise ValueError("P2 final estimand/count contract drifted")

    source_hashes = manifest.get("source_sha256")
    if design.get("source_sha256") != source_hashes or not isinstance(
        source_hashes, dict
    ):
        raise ValueError("P2 design/MANIFEST source hashes differ")
    verifier_hash = source_hashes.get("verify_p2_relu_residence_final.py")
    if receipt.get("verifier_sha256") != verifier_hash:
        raise ValueError("VERIFY verifier source hash mismatch")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("raw P2 MANIFEST artifact inventory missing")
    for relative in ("FINAL_DESIGN_LOCK.json", "SUMMARY.json", *expected_cell_paths()):
        if artifacts.get(relative) != sha256_file(raw_dir / relative):
            raise ValueError(f"raw MANIFEST artifact hash mismatch: {relative}")

    if (
        summary.get("environments") != list(ENVIRONMENTS)
        or summary.get("seeds") != list(SEEDS)
        or int(summary.get("n_runs", -1)) != len(ENVIRONMENTS) * len(SEEDS)
        or int(summary.get("n_states_per_run", -1)) != N_STATES
        or int(summary.get("n_anchors_total", -1))
        != len(ENVIRONMENTS) * len(SEEDS) * N_STATES
        or summary.get("coverage_gate") is not None
        or summary.get("learned_slope") is not None
    ):
        raise ValueError("raw P2 SUMMARY grid or estimand drifted")
    run_keys = {
        (row.get("environment"), int(row.get("seed", -1)))
        for row in summary.get("per_run", [])
        if isinstance(row, dict)
    }
    expected_run_keys = {
        (environment, seed) for environment in ENVIRONMENTS for seed in SEEDS
    }
    if run_keys != expected_run_keys or len(summary.get("per_run", [])) != 12:
        raise ValueError("raw P2 SUMMARY per-run grid mismatch")

    expected_cells = {(total, k) for total in T_VALUES for k in K_VALUES}
    for relative in expected_cell_paths():
        with (raw_dir / relative).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        cells = {(float(row["T"]), int(row["K"])) for row in rows}
        if len(rows) != 20 or cells != expected_cells:
            raise ValueError(f"P2 compact source cell grid mismatch: {relative}")
        if any(int(row["n_states"]) != N_STATES for row in rows):
            raise ValueError(f"P2 compact source state count mismatch: {relative}")
        for row in rows:
            for prefix in ("full_", "step_"):
                total = sum(int(row[f"{prefix}{name}"]) for name in CATEGORIES)
                if total != N_STATES:
                    raise ValueError(f"P2 category denominator mismatch: {relative}")
            if not all(
                math.isfinite(float(row[field]))
                for field in ("T", "h", "full_residence_fraction", "step_residence_fraction")
            ):
                raise ValueError(f"P2 compact source contains nonfinite cells: {relative}")

    return {
        "design": design,
        "manifest": manifest,
        "summary": summary,
        "receipt": receipt,
        "provenance": provenance,
    }


def _readme(revision: str) -> str:
    return f"""# P2 learned-ReLU residence compact archive

This directory is the self-contained release view of the verified
`{PROTOCOL}` audit at source revision `{revision}`.

It contains the frozen design, verification-gated raw metadata, independent
verification receipt, aggregate summary, and all 12 per-run `residence_cells.csv`
tables. The release verifier reconstructs the exact 6-task x 2-seed x 4-T x
5-K grid, category denominators, repeated-horizon aliases, and per-run, task,
pooled, task-equal, and family summaries from these tables.

The raw state indices, geometry arrays, checkpoints, and datasets remain outside
the repository. `VERIFY.json` records the completed raw full-recomputation gate;
`COMPACT_MANIFEST.json` binds every file retained here. The compact archive does
not estimate a learned order slope or turn activation-region residence into a
return, global-proximal, semigroup, or live-chain claim.
"""


def package_verified_bundle(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    documents = _validate_raw_bundle(raw_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"compact output must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative in (*ROOT_PAYLOADS, *expected_cell_paths()):
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(raw_dir / relative, destination)

    revision = documents["provenance"]["git_revision"]
    readme_path = output_dir / "README.md"
    readme_path.write_text(_readme(revision), encoding="utf-8")

    payload_paths = (*ROOT_PAYLOADS, *expected_cell_paths(), "README.md")
    inventory = {
        relative: sha256_file(output_dir / relative) for relative in payload_paths
    }
    compact_manifest = {
        "schema_version": COMPACT_SCHEMA,
        "artifact": "p2_relu_residence_final_compact",
        "protocol": PROTOCOL,
        "status": "verified_compact",
        "scientific_admissible": True,
        "scope": (
            "compact cell-table and aggregate release view; raw geometry and "
            "finite-difference replay remain bound by VERIFY.json"
        ),
        "grid": {
            "environments": list(ENVIRONMENTS),
            "seeds": list(SEEDS),
            "T": list(T_VALUES),
            "K": list(K_VALUES),
            "runs": 12,
            "cells_per_run": 20,
            "states_per_run": N_STATES,
            "anchors_total": 12 * N_STATES,
        },
        "source_git": {
            "revision": revision,
            "origin_main_revision": documents["provenance"]["origin_main_revision"],
            "recorded_clean": True,
        },
        "raw_verification": {
            "status": documents["receipt"]["status"],
            "pass": documents["receipt"]["pass"],
            "scientific_admissible": documents["receipt"]["scientific_admissible"],
            "design_sha256": documents["receipt"]["design_sha256"],
            "manifest_sha256": documents["receipt"]["manifest_sha256"],
            "verifier_sha256": documents["receipt"]["verifier_sha256"],
        },
        "inventory": inventory,
        "payload_tree_sha256": canonical_sha256(inventory),
    }
    manifest_path = output_dir / "COMPACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(compact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return compact_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = package_verified_bundle(args.raw_dir, args.output_dir)
    print(
        "PASS P2 compact package: "
        f"{manifest['grid']['runs']} runs, "
        f"{manifest['grid']['runs'] * manifest['grid']['cells_per_run']} cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
