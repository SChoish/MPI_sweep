from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.diagnostics.package_p2_relu_residence_final import (
    CATEGORIES,
    COMPACT_SCHEMA,
    ENVIRONMENTS,
    K_VALUES,
    N_STATES,
    PROTOCOL,
    RAW_STATUS,
    SEEDS,
    T_VALUES,
    canonical_sha256,
    package_verified_bundle,
    sha256_file,
)
from scripts.verify_release_results import verify_p2_relu_residence_compact


ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMPACT = ROOT / "sweep_results" / "diagnostics" / "p2_relu_residence_final"
LEGACY_RAW = Path("/home/ext_csv/mpi_sweep_lab/p2-relu-final-20260902")
SOURCE_NAMES = (
    "run_p2_relu_residence.py",
    "verify_p2_relu_residence.py",
    "verify_p2_relu_residence_final.py",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _aggregate(
    curves: dict[tuple[str, int], list[dict]], keys: list[tuple[str, int]]
) -> list[dict]:
    rows = []
    for index, template in enumerate(next(iter(curves.values()))):
        categories = {
            name: sum(int(curves[key][index]["categories"][name]) for key in keys)
            for name in CATEGORIES
        }
        n_total = len(keys) * N_STATES
        rows.append(
            {
                "horizon": float(template["horizon"]),
                "n_resident": categories["resident"],
                "n_total": n_total,
                "resident_fraction": categories["resident"] / n_total,
                "categories": categories,
            }
        )
    return rows


def _make_summary() -> dict:
    summary = json.loads((LEGACY_COMPACT / "SUMMARY.json").read_text())
    summary.update(
        {
            "protocol": PROTOCOL,
            "status": RAW_STATUS,
            "scientific_admissible": False,
            "scientific_admissible_when_verified": True,
        }
    )
    for run in summary["per_run"]:
        run.update(
            {
                "protocol": PROTOCOL,
                "status": "run_complete_pending_verification",
                "scientific_admissible": False,
                "scientific_admissible_when_verified": True,
            }
        )
    curves = {
        (run["environment"], int(run["seed"])): deepcopy(run["survival"])
        for run in summary["per_run"]
    }
    all_keys = sorted(curves)
    task = {
        environment: _aggregate(
            curves, [(environment, seed) for seed in SEEDS]
        )
        for environment in ENVIRONMENTS
    }
    family = {
        name: _aggregate(
            curves,
            [key for key in all_keys if key[0].split("-", 1)[0] == name],
        )
        for name in ("hopper", "walker2d")
    }
    task_equal = []
    for index, template in enumerate(next(iter(task.values()))):
        task_equal.append(
            {
                "horizon": float(template["horizon"]),
                "n_tasks": len(ENVIRONMENTS),
                "resident_fraction": statistics.fmean(
                    task[environment][index]["resident_fraction"]
                    for environment in ENVIRONMENTS
                ),
                "category_fractions": {
                    name: statistics.fmean(
                        task[environment][index]["categories"][name]
                        / task[environment][index]["n_total"]
                        for environment in ENVIRONMENTS
                    )
                    for name in CATEGORIES
                },
            }
        )
    summary["pooled_survival"] = _aggregate(curves, all_keys)
    summary["task_survival"] = task
    summary["task_equal_survival"] = task_equal
    summary["family_survival"] = family
    return summary


def _make_raw_v2(tmp_path: Path, *, alias_tamper: bool = False) -> tuple[Path, Path]:
    fake_repo = tmp_path / "repo"
    source_dir = fake_repo / "scripts" / "diagnostics"
    source_dir.mkdir(parents=True)
    source_hashes = {}
    for name in SOURCE_NAMES:
        source = ROOT / "scripts" / "diagnostics" / name
        destination = source_dir / name
        shutil.copyfile(source, destination)
        source_hashes[name] = sha256_file(destination)

    raw = tmp_path / "raw-v2"
    raw.mkdir()
    for environment in ENVIRONMENTS:
        for seed in SEEDS:
            relative = f"{environment}_seed{seed}/residence_cells.csv"
            destination = raw / relative
            destination.parent.mkdir(parents=True)
            shutil.copyfile(LEGACY_COMPACT / relative, destination)

    if alias_tamper:
        path = raw / "hopper-medium-v2_seed0" / "residence_cells.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        assert fields is not None
        target = next(row for row in rows if row["T"] == "0.05" and row["K"] == "2")
        target["step_resident"] = str(int(target["step_resident"]) - 1)
        target["step_relu_cross"] = str(int(target["step_relu_cross"]) + 1)
        target["step_residence_fraction"] = str(int(target["step_resident"]) / N_STATES)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    design = json.loads((LEGACY_COMPACT / "FINAL_DESIGN_LOCK.json").read_text())
    design.update(
        {
            "protocol": PROTOCOL,
            "scientific_admissible_when_verified": True,
            "grid": {"T": list(T_VALUES), "K": list(K_VALUES)},
            "environments": list(ENVIRONMENTS),
            "seeds": list(SEEDS),
            "n_runs": 12,
            "n_states_per_run": N_STATES,
            "source_sha256": source_hashes,
            "coverage_gate": None,
            "learned_slope": None,
        }
    )
    design.pop("design_sha256", None)
    design["design_sha256"] = canonical_sha256(design)
    _write_json(raw / "FINAL_DESIGN_LOCK.json", design)

    summary = _make_summary()
    _write_json(raw / "SUMMARY.json", summary)
    status = {
        "protocol": PROTOCOL,
        "status": RAW_STATUS,
        "scientific_admissible": False,
        "scientific_admissible_when_verified": True,
    }
    _write_json(raw / "STATUS.json", status)
    revision = "a" * 40
    artifacts = {
        "FINAL_DESIGN_LOCK.json": sha256_file(raw / "FINAL_DESIGN_LOCK.json"),
        "SUMMARY.json": sha256_file(raw / "SUMMARY.json"),
    }
    for environment in ENVIRONMENTS:
        for seed in SEEDS:
            relative = f"{environment}_seed{seed}/residence_cells.csv"
            artifacts[relative] = sha256_file(raw / relative)
    manifest = {
        "protocol": PROTOCOL,
        "status": RAW_STATUS,
        "scientific_admissible": False,
        "scientific_admissible_when_verified": True,
        "design_sha256": design["design_sha256"],
        "source_sha256": source_hashes,
        "source_snapshots": {
            name: {"path": f"SOURCE_SNAPSHOT/{name}", "sha256": digest}
            for name, digest in source_hashes.items()
        },
        "git_provenance": {
            "git_dirty": False,
            "git_tracked_dirty": False,
            "git_status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
            "git_tracked_status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
            "head_matches_origin_main": True,
            "source_snapshot_is_durable_provenance": True,
            "git_revision": revision,
            "origin_main_revision": revision,
        },
        "runtime": {"backend": "cpu", "jax_enable_x64": True},
        "counts": {
            "runs": 12,
            "states_per_run": N_STATES,
            "anchors_total": 12 * N_STATES,
        },
        "artifacts": artifacts,
    }
    _write_json(raw / "MANIFEST.json", manifest)
    receipt = {
        "protocol": PROTOCOL,
        "status": "verified_complete",
        "pass": True,
        "scientific_admissible": True,
        "n_runs": 12,
        "n_states_per_run": N_STATES,
        "n_anchors_total": 12 * N_STATES,
        "independent_geometry_recompute_pass": True,
        "finite_difference_pass": True,
        "exit_bracket_pass": True,
        "pooled_survival_pass": True,
        "task_survival_pass": True,
        "task_equal_survival_pass": True,
        "aggregate_boundary_categories_pass": True,
        "family_survival_pass": True,
        "design_sha256": design["design_sha256"],
        "manifest_sha256": sha256_file(raw / "MANIFEST.json"),
        "verifier_sha256": source_hashes["verify_p2_relu_residence_final.py"],
    }
    _write_json(raw / "VERIFY.json", receipt)
    return raw, fake_repo


def test_packager_rejects_legacy_v1_bundle(tmp_path: Path):
    if not LEGACY_RAW.is_dir():
        pytest.skip("local legacy raw P2 bundle is unavailable")
    with pytest.raises(ValueError, match="final_v2"):
        package_verified_bundle(LEGACY_RAW, tmp_path / "compact")


def test_package_and_fresh_clone_release_verification(tmp_path: Path):
    raw, fake_repo = _make_raw_v2(tmp_path)
    compact = fake_repo / "sweep_results" / "diagnostics" / "p2_relu_residence_final"
    manifest = package_verified_bundle(raw, compact)
    assert manifest["schema_version"] == COMPACT_SCHEMA
    assert len(manifest["inventory"]) == 18
    verify_p2_relu_residence_compact(compact.parent)


def test_release_verifier_rejects_semantic_horizon_alias_tamper(tmp_path: Path):
    raw, fake_repo = _make_raw_v2(tmp_path, alias_tamper=True)
    compact = fake_repo / "sweep_results" / "diagnostics" / "p2_relu_residence_final"
    package_verified_bundle(raw, compact)
    with pytest.raises(AssertionError, match="horizon aliases differ"):
        verify_p2_relu_residence_compact(compact.parent)


def test_packager_rejects_dirty_v2_provenance(tmp_path: Path):
    raw, _ = _make_raw_v2(tmp_path)
    manifest_path = raw / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["git_provenance"]["git_dirty"] = True
    _write_json(manifest_path, manifest)
    receipt_path = raw / "VERIFY.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["manifest_sha256"] = sha256_file(manifest_path)
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="git_dirty=false"):
        package_verified_bundle(raw, tmp_path / "compact")
