from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from scripts.diagnostics.verify_p0_score_merge import (
    main,
    sha256_file,
    verify_archive,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "sweep_results" / "diagnostics" / "p0_bar_p4_vs_two_actor_p4"
TAIL = (
    ROOT
    / "sweep_results"
    / "diagnostics"
    / "hosts"
    / "ext_csh"
    / "p0_manifest_tail90"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copy_bundle(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "p0_bar_p4_vs_two_actor_p4"
    tail = tmp_path / "p0_manifest_tail90"
    shutil.copytree(ARCHIVE, archive)
    shutil.copytree(TAIL, tail)
    return archive, tail


def _rewrite_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _refresh_merge_payload_hash(archive: Path, relative: str) -> None:
    path = archive / "MERGE_MANIFEST.json"
    merge = json.loads(path.read_text(encoding="utf-8"))
    merge["files"][relative] = sha256_file(archive / relative)
    _write_json(path, merge)


def test_released_p0_score_merge_passes_integrity_but_not_admissibility(capsys):
    result = verify_archive(ARCHIVE, TAIL)
    assert result["compact_integrity"] == "PASS"
    assert result["scientific_admissibility"] == "NOT_ADMISSIBLE"
    assert result["manuscript_primary_promotion_allowed"] is False
    assert result["runs_verified"] == 180
    assert result["pairs_verified"] == 90
    assert result["optimizer_diverged_runs_retained"] == 5
    assert result["outcome_label"] == "unresolved"
    assert result["task_equal_mean"] == pytest.approx(-2.307552470498189)
    assert result["task_bootstrap_95_interval"] == pytest.approx(
        [-7.247601909110738, 2.4117874955787904]
    )

    assert main(["--archive-dir", str(ARCHIVE), "--tail-dir", str(TAIL)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["compact_integrity"] == "PASS"
    assert output["scientific_admissibility"] == "NOT_ADMISSIBLE"


def test_p0_score_merge_verifier_does_not_import_p0_analyzer_or_launcher():
    source = (
        ROOT / "scripts" / "diagnostics" / "verify_p0_score_merge.py"
    ).read_text(encoding="utf-8")
    assert "import analyze_p0_bar_two_actor_p4" not in source
    assert "from scripts.diagnostics.analyze_p0_bar_two_actor_p4" not in source
    assert "import run_p0_bar_two_actor_p4" not in source
    assert "from scripts.experiments.run_p0_bar_two_actor_p4" not in source


def test_p0_score_merge_rejects_rehashed_pair_arithmetic_tamper(tmp_path: Path):
    archive, tail = _copy_bundle(tmp_path)
    path = archive / "paired_scores.csv"
    rows = _read_csv(path)
    rows[0]["delta_bar_minus_two_actor"] = str(
        float(rows[0]["delta_bar_minus_two_actor"]) + 1.0
    )
    _rewrite_csv(path, rows)
    _refresh_merge_payload_hash(archive, "paired_scores.csv")
    with pytest.raises(AssertionError, match="paired delta_bar_minus_two_actor"):
        verify_archive(archive, tail)


def test_p0_score_merge_rejects_rehashed_host_assignment_tamper(tmp_path: Path):
    archive, tail = _copy_bundle(tmp_path)
    path = archive / "merged_final_scores.csv"
    rows = _read_csv(path)
    rows[0]["host_shard"] = "ext_csh"
    _rewrite_csv(path, rows)
    _refresh_merge_payload_hash(archive, "merged_final_scores.csv")
    with pytest.raises(AssertionError, match="merged row mismatch"):
        verify_archive(archive, tail)


def test_p0_score_merge_rejects_rehashed_summary_arithmetic_tamper(tmp_path: Path):
    archive, tail = _copy_bundle(tmp_path)
    path = archive / "SUMMARY.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["primary"]["task_equal_mean"] += 1.0
    _write_json(path, summary)
    _refresh_merge_payload_hash(archive, "SUMMARY.json")
    with pytest.raises(AssertionError, match="task_equal_mean"):
        verify_archive(archive, tail)


def test_p0_score_merge_rejects_diverged_key_erasure(tmp_path: Path):
    archive, tail = _copy_bundle(tmp_path)
    path = archive / "SUMMARY.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["merge"]["optimizer_diverged_tail_runs"].pop()
    _write_json(path, summary)
    _refresh_merge_payload_hash(archive, "SUMMARY.json")
    with pytest.raises(AssertionError, match="five retained optimizer-diverged"):
        verify_archive(archive, tail)


def test_p0_score_merge_rejects_dependency_identity_laundering(tmp_path: Path):
    archive, tail = _copy_bundle(tmp_path)
    path = archive / "MERGE_MANIFEST.json"
    merge = json.loads(path.read_text(encoding="utf-8"))
    merge["scientific_identity"]["resolved_dependency_snapshot_equal"] = True
    _write_json(path, merge)
    with pytest.raises(AssertionError, match="scientific-identity flag mismatch"):
        verify_archive(archive, tail)


def test_p0_score_merge_rejects_admissibility_laundering(tmp_path: Path):
    archive, tail = _copy_bundle(tmp_path)
    path = archive / "STATUS.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    status["scientific_admissible_under_frozen_p0_contract"] = True
    status["manuscript_primary_promotion_allowed"] = True
    _write_json(path, status)
    with pytest.raises(AssertionError, match="integrity/admissibility semantics"):
        verify_archive(archive, tail)


def test_p0_score_merge_cli_failure_remains_not_admissible(
    tmp_path: Path, capsys
):
    archive, tail = _copy_bundle(tmp_path)
    with (tail / "scores.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert main(["--archive-dir", str(archive), "--tail-dir", str(tail)]) == 1
    failure = json.loads(capsys.readouterr().err)
    assert failure["compact_integrity"] == "FAIL"
    assert failure["scientific_admissibility"] == "NOT_ADMISSIBLE"
