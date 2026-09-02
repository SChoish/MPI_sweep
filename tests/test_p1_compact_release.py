from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.diagnostics.verify_p1_compact_release import (
    COMMON_SCOPE,
    verify_p1_target_value_compact,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sweep_results" / "diagnostics" / "p1_target_value_audit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_release(tmp_path: Path) -> tuple[Path, Path]:
    fake_repo = tmp_path / "repo"
    diagnostics = fake_repo / "sweep_results" / "diagnostics"
    compact = diagnostics / "p1_target_value_audit"
    compact.parent.mkdir(parents=True)
    shutil.copytree(SOURCE, compact)
    scripts = fake_repo / "scripts" / "diagnostics"
    scripts.mkdir(parents=True)
    for name in (
        "run_p1_target_value_audit.py",
        "verify_p1_target_value_audit.py",
    ):
        shutil.copyfile(ROOT / "scripts" / "diagnostics" / name, scripts / name)
    return diagnostics, compact


def _mutate_csv(compact: Path, filename: str, mutate) -> None:
    path = compact / filename
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = compact / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"][filename] = _sha256(path)
    _write_json(manifest_path, manifest)
    receipt_path = compact / "VERIFY.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = _sha256(manifest_path)
    _write_json(receipt_path, receipt)


def test_p1_compact_release_recomputes_promoted_claims(tmp_path: Path):
    diagnostics, _ = _copy_release(tmp_path)
    report = verify_p1_target_value_compact(diagnostics)
    assert report["p3"]["target"] == (88, 0.6503870640408389)
    assert report["p4"]["final"] == (44, 1.0252651612633423)
    assert report["p3"]["common"]["task_medians_below_one"] == 9
    assert report["p4"]["positive_slack_decreases"] == (205, 206)
    assert report["td3_critic_explosions"] == 31


def test_p1_compact_rejects_geometry_ratio_tamper_with_rebound_hashes(
    tmp_path: Path,
):
    diagnostics, compact = _copy_release(tmp_path)

    def mutate(rows: list[dict[str, str]]) -> None:
        row = next(item for item in rows if item["method"] == "p3")
        row["final_target_ratio"] = str(float(row["final_target_ratio"]) + 0.1)

    _mutate_csv(compact, "same_next_state_geometry.csv", mutate)
    with pytest.raises(AssertionError, match="final/target ratio mismatch"):
        verify_p1_target_value_compact(diagnostics)


def test_p1_compact_rejects_common_value_tamper_with_rebound_hashes(
    tmp_path: Path,
):
    diagnostics, compact = _copy_release(tmp_path)

    def mutate(rows: list[dict[str, str]]) -> None:
        row = next(
            item
            for item in rows
            if item["method"] == "p3" and item["critic_scope"] == COMMON_SCOPE
        )
        rms = float(row["delta_y_rms"]) * 1.5
        no_noise = float(row["delta_y_rms_no_noise"])
        row["delta_y_rms"] = str(rms)
        row["smoothing_delta_y_rms_change"] = str(rms - no_noise)
        row["smoothing_delta_y_rms_ratio"] = str(rms / no_noise)

    _mutate_csv(compact, "td_target_value.csv", mutate)
    with pytest.raises(AssertionError, match="method contrast mismatch"):
        verify_p1_target_value_compact(diagnostics)


def test_p1_compact_rejects_common_critic_fingerprint_tamper(
    tmp_path: Path,
):
    diagnostics, compact = _copy_release(tmp_path)

    def mutate(rows: list[dict[str, str]]) -> None:
        row = next(
            item
            for item in rows
            if item["method"] == "p4" and item["critic_scope"] == COMMON_SCOPE
        )
        row["critic_params_sha256"] = "f" * 64

    _mutate_csv(compact, "td_target_value.csv", mutate)
    with pytest.raises(AssertionError, match="common-TD3 critic pairing mismatch"):
        verify_p1_target_value_compact(diagnostics)


def test_p1_compact_rejects_residual_slack_tamper_with_rebound_hashes(
    tmp_path: Path,
):
    diagnostics, compact = _copy_release(tmp_path)

    def mutate(rows: list[dict[str, str]]) -> None:
        row = rows[0]
        row["slack_before"] = str(float(row["slack_before"]) + 0.01)

    _mutate_csv(compact, "comparator_residuals.csv", mutate)
    with pytest.raises(AssertionError, match="slack-before identity mismatch"):
        verify_p1_target_value_compact(diagnostics)
