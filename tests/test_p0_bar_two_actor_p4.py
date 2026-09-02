import copy
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics.analyze_p0_bar_two_actor_p4 import (
    _write_outputs,
    collect_pairs,
    decision_label,
    summarize,
)
from scripts.diagnostics.verify_p0_bar_two_actor_p4 import (
    main as verifier_main,
    verify,
)
import scripts.diagnostics.verify_p0_bar_two_actor_p4 as independent_verifier
from scripts.experiments.run_p0_bar_two_actor_p4 import (
    CONDITIONS,
    ENVIRONMENTS,
    T_VALUES,
    _write_create_only,
    build_manifest,
    dependency_snapshot,
    inspect_checkpoint,
    main as harness_main,
    normalization_sha256,
    prepare_jobs,
    source_identity_snapshot,
    validate_manifest_commands,
    validate_run_grid,
    verify_manifest_software,
)

TEST_DEPENDENCIES = dependency_snapshot()
TEST_SOURCE = {
    **source_identity_snapshot(),
    "tracked_worktree_clean": True,
}


def _manifest(tmp_path: Path) -> dict:
    norm = normalization_sha256(
        np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    )
    datasets = {
        environment: {
            "path": str(tmp_path / f"{environment}.hdf5"),
            "bytes": 1,
            "sha256": "0" * 64,
            "normalization": {"statistics_sha256": norm},
        }
        for environment in ENVIRONMENTS
    }
    return build_manifest(
        data_dir=tmp_path / "data",
        output_root=tmp_path / "results",
        python=Path(sys.executable),
        gpus=["0", "1"],
        slots_per_gpu=1,
        compilation_cache_dir=tmp_path / "cache",
        datasets=datasets,
        dependencies=copy.deepcopy(TEST_DEPENDENCIES),
        hardware={"gpus": [{"index": "0"}, {"index": "1"}]},
        source=copy.deepcopy(TEST_SOURCE),
    )


def _checkpoint_payload(run: dict, step: int) -> dict:
    actor_count = 4 if run["condition"] == "bar_p4" else 2
    return {
        "step": step,
        "rng": np.asarray([0, 1], dtype=np.uint32),
        "mean": np.zeros(2, dtype=np.float32),
        "std": np.ones(2, dtype=np.float32),
        "config": run["resolved_config"],
        "max_action": 1.0,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "actors_params": tuple(
            {"w": np.asarray([1.0 + index], dtype=np.float32)}
            for index in range(actor_count)
        ),
        "actors_steps": tuple(
            step // int(run["resolved_config"]["policy_freq"])
            for _ in range(actor_count)
        ),
        "critic_params": {"w": np.asarray([2.0], dtype=np.float32)},
        "critic_step": step,
        "target_actor_params": {"w": np.asarray([3.0], dtype=np.float32)},
        "target_critic_params": {"w": np.asarray([4.0], dtype=np.float32)},
        "actors_opt_states": tuple({} for _ in range(actor_count)),
        "critic_opt_state": {},
    }


def test_exact_p0_grid_and_commands(tmp_path: Path):
    manifest = _manifest(tmp_path)
    runs = manifest["runs"]

    assert len(runs) == 180
    assert len({run["key"] for run in runs}) == 180
    assert Counter(run["condition"] for run in runs) == {
        "bar_p4": 90,
        "two_actor_p4": 90,
    }
    assert {run["T"] for run in runs} == set(T_VALUES)
    for run in runs:
        command = run["command"]
        assert command[command.index("--mpi-steps") + 1] == "4"
        assert command[command.index("--method") + 1] == (
            "bar" if run["condition"] == "bar_p4" else "mcep"
        )
        assert command[command.index("--eval-episodes") + 1] == "10"
        assert command[command.index("--max-timesteps") + 1] == "1000000"
        assert "--no-resume" in command


def test_default_plan_has_no_filesystem_side_effects(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert harness_main([]) == 0
    assert list(tmp_path.iterdir()) == []


def test_freeze_rejects_a_different_python_before_snapshotting(tmp_path: Path):
    with pytest.raises(ValueError, match="interpreter running freeze"):
        harness_main(
            [
                "freeze",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--data-dir",
                str(tmp_path / "data"),
                "--output-root",
                str(tmp_path / "results"),
                "--python",
                str(tmp_path / "different-python"),
                "--compilation-cache-dir",
                str(tmp_path / "cache"),
            ]
        )


def test_manifest_rejects_deployment_score_and_command_drift(tmp_path: Path):
    manifest = _manifest(tmp_path)
    tampered_runs = copy.deepcopy(manifest["runs"])
    tampered_runs[0]["score_column"] = "d4rl_score"
    with pytest.raises(ValueError, match="tag/score contract drift"):
        validate_run_grid(tampered_runs)

    tampered_manifest = copy.deepcopy(manifest)
    command = tampered_manifest["runs"][0]["command"]
    command[command.index("--seed") + 1] = "7"
    with pytest.raises(ValueError, match="launch command drift"):
        validate_manifest_commands(tampered_manifest)

    tampered_source = copy.deepcopy(manifest)
    source_file = "scripts/diagnostics/analyze_p0_bar_two_actor_p4.py"
    tampered_source["source"]["files"][source_file] = "0" * 64
    with pytest.raises(RuntimeError, match="source revision or file hashes"):
        verify_manifest_software(tampered_source)

    tampered_software = copy.deepcopy(manifest)
    tampered_software["dependencies"]["python_version"] = "0.0"
    with pytest.raises(RuntimeError, match="dependency snapshot"):
        verify_manifest_software(tampered_software)


def test_resume_requires_exact_config_weights_and_normalization_hashes(tmp_path: Path):
    manifest = _manifest(tmp_path)
    run = manifest["runs"][0]
    checkpoint = Path(run["output_dir"]) / "params_250000.pkl"
    checkpoint.parent.mkdir(parents=True)
    payload = _checkpoint_payload(run, 250_000)
    with checkpoint.open("wb") as handle:
        pickle.dump(payload, handle)
    entry = inspect_checkpoint(checkpoint, run, manifest)
    jobs, completed = prepare_jobs(manifest, {"entries": [entry]})
    assert completed == 0
    assert len(jobs) == 180
    resumed = next(job for job in jobs if job.run["key"] == run["key"])
    assert resumed.resume_checkpoint == checkpoint
    assert resumed.command[-2:] == ["--restore-path", str(checkpoint.resolve())]

    tampered = {**entry, "weights_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="hashes changed"):
        prepare_jobs(manifest, {"entries": [tampered]})

    payload["critic_step"] = 249_999
    with checkpoint.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="critic_step"):
        inspect_checkpoint(checkpoint, run, manifest)

    payload["critic_step"] = 250_000
    payload["step"] = 250_000.5
    with checkpoint.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="Python integer"):
        inspect_checkpoint(checkpoint, run, manifest)

    payload["step"] = 250_000
    payload["actors_params"] = payload["actors_params"][:1]
    with checkpoint.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="actor count"):
        inspect_checkpoint(checkpoint, run, manifest)


@pytest.mark.parametrize(
    ("point", "low", "high", "expected"),
    [
        (3.0, 0.1, 4.0, "four_actor_procedure_supporting"),
        (-3.0, -4.0, -0.1, "two_actor_procedure_supporting"),
        (0.0, -2.9, 2.9, "author_band_comparable"),
        (2.0, -0.5, 4.0, "unresolved"),
    ],
)
def test_full_procedure_outcome_labels(point, low, high, expected):
    assert decision_label(point, low, high) == expected


def _write_raw_grid(manifest: dict) -> None:
    task_effect = {task: float(index - 4) for index, task in enumerate(ENVIRONMENTS)}
    for run in manifest["runs"]:
        out_dir = Path(run["output_dir"])
        out_dir.mkdir(parents=True)
        (out_dir / "config.json").write_text(
            json.dumps(run["resolved_config"]), encoding="utf-8"
        )
        control_score = 50.0 + 0.1 * int(run["T"]) + int(run["seed"])
        deployment_score = (
            control_score + task_effect[run["environment"]]
            if run["condition"] == "bar_p4"
            else control_score
        )
        fields = manifest["output_contract"][
            "bar_eval_columns"
            if run["condition"] == "bar_p4"
            else "two_actor_eval_columns"
        ]
        row = {field: 0 for field in fields}
        row.update({"step": 1_000_000, "return": -999.0, "d4rl_score": -999.0})
        row[run["score_column"]] = deployment_score
        with (out_dir / "eval.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        checkpoint_payload = _checkpoint_payload(run, 1_000_000)
        with (out_dir / "params_1000000.pkl").open("wb") as handle:
            pickle.dump(checkpoint_payload, handle)


def test_analysis_and_independent_verifier_detect_tampering(tmp_path: Path):
    manifest = _manifest(tmp_path)
    _write_raw_grid(manifest)
    manifest_path = tmp_path / "FROZEN_MANIFEST.json"
    manifest_digest = _write_create_only(manifest_path, manifest)

    pairs, artifact_hashes = collect_pairs(manifest)
    assert len(pairs) == 90
    assert all(pair["bar_score"] != -999.0 for pair in pairs)
    summary = summarize(pairs)
    assert summary["primary"]["outcome_label"] == "author_band_comparable"
    assert summary["resampling_unit"] == "nine fixed task variants"
    assert "three dynamics families" in summary["inference_scope"]
    assert set(summary["collapse_transitions"]["raw_run"]) == {
        "0", "10", "20", "30", "40"
    }

    analysis_dir = tmp_path / "analysis"
    _write_outputs(
        analysis_dir,
        summary,
        pairs,
        artifact_hashes,
        manifest_digest,
    )
    verify(manifest_path, analysis_dir)
    assert verifier_main(
        [
            "--manifest",
            str(manifest_path),
            "--analysis-dir",
            str(analysis_dir),
        ]
    ) == 0
    receipt_path = analysis_dir / "VERIFY.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "pass"
    assert receipt["run_artifacts_verified"] == 180
    assert receipt["paired_cells_verified"] == 90
    assert receipt["outcome_label"] == "author_band_comparable"
    receipt_digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert (
        receipt_path.with_suffix(".json.sha256")
        .read_text(encoding="utf-8")
        .split()[0]
        == receipt_digest
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        independent_verifier._write_receipt(receipt_path, {})

    checkpoint_records = json.loads(
        (analysis_dir / "FINAL_CHECKPOINTS.json").read_text(encoding="utf-8")
    )
    first_record = checkpoint_records[manifest["runs"][0]["key"]]
    assert {
        "step",
        "scientific_config_sha256",
        "weights_sha256",
        "normalization_sha256",
    }.issubset(first_record)
    source = Path(independent_verifier.__file__).read_text(encoding="utf-8")
    assert "from scripts.diagnostics.analyze_p0_bar_two_actor_p4" not in source
    assert "from scripts.experiments.run_p0_bar_two_actor_p4" not in source

    first_run = manifest["runs"][0]
    first_checkpoint = Path(first_run["output_dir"]) / "params_1000000.pkl"
    original_checkpoint = first_checkpoint.read_bytes()
    with first_checkpoint.open("rb") as handle:
        corrupted_payload = pickle.load(handle)
    corrupted_payload["critic_params"]["w"][0] = np.nan
    with first_checkpoint.open("wb") as handle:
        pickle.dump(corrupted_payload, handle)
    with pytest.raises(ValueError, match="nonfinite"):
        verify(manifest_path, analysis_dir)

    corrupted_payload = pickle.loads(original_checkpoint)
    del corrupted_payload["critic_step"]
    with first_checkpoint.open("wb") as handle:
        pickle.dump(corrupted_payload, handle)
    with pytest.raises(ValueError, match="checkpoint key schema mismatch"):
        verify(manifest_path, analysis_dir)
    first_checkpoint.write_bytes(original_checkpoint)

    with (analysis_dir / "paired_scores.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        verify(manifest_path, analysis_dir)
