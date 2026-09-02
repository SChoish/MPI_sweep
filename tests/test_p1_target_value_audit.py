import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


DIAGNOSTICS = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

import run_p1_target_value_audit as p1  # noqa: E402
import verify_p1_target_value_audit as verify_p1  # noqa: E402


def _valid_verified_record(tmp_path: Path, method: str = "p3"):
    expected = next(
        record
        for record in verify_p1._expected_record_metadata().values()
        if record["method"] == method
    )
    config = {
        "env": expected["environment"],
        "seed": expected["seed"],
        "tau": expected["tau"],
        "mpi_steps": expected["hops"],
        **verify_p1.REQUIRED_CONFIG,
    }
    run_dir = (tmp_path / expected["run_name"]).resolve()
    digest = "a" * 64
    expected_actor_step = verify_p1.CHECKPOINT_STEP // int(config["policy_freq"])
    row = {
        **expected,
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / f"params_{verify_p1.CHECKPOINT_STEP}.pkl"),
        "config_path": str(run_dir / "config.json"),
        "eval_path": str(run_dir / "eval.csv"),
        "checkpoint_step": verify_p1.CHECKPOINT_STEP,
        "external_score_step": verify_p1.CHECKPOINT_STEP,
        "config": config,
        "config_sha256": hashlib.sha256(
            verify_p1._canonical_json_bytes(config)
        ).hexdigest(),
        "external_score": 50.0,
        "collapsed_lt20": False,
        "max_action": 1.0,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "discount": 0.99,
        "learning_rate": 3e-4,
        "resolved": True,
        "residual_supported": True,
        "residual_intervention": "posthoc_final_checkpoint_one_adam_step",
        "optimizer_reconstruction": {
            "optimizer": "optax.adam with imported training defaults",
            "learning_rate": 3e-4,
            "expected_actor_step": expected_actor_step,
            "actors": [
                {
                    "actor": index,
                    "actor_step": expected_actor_step,
                    "adam_count": expected_actor_step,
                    "optimizer_state_sha256": digest,
                }
                for index in range(1, int(expected["hops"]) + 1)
            ],
            "all_optimizer_states_sha256": digest,
        },
    }
    return expected, row

jax = p1.jax


def test_expected_grid_is_exact_270_without_discovery(tmp_path: Path):
    cells = p1.expected_cells(tmp_path)
    assert len(cells) == 270
    assert len({cell["key"] for cell in cells}) == 270
    assert {cell["method"] for cell in cells} == {"td3", "p3", "p4"}
    assert {cell["hops"] for cell in cells} == {1, 3, 4}
    first = cells[0]
    assert first["checkpoint_path"].endswith(
        "results_qnorm/halfcheetah-medium-v2_tau4_seed0/params_1000000.pkl"
    )
    p4 = next(
        cell
        for cell in cells
        if cell["method"] == "p4"
        and cell["environment"] == "walker2d-expert-v2"
        and cell["tau"] == 20
        and cell["seed"] == 1
    )
    assert p4["checkpoint_path"].endswith(
        "results/mpi4_norm/walker2d-expert-v2_tau20_mpi4_seed1/"
        "params_1000000.pkl"
    )
    assert p1.ENVIRONMENTS == verify_p1.ENVIRONMENTS
    assert p1.TAUS == verify_p1.TAUS
    assert p1.SEEDS == verify_p1.SEEDS


def test_fixed_config_contract_matches_verifier_and_rejects_training_drift(tmp_path: Path):
    assert p1.REQUIRED_CONFIG == verify_p1.REQUIRED_CONFIG
    assert p1.OPTIONAL_CONFIG_IF_PRESENT == verify_p1.OPTIONAL_CONFIG_IF_PRESENT
    cell = p1.expected_cells(tmp_path)[0]
    config = {
        "env": cell["environment"],
        "seed": cell["seed"],
        "tau": cell["tau"],
        "mpi_steps": cell["hops"],
        **p1.REQUIRED_CONFIG,
    }
    assert p1._validate_config(config, cell, p1.CHECKPOINT_STEP) == []
    config["method"] = "bar"
    assert p1._validate_config(config, cell, p1.CHECKPOINT_STEP) == []
    config["method"] = "mcep"
    assert any(
        "config method" in error
        for error in p1._validate_config(config, cell, p1.CHECKPOINT_STEP)
    )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("eval_episodes", 1),
        ("batch_size", 1),
        ("polyak", 0.5),
        ("policy_freq", 3),
        ("n_jitted_updates", 1),
    ],
)
def test_fixed_config_contract_rejects_incompatible_headline_settings(
    tmp_path: Path, field: str, tampered: object
):
    cell = p1.expected_cells(tmp_path)[0]
    config = {
        "env": cell["environment"],
        "seed": cell["seed"],
        "tau": cell["tau"],
        "mpi_steps": cell["hops"],
        **p1.REQUIRED_CONFIG,
    }
    config[field] = tampered
    errors = p1._validate_config(config, cell, p1.CHECKPOINT_STEP)
    assert any(f"config {field}" in error for error in errors)


def test_provenance_binds_all_sources_versions_and_git_state():
    provenance = p1._provenance_bundle()
    assert set(provenance["sources"]) == {
        "train_td3bc",
        "runner",
        "verifier",
        "_lab_import",
        "_ckpt_compat",
        "dump_target_policy_exposure",
        "d4rl_data",
    }
    assert set(provenance["environment_versions"]) == {
        "python",
        "jax",
        "flax",
        "optax",
        "numpy",
    }
    assert provenance["historical_training_identity_proven"] is False
    for state in (
        provenance["diagnostic_git"],
        provenance["imported_training_git"],
    ):
        assert len(state["revision"]) in {40, 64}
        assert isinstance(state["dirty"], bool)
        assert len(state["status_sha256"]) == 64


def test_direct_td_target_rms_uses_gamma_and_same_rows():
    summary, raw = p1.compute_target_value_metrics(
        q_actor=np.asarray([3.0, -1.0]),
        q_dataset=np.asarray([1.0, 2.0]),
        rewards=np.asarray([0.5, -0.5]),
        gamma=0.9,
    )
    expected_delta = np.asarray([1.8, -2.7])
    np.testing.assert_allclose(raw["delta_y"], expected_delta)
    assert summary["delta_y_rms"] == pytest.approx(
        np.sqrt(np.mean(np.square(expected_delta)))
    )
    assert summary["target_actor_rms"] == pytest.approx(
        np.sqrt(np.mean(np.square(np.asarray([3.2, -1.4]))))
    )
    assert summary["all_finite"] is True


def test_same_next_state_geometry_and_clipping_sensitivity():
    target = np.asarray([[0.5, -0.5], [0.0, 0.5]], dtype=np.float32)
    final = np.asarray([[1.0, -0.5], [0.5, 0.0]], dtype=np.float32)
    dataset = np.zeros_like(target)
    epsilon = np.asarray([[0.8, -0.8], [0.0, 0.8]], dtype=np.float32)
    smooth_target = np.clip(target + epsilon, -1.0, 1.0)
    smooth_dataset = np.clip(dataset + epsilon, -1.0, 1.0)
    row = p1.compute_same_next_state_geometry(
        target_actions=target,
        final_actions=final,
        dataset_actions=dataset,
        smoothed_target_actions=smooth_target,
        smoothed_dataset_actions=smooth_dataset,
        max_action=1.0,
    )
    target_rms = np.sqrt(np.mean(np.square(target)))
    final_rms = np.sqrt(np.mean(np.square(final)))
    assert row["target_to_next_data_rms"] == pytest.approx(target_rms)
    assert row["final_to_next_data_rms"] == pytest.approx(final_rms)
    assert row["final_target_ratio"] == pytest.approx(final_rms / target_rms)
    assert row["smoothed_target_saturation_fraction"] > 0


def test_realized_comparator_residual_is_conditional_sample_quantity():
    current = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    reference = np.zeros_like(current)
    terms = p1.comparator_residual_terms(
        q_current=np.asarray([3.0, 1.0]),
        q_reference=np.asarray([1.0, 1.0]),
        current_actions=current,
        reference_actions=reference,
        q_weight=0.5,
    )
    # -0.5 * (mean([3,1]) - mean([1,1])) + mean(square(delta)) = 0.
    assert terms["residual"] == pytest.approx(0.0)
    assert terms["feasible_comparator_slack"] == pytest.approx(0.0)
    worse = p1.comparator_residual_terms(
        q_current=np.asarray([1.0, 1.0]),
        q_reference=np.asarray([1.0, 1.0]),
        current_actions=current,
        reference_actions=reference,
        q_weight=0.5,
    )
    assert worse["residual"] == pytest.approx(0.5)
    assert worse["feasible_comparator_slack"] == pytest.approx(0.5)


def test_dry_run_does_not_resolve_jax_backend(tmp_path: Path, monkeypatch):
    def unexpected_backend_resolution():
        raise AssertionError("dry-run resolved the JAX backend")

    monkeypatch.setattr(p1.jax, "default_backend", unexpected_backend_resolution)
    assert p1.main(
        [
            "--mode",
            "dry-run",
            "--root",
            str(tmp_path / "missing"),
            "--out-dir",
            str(tmp_path / "dry-no-backend"),
        ]
    ) == 0


def test_scientific_run_rejects_non_cpu_backend_before_writes(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(p1.jax, "default_backend", lambda: "gpu")
    with pytest.raises(RuntimeError, match="CPU-only"):
        p1.run_analysis(out_dir=tmp_path, inventory={"entries": []}, arrays_by_env={})
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "MANIFEST.json").exists()


def test_dry_run_needs_no_checkpoints_or_datasets(tmp_path: Path):
    root = tmp_path / "empty-results"
    out = tmp_path / "dry"
    assert p1.main(
        [
            "--mode",
            "dry-run",
            "--root",
            str(root),
            "--data-dir",
            str(tmp_path / "no-data"),
            "--out-dir",
            str(out),
        ]
    ) == 0
    document = json.loads((out / "EXPECTED_GRID.json").read_text())
    status = json.loads((out / "DRY_RUN_STATUS.json").read_text())
    assert document["n_expected"] == 270
    assert len(document["cells"]) == 270
    assert document["analysis_started"] is False
    assert status["state"] == "dry_run_only"
    assert not (out / "CHECKPOINTS.json").exists()


def test_preflight_stops_atomically_on_missing_270_inventory(tmp_path: Path):
    out = tmp_path / "preflight"
    assert p1.main(
        [
            "--mode",
            "preflight",
            "--root",
            str(tmp_path / "empty-results"),
            "--data-dir",
            str(tmp_path / "no-data"),
            "--out-dir",
            str(out),
        ]
    ) == 2
    unresolved = json.loads((out / "CHECKPOINTS_UNRESOLVED.json").read_text())
    status = json.loads((out / "PREFLIGHT_BLOCKED.json").read_text())
    assert unresolved["validation_stage"] == "exact_270_checkpoint_path_presence"
    assert len(unresolved["failures"]) == 270
    assert unresolved["n_resolved"] == 0
    assert status["run_allowed"] is False
    assert not (out / "CHECKPOINTS.json").exists()
    assert not (out / "FROZEN_PROTOCOL.json").exists()
    assert not (out / "common_batches").exists()


@pytest.mark.parametrize("mode", ["preflight", "run"])
@pytest.mark.parametrize(
    "marker", ["CHECKPOINTS_UNRESOLVED.json", "PREFLIGHT_BLOCKED.json"]
)
def test_blocked_preflight_bundle_cannot_be_reused(
    tmp_path: Path, mode: str, marker: str
):
    out = tmp_path / f"{mode}-{marker}"
    out.mkdir()
    (out / marker).write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="blocked or partial"):
        p1._guard_output_lifecycle(out, mode)


def test_independent_verifier_detects_byte_corruption(tmp_path: Path):
    artifact = tmp_path / "raw.npz"
    artifact.write_bytes(b"original artifact bytes")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    verify_p1.check_file_hash(artifact, expected)
    artifact.write_bytes(b"corrupted artifact bytes")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        verify_p1.check_file_hash(artifact, expected)


def test_one_cell_smoke_uses_target_critic_and_stored_adam(tmp_path: Path):
    states = np.zeros((4, 3), dtype=np.float32)
    actions = np.zeros((4, 2), dtype=np.float32)
    train_state = p1._train_module.create_train_state(
        jax.random.PRNGKey(7),
        states[:1],
        actions[:1],
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        mpi_steps=3,
    )
    payload = {
        "mean": np.zeros(3, dtype=np.float32),
        "std": np.ones(3, dtype=np.float32),
        "max_action": 1.0,
        "actors_params": tuple(actor.params for actor in train_state.actors),
        "actors_steps": tuple(actor.step for actor in train_state.actors),
        "actors_opt_states": tuple(actor.opt_state for actor in train_state.actors),
        "critic_params": train_state.critic.params,
        "target_actor_params": train_state.target_actor.params,
        "target_critic_params": train_state.target_critic.params,
    }
    checkpoint = tmp_path / "params_1000000.pkl"
    checkpoint.write_bytes(b"hash-bound smoke checkpoint")
    record = {
        "key": "p3|halfcheetah-medium-v2|4|0",
        "method": "p3",
        "environment": "halfcheetah-medium-v2",
        "tau": 4.0,
        "seed": 0,
        "hops": 3,
        "checkpoint_step": 1_000_000,
        "checkpoint_path": str(checkpoint),
        "checkpoint_file_sha256": p1.sha256_file(checkpoint),
        "weights_sha256": "weights",
        "config_sha256": "config",
        "dataset_sha256": "dataset",
        "normalization_sha256": "normalization",
        "selected_transition_sha256": "transitions",
        "common_noise_sha256": "noise",
        "external_score": 50.0,
        "collapsed_lt20": False,
        "discount": 0.99,
        "learning_rate": 3e-4,
    }
    arrays = {
        "value_next_observations": states,
        "value_next_actions": actions,
        "value_rewards": np.ones(4, dtype=np.float32),
        "residual_observations": states,
        "epsilon": np.full_like(actions, 0.1),
    }
    value_rows, geometry, residual_rows = p1._analyze_cell(
        record=record,
        payload=payload,
        common_target_critic_params=payload["target_critic_params"],
        common_critic_sha256=p1.sha256_tree(payload["target_critic_params"]),
        arrays=arrays,
        raw_path=tmp_path / "raw.npz",
        out_dir=tmp_path,
    )
    assert [row["critic_scope"] for row in value_rows] == [
        p1.VALUE_SCOPE_OWN,
        p1.VALUE_SCOPE_COMMON,
    ]
    assert geometry["n_rows"] == 4
    assert [row["hop"] for row in residual_rows] == [2, 3]
    assert all(
        row["optimizer_step_after"] == row["optimizer_step_before"] + 1
        for row in residual_rows
    )
    assert all(
        row["intervention"] == "posthoc_final_checkpoint_one_adam_step"
        for row in residual_rows
    )
    assert all("delta_y_rms_no_noise" in row for row in value_rows)
    assert all("smoothing_delta_y_rms_change" in row for row in value_rows)
    assert all(row["raw_npz"] == "raw.npz" for row in value_rows)
    with np.load(tmp_path / "raw.npz", allow_pickle=False) as raw:
        assert "within_run_target_critic_delta_y_no_noise" in raw.files
        assert "deterministic_target_actions" in raw.files


def test_optimizer_preflight_rejects_actor_step_adam_count_mismatch():
    states = np.zeros((1, 3), dtype=np.float32)
    actions = np.zeros((1, 2), dtype=np.float32)
    train_state = p1._train_module.create_train_state(
        jax.random.PRNGKey(11),
        states,
        actions,
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        mpi_steps=3,
    )
    actors = tuple(actor.params for actor in train_state.actors)
    payload = {
        "actors_opt_states": tuple(actor.opt_state for actor in train_state.actors),
        "actors_steps": tuple(actor.step for actor in train_state.actors),
    }
    supported, _, details = p1._optimizer_state_exact(payload, actors, 3e-4)
    assert supported is True
    assert all(
        actor["actor_step"] == actor["adam_count"] for actor in details["actors"]
    )
    payload["actors_steps"] = (payload["actors_steps"][0] + 1, *payload["actors_steps"][1:])
    supported, reason, details = p1._optimizer_state_exact(payload, actors, 3e-4)
    assert supported is False
    assert "step/count mismatch" in str(reason)
    assert details == {}


def test_scientific_outputs_are_create_only_and_paths_are_bundle_relative(
    tmp_path: Path,
):
    csv_path = tmp_path / "scientific.csv"
    p1._write_csv(csv_path, [{"value": 1}])
    with pytest.raises(FileExistsError):
        p1._write_csv(csv_path, [{"value": 2}])
    assert verify_p1._resolve_declared_path(tmp_path, "raw/cell.npz") == (
        tmp_path / "raw" / "cell.npz"
    )
    with pytest.raises(AssertionError, match="bundle-relative"):
        verify_p1._resolve_declared_path(tmp_path, str((tmp_path / "raw.npz").resolve()))
    completed = tmp_path / "completed"
    completed.mkdir()
    (completed / "MANIFEST.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="completed scientific output"):
        p1.main(
            [
                "--mode",
                "dry-run",
                "--root",
                str(tmp_path / "missing"),
                "--out-dir",
                str(completed),
            ]
        )


def test_independent_verifier_recomputes_method_contrasts():
    value_rows = []
    for environment_index, environment in enumerate(p1.ENVIRONMENTS):
        for tau in p1.TAUS:
            for seed in p1.SEEDS:
                for method_index, method in enumerate(("td3", "p3", "p4")):
                    for scope_index, scope in enumerate(
                        (p1.VALUE_SCOPE_OWN, p1.VALUE_SCOPE_COMMON)
                    ):
                        baseline = environment_index + tau + seed + scope_index
                        value_rows.append(
                            {
                                "method": method,
                                "environment": environment,
                                "tau": tau,
                                "seed": seed,
                                "critic_scope": scope,
                                "delta_y_rms": baseline + method_index * 0.3,
                                "delta_y_rms_no_noise": baseline + method_index * 0.2,
                                "smoothing_delta_y_rms_change": method_index * 0.1,
                            }
                        )
    contrasts = p1._method_contrasts(value_rows)
    assert len(contrasts) == 580
    csv_like_values = [{key: str(value) for key, value in row.items()} for row in value_rows]
    csv_like_contrasts = [
        {key: str(value) for key, value in row.items()} for row in contrasts
    ]
    verify_p1._verify_method_contrasts(csv_like_values, csv_like_contrasts)
    csv_like_contrasts[0]["delta_y_rms_difference"] = "999"
    with pytest.raises(AssertionError):
        verify_p1._verify_method_contrasts(csv_like_values, csv_like_contrasts)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "analysis_incomplete"),
        ("cpu_only", False),
        ("jax_backend", "gpu"),
    ],
)
def test_verifier_rejects_incomplete_or_non_cpu_manifest(field: str, value: object):
    manifest = {
        "status": "analysis_complete",
        "cpu_only": True,
        "jax_backend": "cpu",
    }
    verify_p1._verify_manifest_contract(manifest)
    manifest[field] = value
    with pytest.raises(AssertionError):
        verify_p1._verify_manifest_contract(manifest)


def test_verifier_source_must_match_frozen_provenance():
    actual = verify_p1.sha256_file(Path(verify_p1.__file__).resolve())
    provenance = {"sources": {"verifier": {"sha256": actual}}}
    assert verify_p1._verify_verifier_source(provenance) == actual
    provenance["sources"]["verifier"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="executing verifier source hash"):
        verify_p1._verify_verifier_source(provenance)


def test_verifier_rejects_record_metadata_config_and_optimizer_tampering(
    tmp_path: Path,
):
    expected, row = _valid_verified_record(tmp_path)
    verify_p1._verify_record_contract(expected["key"], row, expected)

    metadata_tamper = copy.deepcopy(row)
    metadata_tamper["method"] = "p4"
    with pytest.raises(AssertionError, match="fixed record metadata mismatch"):
        verify_p1._verify_record_contract(expected["key"], metadata_tamper, expected)

    config_tamper = copy.deepcopy(row)
    config_tamper["config"]["batch_size"] = 1
    config_tamper["config_sha256"] = hashlib.sha256(
        verify_p1._canonical_json_bytes(config_tamper["config"])
    ).hexdigest()
    with pytest.raises(AssertionError, match="checkpoint config mismatch"):
        verify_p1._verify_record_contract(expected["key"], config_tamper, expected)

    hash_tamper = copy.deepcopy(row)
    hash_tamper["config_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="checkpoint config hash mismatch"):
        verify_p1._verify_record_contract(expected["key"], hash_tamper, expected)

    optimizer_tamper = copy.deepcopy(row)
    optimizer_tamper["optimizer_reconstruction"]["actors"][0]["adam_count"] -= 1
    with pytest.raises(AssertionError, match="optimizer reconstruction mismatch"):
        verify_p1._verify_record_contract(expected["key"], optimizer_tamper, expected)

    path_tamper = copy.deepcopy(row)
    path_tamper["checkpoint_path"] = path_tamper["config_path"]
    with pytest.raises(AssertionError, match="checkpoint companion path mismatch"):
        verify_p1._verify_record_contract(expected["key"], path_tamper, expected)


def test_verifier_is_artifact_only_and_rejects_formula_corruption():
    source = (DIAGNOSTICS / "verify_p1_target_value_audit.py").read_text()
    assert "import run_p1_target_value_audit" not in source
    with pytest.raises(AssertionError):
        verify_p1._close(1.0, 1.1, "corrupted metric", tolerance=1e-12)
