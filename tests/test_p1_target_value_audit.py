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
        for record in verify_p1._expected_record_metadata(tmp_path).values()
        if record["method"] == method
    )
    config = {
        "env": expected["environment"],
        "seed": expected["seed"],
        "tau": expected["tau"],
        **{
            key: value
            for key, value in verify_p1.REQUIRED_CONFIG.items()
            if key not in {"mpi_steps", "q_scale_norm", "integrator"}
        },
    }
    if method == "td3":
        config["save_dir"] = "/home/ext_csv/td3_bc_jax/results_qnorm"
    elif method == "p3":
        config.update(
            {
                "save_dir": "/home/ext_csv/td3_bc_jax/results_mpi3",
                "mpi_two_step": False,
                "mpi_three_step": True,
            }
        )
    else:
        config.update(
            {
                "mpi_steps": 4,
                "q_scale_norm": True,
                "integrator": "implicit",
            }
        )
    effective_config, resolution, errors = verify_p1._resolve_config_contract(
        config, expected
    )
    assert errors == []
    digest = "a" * 64
    expected_actor_step = verify_p1.CHECKPOINT_STEP // int(
        effective_config["policy_freq"]
    )
    legacy_profile = (
        resolution["profile_id"]
        != verify_p1.CONFIG_COMPATIBILITY_CONTRACT["fully_serialized_profile"]
    )
    row = {
        **expected,
        "checkpoint_step": verify_p1.CHECKPOINT_STEP,
        "external_score_step": verify_p1.CHECKPOINT_STEP,
        "config": config,
        "config_sha256": hashlib.sha256(
            verify_p1._canonical_json_bytes(config)
        ).hexdigest(),
        "config_schema": resolution["profile_id"],
        "effective_config": effective_config,
        "effective_config_sha256": hashlib.sha256(
            verify_p1._canonical_json_bytes(effective_config)
        ).hexdigest(),
        "compatibility_resolution": resolution,
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
            "optimizer": "optax.adam with frozen audit hyperparameters",
            "learning_rate": 3e-4,
            "storage_layout": verify_p1.OPTIMIZER_COMPATIBILITY_CONTRACT[
                "legacy_layout" if legacy_profile else "current_layout"
            ],
            "actor_step_source": (
                verify_p1.OPTIMIZER_COMPATIBILITY_CONTRACT["legacy_step_source"]
                if legacy_profile
                else "independently_stored_actor_step_and_adam_count"
            ),
            "actor_step_independently_stored": not legacy_profile,
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
            "inactive_legacy_slots": [],
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
    assert p1.CONFIG_COMPATIBILITY_CONTRACT == verify_p1.CONFIG_COMPATIBILITY_CONTRACT
    assert p1.OPTIMIZER_COMPATIBILITY_CONTRACT == verify_p1.OPTIMIZER_COMPATIBILITY_CONTRACT
    assert p1.ALLOWED_RAW_CONFIG_FIELDS == verify_p1.ALLOWED_RAW_CONFIG_FIELDS
    assert p1.PROTOCOL_VERSION == verify_p1.PROTOCOL_VERSION
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


def test_closed_legacy_profiles_match_runner_and_independent_verifier(tmp_path: Path):
    verifier_expected = verify_p1._expected_record_metadata(tmp_path)

    def raw_base(cell):
        return {
            "env": cell["environment"],
            "seed": cell["seed"],
            "tau": cell["tau"],
            **{
                key: value
                for key, value in p1.REQUIRED_CONFIG.items()
                if key not in {"q_scale_norm", "integrator"}
            },
        }

    td3 = next(cell for cell in p1.expected_cells(tmp_path) if cell["method"] == "td3")
    p3 = next(cell for cell in p1.expected_cells(tmp_path) if cell["method"] == "p3")
    p4 = next(cell for cell in p1.expected_cells(tmp_path) if cell["method"] == "p4")
    td3_config = raw_base(td3)
    td3_config["save_dir"] = "/home/ext_csv/td3_bc_jax/results_qnorm"
    p3_config = raw_base(p3)
    p3_config.update(
        {
            "save_dir": "/home/ext_csv/td3_bc_jax/results_mpi3",
            "mpi_two_step": False,
            "mpi_three_step": True,
        }
    )
    p4_config = {
        "env": p4["environment"],
        "seed": p4["seed"],
        "tau": p4["tau"],
        "mpi_steps": 4,
        **p1.REQUIRED_CONFIG,
    }

    expected_profiles = {
        "td3": "legacy_td3_qnorm_minimal_v0",
        "p3": "legacy_p3_implicit_minimal_v0",
        "p4": "fully_serialized_config_v1",
    }
    for cell, config in ((td3, td3_config), (p3, p3_config), (p4, p4_config)):
        before = copy.deepcopy(config)
        effective, resolution, errors = p1._resolve_config_contract(
            config, cell, p1.CHECKPOINT_STEP
        )
        verified_effective, verified_resolution, verified_errors = (
            verify_p1._resolve_config_contract(
                config, verifier_expected[cell["key"]]
            )
        )
        assert errors == verified_errors == []
        assert effective == verified_effective
        assert resolution == verified_resolution
        assert resolution["profile_id"] == expected_profiles[cell["method"]]
        assert config == before
        assert all(field not in config for field in resolution["inferred_fields"])

    mutations = []
    bad_save = copy.deepcopy(td3_config)
    bad_save["save_dir"] = "/home/ext_csv/td3_bc_jax/results_mpi3"
    mutations.append((td3, bad_save))
    bad_td3_mode = copy.deepcopy(td3_config)
    bad_td3_mode["mpi_three_step"] = True
    mutations.append((td3, bad_td3_mode))
    bad_p3_scale = copy.deepcopy(p3_config)
    bad_p3_scale["q_scale_norm"] = False
    mutations.append((p3, bad_p3_scale))
    bad_p3_mode = copy.deepcopy(p3_config)
    bad_p3_mode["explicit_two_step"] = True
    mutations.append((p3, bad_p3_mode))
    missing_p4_field = copy.deepcopy(p4_config)
    missing_p4_field.pop("integrator")
    mutations.append((p4, missing_p4_field))
    bool_hop = copy.deepcopy(p4_config)
    bool_hop["mpi_steps"] = True
    mutations.append((p4, bool_hop))
    raw_float_drift = copy.deepcopy(p4_config)
    raw_float_drift["policy_noise"] = 0.2 + 1e-10
    mutations.append((p4, raw_float_drift))
    string_float = copy.deepcopy(p4_config)
    string_float["discount"] = "0.99"
    mutations.append((p4, string_float))
    for field, value in (
        ("refinement_control", "direct_final"),
        ("critic_target_route", "final"),
        ("route_shadow_critics", True),
        ("track_final_target_actor", True),
    ):
        unknown_semantic_control = copy.deepcopy(p4_config)
        unknown_semantic_control[field] = value
        mutations.append((p4, unknown_semantic_control))

    for cell, config in mutations:
        runner_errors = p1._resolve_config_contract(
            config, cell, p1.CHECKPOINT_STEP
        )[2]
        verifier_errors = verify_p1._resolve_config_contract(
            config, verifier_expected[cell["key"]]
        )[2]
        assert runner_errors
        assert verifier_errors


def test_action_scalar_contract_accepts_only_exact_float_encodings():
    expected_float32 = float(np.float32(0.2) * np.float32(1.0))
    accepted = (0.2, expected_float32)
    rejected = (
        float(np.nextafter(np.float32(expected_float32), np.float32(np.inf))),
        expected_float32 + 1e-12,
        float("nan"),
        float("inf"),
        "0.2",
        np.asarray(0.2),
    )
    for value in accepted:
        assert p1._stored_action_scalar_matches(value, 0.2, 1.0)
        assert verify_p1._stored_action_scalar_matches(value, 0.2, 1.0)
    for value in rejected:
        assert not p1._stored_action_scalar_matches(value, 0.2, 1.0)
        assert not verify_p1._stored_action_scalar_matches(value, 0.2, 1.0)


def test_payload_config_requires_json_objects(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="companion config.json is not a JSON object"):
        p1._payload_config({"config": {}}, config_path)

    config_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="embedded checkpoint config is not a JSON object"):
        p1._payload_config({"config": []}, config_path)


def test_external_score_requires_a_canonical_integer_step(tmp_path: Path):
    eval_path = tmp_path / "eval.csv"
    eval_path.write_text(
        "step,d4rl_score\n1000000,42.5\n",
        encoding="utf-8",
    )
    cell = {"eval_path": str(eval_path), "method": "td3"}
    score, step, _ = p1._read_external_score(cell)
    assert score == 42.5
    assert step == p1.CHECKPOINT_STEP

    for malformed in ("1000000.9", "1000000.00000000001", " 1000000"):
        eval_path.write_text(
            f"step,d4rl_score\n{malformed},42.5\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="found 0"):
            p1._read_external_score(cell)


def test_verifier_integer_parsers_reject_coercible_values():
    assert verify_p1._json_integer(1, "json") == 1
    assert verify_p1._csv_integer("1", "csv") == 1
    for value in ("1", 1.0, 1.9, True, None):
        with pytest.raises(AssertionError, match="exact JSON integer"):
            verify_p1._json_integer(value, "json")
    for value in ("1.0", "1.9", " 1", "+1", 1, True):
        with pytest.raises(AssertionError, match="canonical CSV integer"):
            verify_p1._csv_integer(value, "csv")


def test_present_file_gate_rejects_descendant_symlinks(tmp_path: Path):
    root = tmp_path / "root"
    target = tmp_path / "target"
    root.mkdir()
    first = p1.expected_cells(root)[0]
    target_run = target / first["run_name"]
    target_run.mkdir(parents=True)
    for name in ("params_1000000.pkl", "config.json", "eval.csv"):
        (target_run / name).write_bytes(b"trusted fixture")
    (root / first["result_dir"]).symlink_to(target, target_is_directory=True)

    failures = p1._present_file_gate(
        [first], ("checkpoint_path", "config_path", "eval_path")
    )
    assert len(failures) == 3
    assert {failure["reason"] for failure in failures} == {
        "path is not a canonical nonsymlink descendant"
    }


@pytest.mark.parametrize("stored_step", ["1000000", 1_000_000.9, True])
def test_checkpoint_step_requires_an_exact_integer_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stored_step: object
):
    payload = {
        "step": stored_step,
        "mean": np.zeros(1, dtype=np.float32),
        "std": np.ones(1, dtype=np.float32),
        "max_action": 1.0,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "critic_params": {},
        "target_actor_params": {},
        "target_critic_params": {},
    }
    monkeypatch.setattr(p1, "load_checkpoint", lambda _: payload)
    cell = {
        "checkpoint_path": str(tmp_path / "params_1000000.pkl"),
        "config_path": str(tmp_path / "config.json"),
    }
    with pytest.raises(ValueError, match="is not exact integer 1000000"):
        p1._inspect_checkpoint_cell(cell, p1.CHECKPOINT_STEP)


@pytest.mark.parametrize(
    ("field", "stored_value", "message"),
    [
        ("max_action", "1.0", "max_action must be exact fixed value 1.0"),
        ("policy_noise", "0.2", "stored policy_noise"),
        ("noise_clip", np.asarray(0.5), "stored noise_clip"),
    ],
)
def test_checkpoint_action_scalars_are_checked_before_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    stored_value: object,
    message: str,
):
    payload = {
        "step": p1.CHECKPOINT_STEP,
        "mean": np.zeros(1, dtype=np.float32),
        "std": np.ones(1, dtype=np.float32),
        "max_action": 1.0,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "critic_params": {},
        "target_actor_params": {},
        "target_critic_params": {},
    }
    payload[field] = stored_value
    monkeypatch.setattr(p1, "load_checkpoint", lambda _: payload)
    monkeypatch.setattr(p1, "_payload_config", lambda *_: ({}, "a" * 64, "b" * 64))
    monkeypatch.setattr(
        p1,
        "_resolve_config_contract",
        lambda *_: ({"policy_noise": 0.2, "noise_clip": 0.5}, {}, []),
    )
    monkeypatch.setattr(p1, "_actors", lambda *_: (object(),))
    cell = {
        "checkpoint_path": str(tmp_path / "params_1000000.pkl"),
        "config_path": str(tmp_path / "config.json"),
        "hops": 1,
    }
    with pytest.raises(ValueError, match=message):
        p1._inspect_checkpoint_cell(cell, p1.CHECKPOINT_STEP)

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
        "actors_params": actors,
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


def test_legacy_optimizer_uses_stored_adam_count_and_rejects_dirty_inactive_slots():
    states = np.zeros((2, 3), dtype=np.float32)
    actions = np.zeros((2, 2), dtype=np.float32)
    train_state = p1._train_module.create_train_state(
        jax.random.PRNGKey(19),
        states[:1],
        actions[:1],
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        mpi_steps=3,
    )
    actors = tuple(actor.params for actor in train_state.actors)
    opt_states = tuple(actor.opt_state for actor in train_state.actors)
    legacy_payload = {
        "actor_params": actors[0],
        "actor2_params": actors[1],
        "actor3_params": actors[2],
        "actor_opt_state": opt_states[0],
        "actor2_opt_state": opt_states[1],
        "actor3_opt_state": opt_states[2],
        "critic_params": train_state.critic.params,
    }

    supported, reason, details = p1._optimizer_state_exact(
        legacy_payload, actors[:1], 3e-4
    )
    assert supported is True
    assert reason is None
    assert details["storage_layout"] == p1.OPTIMIZER_COMPATIBILITY_CONTRACT[
        "legacy_layout"
    ]
    assert details["actor_step_source"] == "unique_stored_adam_integer_count"
    assert details["actor_step_independently_stored"] is False
    assert len(details["inactive_legacy_slots"]) == 2
    assert all(
        slot["adam_count"] == 0 and slot["fresh_optimizer_state_exact"]
        for slot in details["inactive_legacy_slots"]
    )

    mixed_layout = copy.copy(legacy_payload)
    mixed_layout["actors_params"] = actors[:1]
    supported, reason, _ = p1._optimizer_state_exact(
        mixed_layout, actors[:1], 3e-4
    )
    assert supported is False
    assert "mixes tuple actor params" in str(reason)

    dirty_moments = copy.copy(legacy_payload)
    dirty_moments["actor2_opt_state"] = jax.tree_util.tree_map(
        lambda leaf: (
            p1.jnp.ones_like(leaf)
            if np.issubdtype(np.asarray(leaf).dtype, np.floating)
            else leaf
        ),
        legacy_payload["actor2_opt_state"],
    )
    supported, reason, _ = p1._optimizer_state_exact(
        dirty_moments, actors[:1], 3e-4
    )
    assert supported is False
    assert "not fresh-zero" in str(reason)

    dirty_count = copy.copy(legacy_payload)
    dirty_count["actor2_opt_state"] = jax.tree_util.tree_map(
        lambda leaf: (
            leaf + 1
            if np.asarray(leaf).shape == ()
            and np.issubdtype(np.asarray(leaf).dtype, np.integer)
            else leaf
        ),
        legacy_payload["actor2_opt_state"],
    )
    supported, reason, _ = p1._optimizer_state_exact(
        dirty_count, actors[:1], 3e-4
    )
    assert supported is False
    assert "expected 0" in str(reason)

    supported, reason, p3_details = p1._optimizer_state_exact(
        legacy_payload, actors, 3e-4
    )
    assert supported is True
    assert reason is None
    assert [actor["actor_step"] for actor in p3_details["actors"]] == [0, 0, 0]

    rows, _ = p1._posthoc_residual_interventions(
        payload=legacy_payload,
        actors=actors,
        actor_model=p1.Actor(action_dim=2, max_action=1.0),
        critic=p1.TwinCritic(),
        states=states,
        tau=4.0,
        learning_rate=3e-4,
    )
    assert [row["optimizer_step_before"] for row in rows] == [0, 0]
    assert [row["optimizer_step_after"] for row in rows] == [1, 1]



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


def test_verifier_protocol_headers_require_exact_types_and_values():
    inventory = {
        "protocol": verify_p1.PROTOCOL_VERSION,
        "atomic_inventory": True,
        "analysis_started": False,
        "no_substitution": True,
        "inventory_complete": True,
        "n_expected": 270,
        "n_resolved": 270,
        "checkpoint_step": verify_p1.CHECKPOINT_STEP,
    }
    protocol = {"protocol": verify_p1.PROTOCOL_VERSION, "locked": True}
    verify_p1._verify_protocol_headers(inventory, protocol)

    mutations = []
    for field, value in (
        ("atomic_inventory", "true"),
        ("analysis_started", 0),
        ("no_substitution", 1),
        ("inventory_complete", "true"),
        ("n_expected", 270.9),
        ("n_resolved", "270"),
        ("checkpoint_step", 1_000_000.9),
    ):
        changed = copy.deepcopy(inventory)
        changed[field] = value
        mutations.append((changed, protocol))
    mutations.append((inventory, {**protocol, "protocol": "wrong"}))
    mutations.append((inventory, {**protocol, "locked": "true"}))
    for changed_inventory, changed_protocol in mutations:
        with pytest.raises(AssertionError):
            verify_p1._verify_protocol_headers(changed_inventory, changed_protocol)


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
    effective_tamper = copy.deepcopy(row)
    effective_tamper["effective_config"]["q_scale_norm"] = False
    effective_tamper["effective_config_sha256"] = hashlib.sha256(
        verify_p1._canonical_json_bytes(effective_tamper["effective_config"])
    ).hexdigest()
    with pytest.raises(AssertionError, match="effective checkpoint config mismatch"):
        verify_p1._verify_record_contract(expected["key"], effective_tamper, expected)

    resolution_tamper = copy.deepcopy(row)
    resolution_tamper["compatibility_resolution"]["profile_id"] = (
        "legacy_p3_implicit_explicit_flags_v1"
    )
    with pytest.raises(
        AssertionError, match="checkpoint compatibility resolution mismatch"
    ):
        verify_p1._verify_record_contract(expected["key"], resolution_tamper, expected)

    scalar_tamper = copy.deepcopy(row)
    scalar_tamper["policy_noise"] = float(
        np.nextafter(np.float32(0.2), np.float32(np.inf))
    )
    with pytest.raises(AssertionError, match="policy_noise encoding mismatch"):
        verify_p1._verify_record_contract(expected["key"], scalar_tamper, expected)

    rescaled_action_tamper = copy.deepcopy(row)
    rescaled_action_tamper.update(
        {"max_action": 2.0, "policy_noise": 0.4, "noise_clip": 1.0}
    )
    with pytest.raises(AssertionError, match="max_action is invalid"):
        verify_p1._verify_record_contract(
            expected["key"], rescaled_action_tamper, expected
        )

    step_source_tamper = copy.deepcopy(row)
    step_source_tamper["optimizer_reconstruction"]["actor_step_source"] = (
        "independently_stored_actor_step_and_adam_count"
    )
    with pytest.raises(AssertionError, match="optimizer actor-step source mismatch"):
        verify_p1._verify_record_contract(expected["key"], step_source_tamper, expected)

    missing_step_storage = copy.deepcopy(row)
    missing_step_storage["optimizer_reconstruction"].pop(
        "actor_step_independently_stored"
    )
    with pytest.raises(AssertionError, match="optimizer actor-step storage claim mismatch"):
        verify_p1._verify_record_contract(expected["key"], missing_step_storage, expected)



    optimizer_tamper = copy.deepcopy(row)
    optimizer_tamper["optimizer_reconstruction"]["actors"][0]["adam_count"] -= 1
    with pytest.raises(AssertionError, match="optimizer reconstruction mismatch"):
        verify_p1._verify_record_contract(expected["key"], optimizer_tamper, expected)

    path_tamper = copy.deepcopy(row)
    path_tamper["checkpoint_path"] = path_tamper["config_path"]
    with pytest.raises(AssertionError, match="canonical checkpoint path mismatch"):
        verify_p1._verify_record_contract(expected["key"], path_tamper, expected)

    path_alias_tamper = copy.deepcopy(row)
    path_alias_tamper["run_dir"] = path_alias_tamper["run_dir"] + "/."
    with pytest.raises(AssertionError, match="canonical checkpoint path mismatch"):
        verify_p1._verify_record_contract(expected["key"], path_alias_tamper, expected)


def test_verifier_is_artifact_only_and_rejects_formula_corruption():
    source = (DIAGNOSTICS / "verify_p1_target_value_audit.py").read_text()
    assert "import run_p1_target_value_audit" not in source
    with pytest.raises(AssertionError):
        verify_p1._close(1.0, 1.1, "corrupted metric", tolerance=1e-12)
