import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


DIAGNOSTICS = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

import run_p2_relu_residence as p2  # noqa: E402
import verify_p2_relu_residence as verify_p2  # noqa: E402
import verify_p2_relu_residence_final as verify_final  # noqa: E402


def test_import_isolation_preserves_env_and_jax_x64():
    watched = {
        "CUDA_VISIBLE_DEVICES": "5",
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "OPENBLAS_NUM_THREADS": "3",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "5",
        "NUMEXPR_NUM_THREADS": "6",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_FLAGS": (
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=2"
        ),
    }
    program = f"""
import os
import jax
import sys
sys.path.insert(0, {str(DIAGNOSTICS)!r})
keys = {tuple(watched)!r}
before_env = {{key: os.environ.get(key) for key in keys}}
before_x64 = bool(jax.config.jax_enable_x64)
import run_p2_relu_residence
import verify_p2_relu_residence
after_env = {{key: os.environ.get(key) for key in keys}}
after_x64 = bool(jax.config.jax_enable_x64)
assert before_env == after_env, (before_env, after_env)
assert before_x64 is False, before_x64
assert after_x64 == before_x64, (before_x64, after_x64)
"""
    environment = os.environ.copy()
    environment.update(watched)
    environment.pop("JAX_ENABLE_X64", None)
    result = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_design_lock_is_written_before_analysis_and_separates_final(tmp_path: Path):
    design = p2.write_design_lock(tmp_path)
    assert (tmp_path / "DESIGN_LOCK.json").is_file()
    assert design["estimand"]["learned_primary"] == "relu_activation_region_residence"
    assert design["estimand"]["learned_slope"] is None
    assert design["estimand"]["learned_support_gate"] is None
    assert design["pilot_contract"]["scientific_admissible"] is False
    final = design["final_contract_not_executed"]
    assert final["n_runs"] == 12
    assert final["excluded_family"] == "halfcheetah"
    assert all(not env.startswith("halfcheetah-") for env in final["environments"])
    raw = json.loads((tmp_path / "DESIGN_LOCK.json").read_text())
    without_hash = dict(raw)
    claimed = without_hash.pop("design_sha256")
    assert p2.sha256_bytes(p2.canonical_bytes(without_hash)) == claimed
    with pytest.raises(FileExistsError):
        p2.write_design_lock(tmp_path)




def test_run_pilot_fails_closed_without_jax_x64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    p2.write_design_lock(tmp_path)
    p2.run_harness(tmp_path)
    monkeypatch.setattr(p2.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(p2.jax, "config", SimpleNamespace(jax_enable_x64=False))

    with pytest.raises(RuntimeError, match="jax_enable_x64=true"):
        p2.run_pilot(
            tmp_path,
            Path("never-opened-checkpoint.pkl"),
            Path("never-opened-config.json"),
            Path("never-opened-dataset.hdf5"),
            [],
        )

    assert not (tmp_path / p2.SOURCE_SNAPSHOT_DIR).exists()
    assert not (tmp_path / "INPUT_LOCK.json").exists()

def test_smooth_and_known_boundary_harness_passes(tmp_path: Path):
    p2.write_design_lock(tmp_path)
    report = p2.run_harness(tmp_path)
    assert report["pass"] is True
    slopes = report["smooth_quadratic"]["slopes_K4_8_16"]
    assert 0.9 <= slopes["explicit"] <= 1.1
    assert 0.9 <= slopes["implicit"] <= 1.1
    assert set(report["synthetic_relu"]) == {
        "resident",
        "layer1_exit",
        "layer2_exit",
        "box_exit",
        "simultaneous_exit",
        "anchor_boundary",
    }
    assert all(case["pass"] for case in report["synthetic_relu"].values())
    with pytest.raises(FileExistsError):
        p2.run_harness(tmp_path)


@pytest.mark.parametrize(
    ("case", "expected_time", "expected_code", "expected_layer"),
    [
        ("layer1_exit", 0.1, p2.FIRST_RELU, 1),
        ("layer2_exit", 0.1, p2.FIRST_RELU, 2),
        ("box_exit", 0.1, p2.FIRST_BOX, -1),
        ("simultaneous_exit", 0.1, p2.FIRST_TIE, 1),
    ],
)
def test_exact_exit_geometry(case, expected_time, expected_code, expected_layer):
    params, action = p2.synthetic_params(case)
    geometry = p2.analyze_geometry(params, np.zeros((1, 0)), action, 1.0)
    assert geometry["first_hit_time"][0] == pytest.approx(expected_time, abs=1e-12)
    assert int(geometry["first_hit_code"][0]) == expected_code
    assert int(geometry["first_layer"][0]) == expected_layer
    assert p2.classify_horizon(geometry, 0, 0.05) == "resident"


def test_anchor_boundary_is_retained_as_ambiguous():
    params, action = p2.synthetic_params("anchor_boundary")
    geometry = p2.analyze_geometry(params, np.zeros((1, 0)), action, 1.0)
    assert bool(geometry["boundary_ambiguous"][0])
    assert p2.classify_horizon(geometry, 0, 0.05) == "boundary_ambiguous"


def test_independent_numpy_geometry_matches_runner_without_importing_it():
    source = Path(verify_p2.__file__).read_text()
    assert "import run_p2_relu_residence" not in source
    params, actions = p2.synthetic_params("layer2_exit")
    states = np.zeros((1, 0))
    runner = p2.analyze_geometry(params, states, actions, 1.0)
    verifier = verify_p2.independent_geometry(params, states, actions, 1.0)
    for name in (
        "q_anchor",
        "grad",
        "velocity",
        "relu_hit_time",
        "box_hit_time",
        "first_hit_time",
        "first_hit_code",
        "first_layer",
        "first_unit",
        "boundary_ambiguous",
        "nonfinite",
        "anchor_min_scaled_margin",
    ):
        if runner[name].dtype.kind in "biu":
            assert np.array_equal(runner[name], verifier[name])
        else:
            assert np.allclose(runner[name], verifier[name], equal_nan=True)


def test_step_residence_is_monotone_and_full_residence_is_k_invariant():
    params, action = p2.synthetic_params("layer1_exit")
    geometry = p2.analyze_geometry(params, np.zeros((1, 0)), action, 1.0)
    total = 0.2
    full = [p2.classify_horizon(geometry, 0, total) for _ in p2.SUBSTEPS]
    step = [p2.classify_horizon(geometry, 0, total / k) for k in p2.SUBSTEPS]
    assert len(set(full)) == 1
    assert full[0] == "relu_cross"
    resident = [value == "resident" for value in step]
    assert resident == sorted(resident)


def test_pilot_indices_are_new_and_deterministic(tmp_path: Path):
    actions = np.zeros((100, 2), dtype=np.float64)
    old_a = tmp_path / "old_a.npy"
    old_b = tmp_path / "old_b.npy"
    np.save(old_a, np.arange(10, dtype=np.int64))
    np.save(old_b, np.arange(10, 20, dtype=np.int64))
    first, records, union_count = p2.select_pilot_indices(actions, [old_a, old_b])
    second, _, _ = p2.select_pilot_indices(actions, [old_a, old_b])
    assert len(first) == p2.N_PILOT
    assert np.array_equal(first, second)
    assert not set(first).intersection(range(20))
    assert len(records) == 2
    assert union_count == 20


def test_verifier_finite_difference_and_exit_bracket_on_known_case():
    params, actions = p2.synthetic_params("layer2_exit")
    states = np.zeros((1, 0))
    geometry = verify_p2.independent_geometry(params, states, actions, 1.0)
    checked, max_error = verify_p2.finite_difference_checks(
        params, states, actions, geometry
    )
    exits, exit_error = verify_p2.exit_bracket_checks(
        params, states, actions, geometry
    )
    assert checked == 1
    assert max_error < 1e-7
    assert exits == 1
    assert exit_error < 1e-8


def test_verifier_recomputes_harness_and_rejects_tampering(tmp_path: Path):
    p2.write_design_lock(tmp_path)
    report = p2.run_harness(tmp_path)
    verify_p2.verify_harness(report)

    tampered_error = copy.deepcopy(report)
    tampered_error["smooth_quadratic"]["rows"][0]["error"] *= 2.0
    with pytest.raises(AssertionError, match="quadratic error mismatch"):
        verify_p2.verify_harness(tampered_error)

    tampered_case = copy.deepcopy(report)
    tampered_case["synthetic_relu"]["layer1_exit"]["first_hit_time"] = 0.11
    with pytest.raises(AssertionError, match="synthetic reported time mismatch"):
        verify_p2.verify_harness(tampered_case)


@pytest.mark.parametrize("case", ["box_exit", "simultaneous_exit"])
def test_direct_box_and_combined_event_witnesses(case: str):
    params, actions = p2.synthetic_params(case)
    states = np.zeros((1, 0))
    geometry = verify_p2.independent_geometry(params, states, actions, 1.0)
    report = verify_p2.box_and_combined_event_checks(
        params, states, actions, geometry
    )
    assert report["finite_box_rows_checked"] == 1
    assert report["combined_event_rows_checked"] == 1
    if case == "simultaneous_exit":
        assert report["combined_ties_checked"] == 1


def test_layer2_radius_propagates_layer1_cancellation_error():
    params = {
        "w1": np.array([[1e16, 1e16], [-2e16, -2e16]], dtype=np.float64),
        "b1": np.array([1000.0, 1000.0]),
        "w2": np.array([[1.0], [-1.0]]),
        "b2": np.array([100.0]),
        "w3": np.array([[1.0]]),
        "b3": np.array([0.0]),
    }
    states = np.array([[1.0]])
    actions = np.array([[0.5]])
    _, z1, z2 = p2.q1_forward(params, states, actions)
    inputs = np.concatenate((states, actions), axis=-1)
    scale1 = np.abs(inputs) @ np.abs(params["w1"]) + np.abs(params["b1"])
    radius1 = p2.affine_tol(scale1, inputs.shape[-1] + 1)
    h1 = np.maximum(z1, 0.0)
    scale2 = np.abs(h1) @ np.abs(params["w2"]) + np.abs(params["b2"])
    local2 = p2.affine_tol(scale2, h1.shape[-1] + 1)
    propagated2 = (
        1.0 + p2.ROUNDING_SAFETY * p2.gamma_n(h1.shape[-1] + 1)
    ) * (radius1 @ np.abs(params["w2"]))
    assert np.all(np.abs(z1) > radius1)
    assert np.all(np.abs(z2) > local2)
    assert np.all(np.abs(z2) <= local2 + propagated2)
    runner = p2.analyze_geometry(params, states, actions, 1.0)
    verifier = verify_p2.independent_geometry(params, states, actions, 1.0)
    assert bool(runner["boundary_ambiguous"][0])
    assert bool(verifier["boundary_ambiguous"][0])


def test_input_lock_rejects_incomplete_inventory(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "params.pkl"
    config = tmp_path / "config.json"
    dataset = tmp_path / "dataset.hdf5"
    seed0 = tmp_path / "seed0.npy"
    seed1 = tmp_path / "seed1.npy"
    checkpoint.write_bytes(b"checkpoint")
    config.write_bytes(b"config")
    dataset.write_bytes(b"dataset")
    seed0.write_bytes(b"seed0")
    seed1.write_bytes(b"seed1")
    inputs = {
        "checkpoint": {
            "semantic_id": "checkpoint",
            "basename": checkpoint.name,
            "sha256": p2.sha256_file(checkpoint),
        },
        "config": {
            "semantic_id": "config",
            "basename": config.name,
            "sha256": p2.sha256_file(config),
        },
        "dataset": {
            "semantic_id": "dataset",
            "basename": dataset.name,
            "sha256": p2.sha256_file(dataset),
        },
    }
    exclusions = (
        {"semantic_id": "seed0", "basename": seed0.name, "sha256": p2.sha256_file(seed0), "n_indices": 1},
        {"semantic_id": "seed1", "basename": seed1.name, "sha256": p2.sha256_file(seed1), "n_indices": 1},
    )
    monkeypatch.setattr(p2, "PINNED_INPUTS", inputs)
    monkeypatch.setattr(p2, "PINNED_EXCLUSIONS", exclusions)
    complete = tmp_path / "complete"
    complete.mkdir()
    document = p2.write_input_lock(
        complete, checkpoint, config, dataset, [seed0, seed1]
    )
    assert document["status"] == "inputs_locked_before_semantic_read"
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="exactly seed0 and seed1"):
        p2.write_input_lock(incomplete, checkpoint, config, dataset, [seed0])


def test_artifact_inventory_rejects_omission_and_hash_tamper(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "nested" / "second.bin"
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    expected_paths = {"first.json", "nested/second.bin"}
    artifacts = {
        "first.json": verify_p2.sha256_file(first),
        "nested/second.bin": verify_p2.sha256_file(second),
    }
    verify_p2.verify_artifact_inventory(tmp_path, artifacts, expected_paths)

    with pytest.raises(AssertionError, match="artifact inventory mismatch"):
        verify_p2.verify_artifact_inventory(
            tmp_path,
            {"first.json": artifacts["first.json"]},
            expected_paths,
        )

    second.write_bytes(b"tampered")
    with pytest.raises(AssertionError, match="artifact hash mismatch"):
        verify_p2.verify_artifact_inventory(tmp_path, artifacts, expected_paths)


def test_stored_verify_rejects_stale_recomputation(tmp_path: Path):
    verify_path = tmp_path / "VERIFY.json"
    recomputed = {
        "protocol": verify_p2.PROTOCOL,
        "pass": True,
        "finite_difference_coordinates_checked": 384,
        "written_at": "2026-09-02T22:00:00+09:00",
    }
    stored = {
        **recomputed,
        "written_at": "2026-09-02T21:00:00+09:00",
    }
    verify_path.write_text(json.dumps(stored))
    verify_p2.verify_stored_report(verify_path, recomputed)

    stored["finite_difference_coordinates_checked"] = 383
    verify_path.write_text(json.dumps(stored))
    with pytest.raises(AssertionError, match="stored VERIFY does not match"):
        verify_p2.verify_stored_report(verify_path, recomputed)


def test_git_provenance_rejects_tampering():
    valid = {
        "git_dirty": True,
        "git_revision": "a" * 40,
        "source_snapshot_is_durable_provenance": True,
    }
    verify_p2.verify_git_provenance(valid)

    bad_dirty = {**valid, "git_dirty": "true"}
    with pytest.raises(AssertionError, match="git dirty provenance missing"):
        verify_p2.verify_git_provenance(bad_dirty)

    bad_revision = {**valid, "git_revision": "not-a-revision"}
    with pytest.raises(AssertionError, match="git revision provenance missing"):
        verify_p2.verify_git_provenance(bad_revision)

    bad_snapshot = {**valid, "source_snapshot_is_durable_provenance": False}
    with pytest.raises(AssertionError, match="durable source provenance"):
        verify_p2.verify_git_provenance(bad_snapshot)


def test_final_verifier_requires_clean_origin_main_provenance():
    valid = {
        "git_dirty": False,
        "git_tracked_dirty": False,
        "git_revision": "a" * 40,
        "origin_main_revision": "a" * 40,
        "head_matches_origin_main": True,
        "source_snapshot_is_durable_provenance": True,
    }
    p2.require_clean_origin_main(valid)
    verify_final.verify_final_git_provenance(valid)

    untracked_dirty = {**valid, "git_dirty": True}
    with pytest.raises(RuntimeError, match="fully clean worktree"):
        p2.require_clean_origin_main(untracked_dirty)
    with pytest.raises(AssertionError, match="fully clean worktree"):
        verify_final.verify_final_git_provenance(untracked_dirty)

    dirty = {**valid, "git_tracked_dirty": True}
    with pytest.raises(RuntimeError, match="clean tracked worktree"):
        p2.require_clean_origin_main(dirty)
    with pytest.raises(AssertionError, match="clean tracked worktree"):
        verify_final.verify_final_git_provenance(dirty)

    diverged = {
        **valid,
        "origin_main_revision": "b" * 40,
        "head_matches_origin_main": False,
    }
    with pytest.raises(RuntimeError, match="HEAD to equal"):
        p2.require_clean_origin_main(diverged)
    with pytest.raises(AssertionError, match="HEAD == origin/main"):
        verify_final.verify_final_git_provenance(diverged)


def test_cell_matcher_rejects_derived_h_tamper():
    expected = {
        "environment": verify_p2.ENVIRONMENT,
        "seed": verify_p2.SEED,
        "T": 0.1,
        "K": 4,
        "h": 0.025,
        "n_states": verify_p2.N_PILOT,
        **{f"full_{name}": 0 for name in verify_p2.CATEGORIES},
        **{f"step_{name}": 0 for name in verify_p2.CATEGORIES},
        "full_residence_fraction": 44 / verify_p2.N_PILOT,
        "step_residence_fraction": 55 / verify_p2.N_PILOT,
        "full_affine_q_max_abs_residual": 4e-13,
    }
    expected["full_resident"] = 44
    expected["full_relu_cross"] = 20
    expected["step_resident"] = 55
    expected["step_relu_cross"] = 9
    actual = copy.deepcopy(expected)
    verify_p2.assert_cell_matches(
        actual,
        expected,
        "hermetic",
        affine_comparison_tolerance=1e-12,
    )

    actual["h"] += 1e-3
    with pytest.raises(AssertionError, match="float mismatch: h"):
        verify_p2.assert_cell_matches(
            actual,
            expected,
            "hermetic",
            affine_comparison_tolerance=1e-12,
        )


# --- Final scientific audit unit tests ------------------------------------


def test_final_verifier_does_not_import_runner():
    source = Path(verify_final.__file__).read_text()
    assert "import run_p2_relu_residence" not in source


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ("hopper-medium-v2", "hopper_medium-v2.hdf5"),
        ("hopper-medium-replay-v2", "hopper_medium_replay-v2.hdf5"),
        ("hopper-expert-v2", "hopper_expert-v2.hdf5"),
        ("walker2d-medium-replay-v2", "walker2d_medium_replay-v2.hdf5"),
    ],
)
def test_final_dataset_filename_mapping_matches_runner_and_verifier(env, expected):
    assert p2.dataset_filename(env) == expected
    assert verify_final.dataset_filename(env) == expected


def test_final_sample_seeds_are_unique_and_agree_across_modules():
    seeds = {
        (env, seed): p2.final_sample_seed(env, seed)
        for env in p2.FINAL_ENVIRONMENTS
        for seed in p2.FINAL_SEEDS
    }
    assert len(set(seeds.values())) == len(seeds)
    for (env, seed), value in seeds.items():
        assert verify_final.final_sample_seed(env, seed) == value


def test_final_unique_horizons_collapse_and_agree():
    horizons = p2.unique_horizons()
    assert horizons == verify_final.unique_horizons()
    # Repeated (T, K) horizons collapse: fewer than the full 4x5 grid.
    assert len(horizons) < len(p2.TIMES) * len(p2.SUBSTEPS)
    assert horizons == sorted(set(horizons))


def _write_indices(path: Path, values):
    np.save(path, np.asarray(values, dtype=np.int64), allow_pickle=False)


def test_final_exclusion_inventory_and_env_scoped_application(tmp_path: Path):
    archived = tmp_path / "state_indices"
    archived.mkdir()
    _write_indices(archived / "hopper-medium-v2_seed0.npy", [1, 2, 3])
    _write_indices(archived / "hopper-medium-v2_seed1.npy", [3, 4, 5])
    _write_indices(archived / "walker2d-expert-v2_seed0.npy", [10, 11])
    _write_indices(archived / "walker2d-expert-v2_seed1.npy", [12, 13])

    bundle = tmp_path / "p2_relu_residence_pilot.ABCDEF"
    bundle.mkdir()
    _write_indices(bundle / "pilot_state_indices.npy", [1, 2])
    (bundle / "PILOT_INPUTS.json").write_text(
        json.dumps({"environment": "halfcheetah-medium-v2"})
    )

    inventory = p2.build_exclusion_inventory(
        archived, str(tmp_path / "p2_relu_residence_pilot.*")
    )
    roles = {record["role"] for record in inventory}
    assert roles == {"archived_fixed_operator", "dev_pilot_bundle"}
    dev = [record for record in inventory if record["role"] == "dev_pilot_bundle"]
    assert dev[0]["bundle_id"] == "ABCDEF"
    assert dev[0]["env"] == "halfcheetah-medium-v2"

    # Env-scoped numeric application: only the matching env's archived rows apply.
    excluded, applied = p2.applied_exclusions_for_env(inventory, "hopper-medium-v2")
    assert excluded == {1, 2, 3, 4, 5}
    assert len(applied) == 2
    # The HalfCheetah dev-bundle rows are documented but never numerically applied.
    walker_excluded, _ = p2.applied_exclusions_for_env(inventory, "walker2d-expert-v2")
    assert walker_excluded == {10, 11, 12, 13}

    broken = tmp_path / "p2_relu_residence_pilot.BROKEN"
    broken.mkdir()
    _write_indices(broken / "pilot_state_indices.npy", [7, 8])
    with pytest.raises(ValueError, match="environment is missing or unparseable"):
        p2.build_exclusion_inventory(
            archived, str(tmp_path / "p2_relu_residence_pilot.*")
        )


def test_final_select_indices_excludes_and_is_deterministic():
    actions = np.zeros((200, 3), dtype=np.float64)
    excluded = {0, 1, 2, 3, 4}
    first, n_eligible = p2.select_final_indices(actions, excluded, 32, sample_seed=777)
    second, _ = p2.select_final_indices(actions, excluded, 32, sample_seed=777)
    assert n_eligible == 195
    assert len(first) == 32
    assert np.array_equal(first, second)
    assert not excluded.intersection(int(value) for value in first)


def test_final_survival_and_pooled_aggregation_consistent():
    params, action = p2.synthetic_params("layer1_exit")  # exits at t=0.1
    geometry = p2.analyze_geometry(params, np.zeros((1, 0)), action, 1.0)
    horizons = p2.unique_horizons()
    survival = p2.survival_curve(geometry, horizons)
    # Single anchor: resident for horizons strictly below the 0.1 exit.
    for row in survival:
        expected = 1 if row["horizon"] < 0.1 - 1e-9 else 0
        assert row["n_resident"] == expected

    run = {"environment": "hopper-medium-v2", "seed": 0, "n_states": 1, "survival": survival}
    pooled = verify_final.recompute_pooled([run, run], horizons)
    for pooled_row, survival_row in zip(pooled, survival, strict=True):
        assert pooled_row["n_total"] == 2
        assert pooled_row["n_resident"] == 2 * survival_row["n_resident"]
        assert pooled_row["categories"] == {
            name: 2 * survival_row["categories"][name] for name in p2.CATEGORIES
        }

    tasks = {
        "hopper-medium-v2": pooled,
        "walker2d-medium-v2": pooled,
    }
    task_equal = p2.task_equal_survival(tasks, horizons)
    independently_recomputed = verify_final.recompute_task_equal(tasks, horizons)
    assert task_equal == independently_recomputed
    for row, survival_row in zip(task_equal, survival, strict=True):
        assert row["n_tasks"] == 2
        assert row["resident_fraction"] == survival_row["resident_fraction"]
        assert math.isclose(sum(row["category_fractions"].values()), 1.0)
