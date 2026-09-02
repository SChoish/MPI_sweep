import copy
import json
import subprocess
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


DIAGNOSTICS = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

import run_p2_relu_residence as p2  # noqa: E402
import verify_p2_relu_residence as verify_p2  # noqa: E402


def test_import_does_not_enable_jax_x64_globally():
    program = f"""
import jax
before = bool(jax.config.jax_enable_x64)
import sys
sys.path.insert(0, {str(DIAGNOSTICS)!r})
import run_p2_relu_residence
after = bool(jax.config.jax_enable_x64)
assert after == before, (before, after)
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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


FRESH_BUNDLE = Path(
    "/home/ext_csv/mpi_sweep_lab/p2_relu_residence_pilot.sqfIVj"
)


def copy_verified_bundle(tmp_path: Path, name: str) -> Path:
    if not FRESH_BUNDLE.is_dir():
        pytest.skip(f"external development bundle unavailable: {FRESH_BUNDLE}")
    target = tmp_path / name
    shutil.copytree(FRESH_BUNDLE, target)
    (target / "VERIFY.json").unlink()
    return target


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_bundle_rejects_summary_derived_field_tamper(tmp_path: Path):
    bundle = copy_verified_bundle(tmp_path, "summary_tamper")
    summary_path = bundle / "SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["cells"][0]["h"] += 1e-3
    write_json(summary_path, summary)

    # Update the shallow artifact digest so the typed semantic recomputation,
    # rather than only MANIFEST hashing, must catch the altered derived value.
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["SUMMARY.json"] = verify_p2.sha256_file(summary_path)
    write_json(manifest_path, manifest)
    with pytest.raises(AssertionError, match="SUMMARY .* float mismatch: h"):
        verify_p2.verify_bundle(bundle)


def test_bundle_rejects_source_snapshot_byte_tamper(tmp_path: Path):
    bundle = copy_verified_bundle(tmp_path, "source_tamper")
    snapshot = bundle / "SOURCE_SNAPSHOT" / "run_p2_relu_residence.py"
    snapshot.write_bytes(snapshot.read_bytes() + b"\n# tampered\n")
    with pytest.raises(AssertionError, match="artifact hash mismatch"):
        verify_p2.verify_bundle(bundle)


def test_bundle_rejects_input_lock_tamper(tmp_path: Path):
    bundle = copy_verified_bundle(tmp_path, "input_lock_tamper")
    lock_path = bundle / "INPUT_LOCK.json"
    lock = json.loads(lock_path.read_text())
    lock["status"] = "tampered"
    write_json(lock_path, lock)
    with pytest.raises(AssertionError, match="INPUT_LOCK canonical hash mismatch"):
        verify_p2.verify_bundle(bundle)


def test_bundle_rejects_manifest_provenance_tamper(tmp_path: Path):
    bundle = copy_verified_bundle(tmp_path, "provenance_tamper")
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["git_provenance"]["git_revision"] = "not-a-revision"
    write_json(manifest_path, manifest)
    with pytest.raises(AssertionError, match="git revision provenance missing"):
        verify_p2.verify_bundle(bundle)
