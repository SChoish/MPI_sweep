from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.diagnostics import profile_actor_cost as profiler
from scripts.diagnostics.verify_actor_cost import VerificationError, verify_directory


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "scripts" / "diagnostics" / "profile_actor_cost.py"
MEMORY_SCOPE = (
    "fresh worker process-lifetime backend high-water mark through dataset/batch "
    "transfer, actor/critic state initialization, lowering/compile, warmup, and "
    "timed trials; not an isolated steady-state actor-call peak"
)


def _digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


TEST_GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"


def _common_config(dataset_path: Path, *, fallback: bool = False, cpu: bool = False):
    return {
        "allow_cpu_smoke": cpu,
        "batch_size": 256,
        "calls_per_trial": 2,
        "dataset_path": str(dataset_path.resolve()),
        "device_selector": None if cpu else TEST_GPU_UUID,
        "init_seed": 0,
        "integrator": "implicit",
        "learning_rate": 3e-4,
        "memory_fallback": "process-rss" if fallback or cpu else "none",
        "method": "bar",
        "normalization_epsilon": 1e-3,
        "platform": "cpu" if cpu else "gpu",
        "polyak": 0.005,
        "production_actor_functions": [
            "train_td3bc.update_first_actor",
            "train_td3bc.update_actor_hop",
        ],
        "sample_seed": 20260902,
        "scale_norm": True,
        "tau": 12.0,
        "trials": 2,
        "warmup_calls": 1,
    }


def _create_source_repo(path: Path) -> tuple[Path, str, dict[str, str]]:
    path.mkdir()
    for relative in profiler.SOURCE_PATHS:
        source_path = path / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(f"# fixture source for {relative}\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=path, check=True)
    subprocess.run(("git", "add", "."), cwd=path, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Actor Cost Tests",
            "-c",
            "user.email=actor-cost@example.invalid",
            "commit",
            "-qm",
            "fixture sources",
        ),
        cwd=path,
        check=True,
    )
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files = {relative: _file_digest(path / relative) for relative in profiler.SOURCE_PATHS}
    return path, revision, files


def _worker_result(
    common,
    k: int,
    parent_pid: int,
    *,
    source_files: dict[str, str],
    git: dict,
    context_sha256: str,
):
    config = {**common, "k": k}
    calls = [[float(k), float(k)], [float(k), float(k)]]
    memory_source = (
        "process_rss_fallback_not_accelerator_memory"
        if common["memory_fallback"] == "process-rss"
        else "device.memory_stats"
    )
    phases = {
        name: {
            "source": memory_source,
            "current_bytes": 100 * k,
            "peak_bytes": 200 * k,
            "raw_stats": (
                {"bytes_in_use": 100 * k, "peak_bytes_in_use": 200 * k}
                if memory_source == "device.memory_stats"
                else None
            ),
        }
        for name in (
            "runtime_baseline",
            "before_compile",
            "after_compile",
            "after_warmup",
            "after_trials",
        )
    }
    cpu = common["platform"] == "cpu"
    environment = {
        "CUDA_VISIBLE_DEVICES": "" if cpu else TEST_GPU_UUID,
        "ROCR_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu" if cpu else "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    device = {
        "selector": common["device_selector"],
        "platform": common["platform"],
        "device_kind": "CPU" if cpu else "Test GPU",
        "id": 0,
        "local_hardware_id": 0,
        "process_index": 0,
        "uuid": None,
        "uuid_source": None,
    }
    device_binding = (
        {
            "method": "cpu_smoke_visibility",
            "resolved_uuid": None,
            "cuda_visible_devices": "",
            "jax_device_uuid": None,
            "jax_device_uuid_source": None,
        }
        if cpu
        else {
            "method": "resolved_uuid_visibility",
            "resolved_uuid": TEST_GPU_UUID,
            "cuda_visible_devices": TEST_GPU_UUID,
            "jax_device_uuid": None,
            "jax_device_uuid_source": None,
        }
    )
    provenance = {
        "git": git,
        "python": {"version": "3.12.0"},
        "packages": {"jax": "0.5.0", "jaxlib": "0.5.0"},
        "backend_platform_version": "CUDA test",
        "environment": environment,
        "device": device,
        "device_binding": device_binding,
        "visible_device_count": 1,
        "selected_driver_record": None
        if cpu
        else {
            "index": "0",
            "uuid": TEST_GPU_UUID,
            "name": "Test GPU",
            "driver_version": "test-driver",
        },
    }
    return {
        "schema_version": "bar-actor-cost-v1",
        "artifact": "actor_cost_worker_result",
        "k": k,
        "config": config,
        "hashes": {
            "config_sha256": _digest(config),
            "invariant_config_sha256": _digest(common),
            "code_sha256": _digest(source_files),
            "source_files": source_files,
            "dataset_sha256": _file_digest(Path(common["dataset_path"])),
            "batch_sha256": "e" * 64,
            "normalization_sha256": "f" * 64,
            "run_context_sha256": context_sha256,
        },
        "dataset_batch": {
            "dataset_path": common["dataset_path"],
            "dataset_rows_after_transition_filter": 1000,
            "observation_shape": [1000, 17],
            "action_shape": [1000, 6],
            "batch_size": common["batch_size"],
            "sample_rng": {"algorithm": "numpy.PCG64", "seed": common["sample_seed"]},
            "indices_sha256": "1" * 64,
            "normalization": "full filtered dataset mean/std plus epsilon",
        },
        "actor_update": {
            "scope": "actor_only_no_critic_update",
            "production_functions": common["production_actor_functions"],
            "tau_step": common["tau"] / k,
            "fixed_input_state_each_call": True,
            "synchronization": "every call blocks every JAX array leaf",
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "lowering_ms": 1.0,
            "compile_ms": 2.0,
            "warmup_ms": [3.0],
            "raw_call_ms": calls,
            "raw_trial_ms": [float(k), float(k)],
            "trial_statistic": "arithmetic mean of synchronized calls in that trial",
            "median_actor_update_ms": float(k),
        },
        "memory": {
            "source": memory_source,
            "fallback_selected": common["memory_fallback"],
            "is_accelerator_memory": memory_source == "device.memory_stats",
            "phases": phases,
            "phase_order": [
                "runtime_baseline",
                "before_compile",
                "after_compile",
                "after_warmup",
                "after_trials",
            ],
            "peak_counter_scope": MEMORY_SCOPE,
            "baseline_phase": "runtime_baseline",
            "absolute_peak_bytes": 200 * k,
            "baseline_current_bytes": 100 * k,
            "baseline_adjusted_peak_bytes": 100 * k,
            "baseline_adjusted_formula": (
                "max(0, absolute_peak_bytes - baseline_current_bytes)"
            ),
            "before_compile_resident_bytes": 100 * k,
            "after_warmup_resident_bytes": 100 * k,
            "after_trials_resident_bytes": 100 * k,
        },
        "process": {
            "pid": parent_pid + k,
            "parent_pid": parent_pid,
            "process_token": f"process-{k}",
            "started_utc": "2026-09-02T00:00:00+00:00",
        },
        "provenance": provenance,
    }


def _measurement_directory(
    path: Path, *, fallback: bool = False, cpu: bool = False
) -> Path:
    path.mkdir()
    dataset_path = path.parent / f"{path.name}-offline.hdf5"
    dataset_path.write_bytes(b"actor-cost-fixture-dataset\n")
    source_repo, revision, source_files = _create_source_repo(
        path.parent / f"{path.name}-source"
    )
    common = _common_config(dataset_path, fallback=fallback, cpu=cpu)
    parent_pid = 4000
    git = {
        "available": True,
        "revision": revision,
        "dirty": False,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
    }
    source = {
        "repository_path": str(source_repo),
        "files": source_files,
        "sha256": _digest(source_files),
    }
    dataset_sha256 = _file_digest(dataset_path)
    selected_driver_record = {
        "index": "0",
        "uuid": TEST_GPU_UUID,
        "name": "Test GPU",
        "driver_version": "test-driver",
    }
    driver_inventory = {"provider": "nvidia-smi", "records": [selected_driver_record]}
    host = {"hostname": "test-host", "machine": "x86_64", "system": "test-system"}
    python = {"version": "3.12.0"}
    packages = {"jax": "0.5.0", "jaxlib": "0.5.0"}
    parent_environment = {"CUDA_VISIBLE_DEVICES": None}
    exclusivity = {
        "operator_confirmed": True,
        "scope": "operator attestation plus cross-run harness file lock",
        "lock_path": "/tmp/test-actor-cost.lock",
    }
    orchestrator = {
        "pid": parent_pid,
        "invocation_id": "fixture-invocation",
        "fresh_process_per_k": True,
        "sequential_workers": True,
        "exact_k_set": [1, 2, 3, 4],
    }
    context = {
        "schema_version": "bar-actor-cost-v1",
        "created_utc": "2026-09-02T00:00:00+00:00",
        "orchestrator": orchestrator,
        "common_config": common,
        "dataset": {"path": str(dataset_path.resolve()), "sha256": dataset_sha256},
        "source": source,
        "git": git,
        "host": host,
        "python": python,
        "packages_before_worker": packages,
        "driver_inventory": driver_inventory,
        "selected_driver_record": None if cpu else selected_driver_record,
        "parent_environment": parent_environment,
        "exclusivity": exclusivity,
    }
    context_path = path / "RUN_CONTEXT.json"
    _write(context_path, context)
    context_sha256 = _file_digest(context_path)
    workers = {
        k: _worker_result(
            common,
            k,
            parent_pid,
            source_files=source_files,
            git=git,
            context_sha256=context_sha256,
        )
        for k in (1, 2, 3, 4)
    }
    for k, result in workers.items():
        _write(path / f"K={k}.json", result)
    source_files = workers[1]["hashes"]["source_files"]
    summaries = []
    for k, result in workers.items():
        summaries.append(
            {
                "k": k,
                "result_file": f"K={k}.json",
                "result_sha256": _file_digest(path / f"K={k}.json"),
                "raw_trial_ms": result["timing"]["raw_trial_ms"],
                "median_actor_update_ms": float(k),
                "time_ratio_to_k1": float(k),
                "absolute_peak_bytes": 200 * k,
                "absolute_peak_ratio_to_k1": float(k),
                "baseline_adjusted_peak_bytes": 100 * k,
                "baseline_adjusted_peak_ratio_to_k1": float(k),
            }
        )
    provenance = workers[1]["provenance"]
    manifest = {
        "schema_version": "bar-actor-cost-v1",
        "artifact": "actor_cost_manifest",
        "status": "cpu_smoke" if cpu else "fallback_smoke" if fallback else "measured",
        "accelerator_evidence": not (fallback or cpu),
        "exact_k_set": [1, 2, 3, 4],
        "only_swept_field": "k",
        "common_config": common,
        "invariant_config_sha256": _digest(common),
        "run_context": {"file": "RUN_CONTEXT.json", "sha256": context_sha256},
        "hashes": {
            "code_sha256": _digest(source_files),
            "invariant_config_sha256": _digest(common),
            "dataset_sha256": dataset_sha256,
            "batch_sha256": "e" * 64,
            "normalization_sha256": "f" * 64,
            "run_context_sha256": context_sha256,
        },
        "source": source,
        "dataset": {
            "path": common["dataset_path"],
            "sha256": dataset_sha256,
            "rows_after_transition_filter": 1000,
            "observation_shape": [1000, 17],
            "action_shape": [1000, 6],
        },
        "git": git,
        "host": host,
        "python": provenance["python"],
        "packages": provenance["packages"],
        "backend_platform_version": provenance["backend_platform_version"],
        "device": provenance["device"],
        "device_binding": provenance["device_binding"],
        "worker_environment": provenance["environment"],
        "selected_driver_record": provenance["selected_driver_record"],
        "driver_inventory": driver_inventory,
        "parent_environment": parent_environment,
        "exclusivity": exclusivity,
        "orchestrator": orchestrator,
        "memory_scope": MEMORY_SCOPE,
        "results": summaries,
    }
    _write(path / "MANIFEST.json", manifest)
    return path


def _verify_bundle(path: Path):
    manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
    return verify_directory(path, source_repo=Path(manifest["source"]["repository_path"]))


def _reseal_bundle(path: Path) -> None:
    manifest_path = path / "MANIFEST.json"
    context_path = path / "RUN_CONTEXT.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["common_config"] = deepcopy(manifest["common_config"])
    context["source"] = deepcopy(manifest["source"])
    context["dataset"] = {
        "path": manifest["dataset"]["path"],
        "sha256": manifest["dataset"]["sha256"],
    }
    for field in (
        "git",
        "host",
        "python",
        "driver_inventory",
        "selected_driver_record",
        "parent_environment",
        "exclusivity",
        "orchestrator",
    ):
        context[field] = deepcopy(manifest[field])
    context["packages_before_worker"] = deepcopy(manifest["packages"])
    _write(context_path, context)
    context_sha256 = _file_digest(context_path)

    manifest["invariant_config_sha256"] = _digest(manifest["common_config"])
    manifest["source"]["sha256"] = _digest(manifest["source"]["files"])
    manifest["run_context"] = {"file": "RUN_CONTEXT.json", "sha256": context_sha256}
    manifest["hashes"] = {
        "code_sha256": manifest["source"]["sha256"],
        "invariant_config_sha256": _digest(manifest["common_config"]),
        "dataset_sha256": manifest["dataset"]["sha256"],
        "batch_sha256": manifest["hashes"]["batch_sha256"],
        "normalization_sha256": manifest["hashes"]["normalization_sha256"],
        "run_context_sha256": context_sha256,
    }
    provenance_fields = {
        "git": "git",
        "python": "python",
        "packages": "packages",
        "backend_platform_version": "backend_platform_version",
        "device": "device",
        "device_binding": "device_binding",
        "worker_environment": "environment",
        "selected_driver_record": "selected_driver_record",
    }
    for k in (1, 2, 3, 4):
        result_path = path / f"K={k}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["config"] = {**manifest["common_config"], "k": k}
        result["hashes"].update(
            {
                "config_sha256": _digest(result["config"]),
                "invariant_config_sha256": _digest(manifest["common_config"]),
                "code_sha256": manifest["source"]["sha256"],
                "source_files": deepcopy(manifest["source"]["files"]),
                "dataset_sha256": manifest["dataset"]["sha256"],
                "run_context_sha256": context_sha256,
            }
        )
        result["dataset_batch"].update(
            {
                "dataset_path": manifest["dataset"]["path"],
                "dataset_rows_after_transition_filter": manifest["dataset"][
                    "rows_after_transition_filter"
                ],
                "observation_shape": deepcopy(manifest["dataset"]["observation_shape"]),
                "action_shape": deepcopy(manifest["dataset"]["action_shape"]),
            }
        )
        for manifest_field, worker_field in provenance_fields.items():
            result["provenance"][worker_field] = deepcopy(manifest[manifest_field])
        _write(result_path, result)
        summary = next(row for row in manifest["results"] if row["k"] == k)
        summary["result_sha256"] = _file_digest(result_path)
    _write(manifest_path, manifest)


def _reseal_context_reference_only(path: Path) -> None:
    manifest_path = path / "MANIFEST.json"
    context_path = path / "RUN_CONTEXT.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context_sha256 = _file_digest(context_path)
    manifest["run_context"] = {"file": "RUN_CONTEXT.json", "sha256": context_sha256}
    manifest["hashes"]["run_context_sha256"] = context_sha256
    for k in (1, 2, 3, 4):
        result_path = path / f"K={k}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["hashes"]["run_context_sha256"] = context_sha256
        _write(result_path, result)
        summary = next(row for row in manifest["results"] if row["k"] == k)
        summary["result_sha256"] = _file_digest(result_path)
    _write(manifest_path, manifest)


def test_default_plan_is_side_effect_free_even_when_jax_and_dataset_are_unavailable(
    tmp_path: Path,
):
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "jax.py").write_text(
        "raise RuntimeError('plan imported JAX')\n", encoding="utf-8"
    )
    missing_dataset = tmp_path / "must-not-be-opened.hdf5"
    output = tmp_path / "must-not-be-created"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(poison), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PROFILE_SCRIPT),
            "--dataset-path",
            str(missing_dataset),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    plan = json.loads(completed.stdout)
    assert plan["mode"] == "plan"
    assert plan["k_values"] == [1, 2, 3, 4]
    assert all(value is False for value in plan["side_effects"].values())
    assert not output.exists()
    assert not missing_dataset.exists()


class _NoMemoryStatsDevice:
    @staticmethod
    def memory_stats():
        return None


def test_memory_stats_unavailable_fails_closed_by_default():
    with pytest.raises(RuntimeError, match=r"device\.memory_stats.*did not provide"):
        profiler._memory_probe(_NoMemoryStatsDevice(), "none")


def test_explicit_rss_fallback_is_labeled_not_accelerator_memory():
    probe = profiler._memory_probe(_NoMemoryStatsDevice(), "process-rss")
    assert probe["source"] == "process_rss_fallback_not_accelerator_memory"
    assert probe["current_bytes"] > 0
    assert probe["peak_bytes"] > 0


def _gpu_context(*, fallback: str = "none", dirty: bool = False, record: bool = True):
    return {
        "common_config": {
            "platform": "gpu",
            "device_selector": "GPU-fixed",
            "memory_fallback": fallback,
        },
        "git": {
            "available": True,
            "revision": "deadbeef",
            "dirty": dirty,
        },
        "selected_driver_record": (
            {"index": "0", "uuid": "GPU-fixed", "driver_version": "test"}
            if record
            else None
        ),
    }


def test_gpu_context_rejects_unresolved_selector():
    context = _gpu_context(record=False)
    with pytest.raises(RuntimeError, match="fixed accelerator UUID"):
        profiler._validate_run_context(context)


def test_release_context_rejects_dirty_git_but_explicit_fallback_is_smoke_only():
    release = _gpu_context(dirty=True)
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        profiler._validate_run_context(release)
    smoke = _gpu_context(fallback="process-rss", dirty=True)
    profiler._validate_run_context(smoke)


def test_verifier_accepts_complete_exact_k_bundle(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "valid")
    verified = _verify_bundle(directory)
    assert verified["status"] == "verified"
    assert verified["k_values"] == [1, 2, 3, 4]
    assert verified["evidence_status"] == "measured"


def test_verifier_accepts_explicit_fallback_as_smoke_not_accelerator_evidence(
    tmp_path: Path,
):
    directory = _measurement_directory(tmp_path / "fallback", fallback=True)
    verified = _verify_bundle(directory)
    assert verified["evidence_status"] == "fallback_smoke"


def test_verifier_accepts_cpu_only_as_non_evidentiary_smoke(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "cpu", cpu=True)
    verified = _verify_bundle(directory)
    assert verified["evidence_status"] == "cpu_smoke"


def test_verifier_rejects_dirty_git_artifact_labeled_measured(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "dirty")
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git"]["dirty"] = True
    for k in (1, 2, 3, 4):
        result_path = directory / f"K={k}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["provenance"]["git"]["dirty"] = True
        _write(result_path, result)
        summary = next(row for row in manifest["results"] if row["k"] == k)
        summary["result_sha256"] = _file_digest(result_path)
    _write(manifest_path, manifest)

    with pytest.raises(VerificationError, match="available clean Git revision"):
        _verify_bundle(directory)


def test_verifier_rejects_non_k_config_corruption_even_with_updated_hashes(
    tmp_path: Path,
):
    directory = _measurement_directory(tmp_path / "corrupt")
    path = directory / "K=2.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result = deepcopy(result)
    result["config"]["batch_size"] = 512
    result["dataset_batch"]["batch_size"] = 512
    invariant = dict(result["config"])
    del invariant["k"]
    result["hashes"]["config_sha256"] = _digest(result["config"])
    result["hashes"]["invariant_config_sha256"] = _digest(invariant)
    _write(path, result)
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = next(row for row in manifest["results"] if row["k"] == 2)
    summary["result_sha256"] = _file_digest(path)
    _write(manifest_path, manifest)

    with pytest.raises(VerificationError, match="K is the only config field allowed to vary"):
        _verify_bundle(directory)


def test_verifier_rejects_worker_file_hash_corruption(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "file-corrupt")
    result_path = directory / "K=3.json"
    result_path.write_text(result_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(VerificationError, match="worker result file hash mismatch"):
        _verify_bundle(directory)


def test_verifier_rejects_missing_exact_k_result(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "missing")
    (directory / "K=4.json").unlink()
    with pytest.raises(VerificationError, match="worker result files must be exactly"):
        _verify_bundle(directory)


@pytest.mark.parametrize(
    ("configured_platform", "observed_platform"),
    (("tpu", "tpu"), ("gpu", "tpu")),
)
def test_verifier_rejects_non_gpu_measured_bundle_after_coherent_reseal(
    tmp_path: Path, configured_platform: str, observed_platform: str
):
    directory = _measurement_directory(tmp_path / f"platform-{configured_platform}")
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["common_config"]["platform"] = configured_platform
    manifest["device"]["platform"] = observed_platform
    _write(manifest_path, manifest)
    _reseal_bundle(directory)

    with pytest.raises(
        VerificationError, match="unsupported configured platform|configured and observed"
    ):
        _verify_bundle(directory)


def test_verifier_requires_run_context_file(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "missing-context")
    (directory / "RUN_CONTEXT.json").unlink()
    with pytest.raises(VerificationError, match="RUN_CONTEXT"):
        _verify_bundle(directory)


def test_verifier_rejects_run_context_byte_tampering(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "context-bytes")
    context_path = directory / "RUN_CONTEXT.json"
    context_path.write_text(context_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(VerificationError, match="RUN_CONTEXT.json file hash mismatch"):
        _verify_bundle(directory)


def test_verifier_rejects_coherently_rehashed_context_claim(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "context-claim")
    context_path = directory / "RUN_CONTEXT.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["host"]["hostname"] = "forged-host"
    _write(context_path, context)
    _reseal_context_reference_only(directory)

    with pytest.raises(VerificationError, match="RUN_CONTEXT.json host differs"):
        _verify_bundle(directory)


def test_verifier_hashes_actual_dataset_bytes(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "dataset-bytes")
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    Path(manifest["dataset"]["path"]).write_bytes(b"changed after measurement\n")
    with pytest.raises(VerificationError, match="dataset file hash differs"):
        _verify_bundle(directory)


def test_verifier_rejects_coherently_forged_dataset_digest(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "dataset-digest")
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["sha256"] = "a" * 64
    _write(manifest_path, manifest)
    _reseal_bundle(directory)
    with pytest.raises(VerificationError, match="dataset file hash differs"):
        _verify_bundle(directory)


def test_verifier_rejects_source_digest_not_in_recorded_git_revision(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "source-digest")
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["files"]["train_td3bc.py"] = "a" * 64
    _write(manifest_path, manifest)
    _reseal_bundle(directory)
    with pytest.raises(VerificationError, match="recorded Git revision"):
        _verify_bundle(directory)


def test_verifier_requires_exact_executed_source_set(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "source-set")
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["files"]["unreviewed_helper.py"] = "a" * 64
    _write(manifest_path, manifest)
    _reseal_bundle(directory)
    with pytest.raises(VerificationError, match="must contain exactly"):
        _verify_bundle(directory)


def test_verifier_rejects_missing_or_invalid_worker_shapes(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "shape-missing")
    result_path = directory / "K=3.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["dataset_batch"].pop("observation_shape")
    _write(result_path, result)
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(row for row in manifest["results"] if row["k"] == 3)[
        "result_sha256"
    ] = _file_digest(result_path)
    _write(manifest_path, manifest)
    with pytest.raises(VerificationError, match="observation shape"):
        _verify_bundle(directory)


def test_verifier_rejects_manifest_worker_shape_drift(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "shape-drift")
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["action_shape"] = [1000, 7]
    _write(manifest_path, manifest)
    with pytest.raises(VerificationError, match="dataset dimensions differ"):
        _verify_bundle(directory)


def test_device_identity_does_not_backfill_unobserved_uuid():
    class DeviceWithoutUuid:
        platform = "gpu"
        device_kind = "Test GPU"
        id = 0
        local_hardware_id = 0
        process_index = 0

    identity = profiler._device_identity(DeviceWithoutUuid(), TEST_GPU_UUID)
    assert identity["uuid"] is None
    assert identity["uuid_source"] is None


def test_verifier_rejects_uuid_copied_from_driver_record_as_observation(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "uuid-source")
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["device"]["uuid"] = TEST_GPU_UUID
    manifest["device"]["uuid_source"] = "selected_driver_record"
    manifest["device_binding"]["jax_device_uuid"] = TEST_GPU_UUID
    manifest["device_binding"]["jax_device_uuid_source"] = "selected_driver_record"
    _write(manifest_path, manifest)
    _reseal_bundle(directory)
    with pytest.raises(VerificationError, match="JAX-observed UUID/source"):
        _verify_bundle(directory)


def test_verifier_rejects_gpu_environment_not_bound_to_resolved_uuid(tmp_path: Path):
    directory = _measurement_directory(tmp_path / "uuid-environment")
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["worker_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    manifest["device_binding"]["cuda_visible_devices"] = "0"
    _write(manifest_path, manifest)
    _reseal_bundle(directory)
    with pytest.raises(VerificationError, match="CUDA visibility"):
        _verify_bundle(directory)
