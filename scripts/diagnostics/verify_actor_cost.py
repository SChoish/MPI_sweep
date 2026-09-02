#!/usr/bin/env python3
"""Independently verify a K=1..4 actor-cost measurement directory.

This module intentionally does not import ``profile_actor_cost``.  It
recomputes structural hashes, timing summaries, ratios, memory derivations,
and cross-worker invariants from the serialized artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_SCHEMA = "bar-actor-cost-v1"
EXPECTED_KS = (1, 2, 3, 4)
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE_PATHS = {
    "d4rl_data.py",
    "scripts/diagnostics/profile_actor_cost.py",
    "scripts/diagnostics/verify_actor_cost.py",
    "train_td3bc.py",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GIT_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z"
)
EXPECTED_PHASES = {
    "runtime_baseline",
    "before_compile",
    "after_compile",
    "after_warmup",
    "after_trials",
}
EXPECTED_MEMORY_SCOPE = (
    "fresh worker process-lifetime backend high-water mark through dataset/batch "
    "transfer, actor/critic state initialization, lowering/compile, warmup, and "
    "timed trials; not an isolated steady-state actor-call peak"
)


class VerificationError(ValueError):
    """The measurement artifacts violate the actor-cost protocol."""


def _fail(message: str) -> None:
    raise VerificationError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"non-canonical JSON value: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_blob_sha256(repository: Path, revision: str, relative_path: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "cat-file", "blob", f"{revision}:{relative_path}"),
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as exc:
        raise VerificationError(
            f"cannot read committed source blob {revision}:{relative_path} from {repository}"
        ) from exc
    return hashlib.sha256(completed.stdout).hexdigest()


def _verify_source_binding(
    source: Mapping[str, Any],
    git: Mapping[str, Any],
    repository: Path,
    *,
    measured: bool,
) -> None:
    source_files = source.get("files")
    if not isinstance(source_files, dict) or set(source_files) != EXPECTED_SOURCE_PATHS:
        _fail(f"source hash bundle must contain exactly {sorted(EXPECTED_SOURCE_PATHS)}")
    for relative, expected in source_files.items():
        expected = _require_sha256(expected, f"source hash for {relative}")
        if measured:
            revision = git.get("revision")
            if (
                not isinstance(revision, str)
                or GIT_REVISION_PATTERN.fullmatch(revision) is None
            ):
                _fail("release measurement requires a full hexadecimal Git revision")
            actual = _git_blob_sha256(repository, revision, relative)
        else:
            candidate = repository / relative
            if not candidate.is_file():
                _fail(f"source file is unavailable for smoke verification: {candidate}")
            actual = _file_sha256(candidate)
        if actual != expected:
            anchor = "recorded Git revision" if measured else "verification source tree"
            _fail(f"source hash for {relative} differs from the {anchor}")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"expected a JSON object in {path}")
    return value


def _require_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        _fail(f"{label} is missing keys: {missing}")


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        _fail(f"{label} must be finite and positive")
    return number


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{label} must be a nonnegative integer")
    return value


def _shape(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        _fail(f"{label} must be a nonempty list of positive integers")
    return value


def _close(actual: Any, expected: Any, label: str) -> None:
    if expected is None:
        if actual is not None:
            _fail(f"{label}: expected null, got {actual!r}")
        return
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        _fail(f"{label} must be numeric")
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        _fail(f"{label}: {actual!r} != recomputed {expected!r}")


def _without_k(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if "k" not in normalized:
        _fail("worker config is missing k")
    del normalized["k"]
    return normalized


def _verify_timing(result: Mapping[str, Any], k: int) -> float:
    timing = result.get("timing")
    config = result.get("config")
    if not isinstance(timing, dict) or not isinstance(config, dict):
        _fail(f"K={k}: timing and config must be objects")
    _require_keys(
        timing,
        {
            "clock",
            "lowering_ms",
            "compile_ms",
            "warmup_ms",
            "raw_call_ms",
            "raw_trial_ms",
            "trial_statistic",
            "median_actor_update_ms",
        },
        f"K={k} timing",
    )
    if timing["clock"] != "time.perf_counter_ns":
        _fail(f"K={k}: timing clock is not the protocol clock")
    if timing["trial_statistic"] != "arithmetic mean of synchronized calls in that trial":
        _fail(f"K={k}: timing trial statistic is not the protocol statistic")
    _positive_finite(timing["lowering_ms"], f"K={k} lowering_ms")
    _positive_finite(timing["compile_ms"], f"K={k} compile_ms")
    warmups = timing["warmup_ms"]
    trials = timing["raw_trial_ms"]
    calls_by_trial = timing["raw_call_ms"]
    if not isinstance(warmups, list) or len(warmups) != config.get("warmup_calls"):
        _fail(f"K={k}: wrong warmup count")
    for index, value in enumerate(warmups):
        _positive_finite(value, f"K={k} warmup[{index}]")
    if not isinstance(trials, list) or len(trials) != config.get("trials"):
        _fail(f"K={k}: wrong raw trial count")
    if not isinstance(calls_by_trial, list) or len(calls_by_trial) != len(trials):
        _fail(f"K={k}: raw_call_ms must have one list per trial")
    recomputed_trials = []
    for trial_index, calls in enumerate(calls_by_trial):
        if not isinstance(calls, list) or len(calls) != config.get("calls_per_trial"):
            _fail(f"K={k}: wrong call count in trial {trial_index}")
        checked = [
            _positive_finite(value, f"K={k} trial {trial_index} call {call_index}")
            for call_index, value in enumerate(calls)
        ]
        recomputed = statistics.fmean(checked)
        _close(trials[trial_index], recomputed, f"K={k} raw trial {trial_index}")
        recomputed_trials.append(recomputed)
    median = statistics.median(recomputed_trials)
    _close(timing["median_actor_update_ms"], median, f"K={k} timing median")
    return median


def _verify_memory(result: Mapping[str, Any], k: int) -> tuple[int, int, str, str]:
    memory = result.get("memory")
    config = result.get("config")
    if not isinstance(memory, dict) or not isinstance(config, dict):
        _fail(f"K={k}: memory and config must be objects")
    _require_keys(
        memory,
        {
            "source",
            "fallback_selected",
            "is_accelerator_memory",
            "phases",
            "phase_order",
            "peak_counter_scope",
            "baseline_phase",
            "absolute_peak_bytes",
            "baseline_current_bytes",
            "baseline_adjusted_peak_bytes",
            "before_compile_resident_bytes",
            "after_warmup_resident_bytes",
            "after_trials_resident_bytes",
        },
        f"K={k} memory",
    )
    source = memory["source"]
    fallback = memory["fallback_selected"]
    if fallback != config.get("memory_fallback"):
        _fail(f"K={k}: memory fallback differs from config")
    if source == "device.memory_stats":
        if memory["is_accelerator_memory"] is not True:
            _fail(f"K={k}: device memory must be labeled accelerator memory")
    elif source == "process_rss_fallback_not_accelerator_memory":
        if fallback != "process-rss":
            _fail(f"K={k}: RSS fallback was not explicitly selected")
        if memory["is_accelerator_memory"] is not False:
            _fail(f"K={k}: RSS fallback must not be labeled accelerator memory")
    else:
        _fail(f"K={k}: unsupported memory source {source!r}")
    phases = memory["phases"]
    if not isinstance(phases, dict) or set(phases) != EXPECTED_PHASES:
        _fail(f"K={k}: memory phases must be exactly {sorted(EXPECTED_PHASES)}")
    if memory["phase_order"] != [
        "runtime_baseline", "before_compile", "after_compile", "after_warmup", "after_trials"
    ]:
        _fail(f"K={k}: invalid memory phase order")
    if memory["peak_counter_scope"] != EXPECTED_MEMORY_SCOPE:
        _fail(f"K={k}: memory peak scope is missing or misleading")
    peaks = []
    for phase_name, phase in phases.items():
        if not isinstance(phase, dict) or phase.get("source") != source:
            _fail(f"K={k}: invalid or changing memory source at {phase_name}")
        current = _nonnegative_int(phase.get("current_bytes"), f"K={k} {phase_name} current")
        peak = _nonnegative_int(phase.get("peak_bytes"), f"K={k} {phase_name} peak")
        if peak < current:
            _fail(f"K={k}: backend peak is below current bytes at {phase_name}")
        if source == "device.memory_stats":
            raw = phase.get("raw_stats")
            if (
                not isinstance(raw, dict)
                or raw.get("bytes_in_use") != current
                or raw.get("peak_bytes_in_use") != peak
            ):
                _fail(f"K={k}: serialized backend memory_stats disagree at {phase_name}")
        elif phase.get("raw_stats") is not None:
            _fail(f"K={k}: RSS fallback must not contain backend raw stats")
        peaks.append(peak)
    if memory["baseline_phase"] != "runtime_baseline":
        _fail(f"K={k}: baseline phase must be runtime_baseline")
    absolute = _nonnegative_int(memory["absolute_peak_bytes"], f"K={k} absolute peak")
    if absolute != max(peaks):
        _fail(f"K={k}: absolute peak does not equal the maximum backend peak")
    baseline = _nonnegative_int(memory["baseline_current_bytes"], f"K={k} baseline")
    if baseline != phases["runtime_baseline"]["current_bytes"]:
        _fail(f"K={k}: baseline bytes do not match runtime_baseline")
    resident_fields = {
        "before_compile_resident_bytes": "before_compile",
        "after_warmup_resident_bytes": "after_warmup",
        "after_trials_resident_bytes": "after_trials",
    }
    for field, phase_name in resident_fields.items():
        if memory[field] != phases[phase_name]["current_bytes"]:
            _fail(f"K={k}: {field} does not match its phase")
    adjusted = _nonnegative_int(
        memory["baseline_adjusted_peak_bytes"], f"K={k} adjusted peak"
    )
    if adjusted != max(0, absolute - baseline):
        _fail(f"K={k}: baseline-adjusted peak formula mismatch")
    return absolute, adjusted, source, memory["peak_counter_scope"]


def _verify_result(
    result: Mapping[str, Any], k: int, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if result.get("schema_version") != EXPECTED_SCHEMA:
        _fail(f"K={k}: unsupported schema version")
    if result.get("artifact") != "actor_cost_worker_result" or result.get("k") != k:
        _fail(f"K={k}: result identity mismatch")
    config = result.get("config")
    hashes = result.get("hashes")
    if not isinstance(config, dict) or not isinstance(hashes, dict):
        _fail(f"K={k}: config and hashes must be objects")
    if config.get("k") != k:
        _fail(f"K={k}: config.k mismatch")
    if hashes.get("config_sha256") != _digest(config):
        _fail(f"K={k}: config hash mismatch")
    invariant = _without_k(config)
    if hashes.get("invariant_config_sha256") != _digest(invariant):
        _fail(f"K={k}: invariant config hash mismatch")
    source_files = hashes.get("source_files")
    if not isinstance(source_files, dict) or hashes.get("code_sha256") != _digest(source_files):
        _fail(f"K={k}: code hash bundle mismatch")
    if hashes.get("dataset_sha256") != manifest.get("dataset", {}).get("sha256"):
        _fail(f"K={k}: dataset hash differs from manifest")
    if hashes.get("run_context_sha256") != manifest.get("run_context", {}).get("sha256"):
        _fail(f"K={k}: result is not bound to RUN_CONTEXT.json")
    batch = result.get("dataset_batch")
    actor_update = result.get("actor_update")
    if not isinstance(batch, dict) or not isinstance(actor_update, dict):
        _fail(f"K={k}: dataset_batch and actor_update must be objects")
    if batch.get("batch_size") != config.get("batch_size"):
        _fail(f"K={k}: batch size mismatch")
    rows = _nonnegative_int(
        batch.get("dataset_rows_after_transition_filter"), f"K={k} filtered dataset rows"
    )
    observation_shape = _shape(batch.get("observation_shape"), f"K={k} observation shape")
    action_shape = _shape(batch.get("action_shape"), f"K={k} action shape")
    if rows < 1 or observation_shape[0] != rows or action_shape[0] != rows:
        _fail(f"K={k}: observation/action shapes disagree with filtered row count")
    expected_rng = {"algorithm": "numpy.PCG64", "seed": config.get("sample_seed")}
    if batch.get("sample_rng") != expected_rng:
        _fail(f"K={k}: fixed batch RNG mismatch")
    if actor_update.get("scope") != "actor_only_no_critic_update":
        _fail(f"K={k}: benchmark is not actor-only")
    if actor_update.get("production_functions") != config.get("production_actor_functions"):
        _fail(f"K={k}: production actor function declaration mismatch")
    _close(actor_update.get("tau_step"), config.get("tau") / k, f"K={k} tau_step")
    if actor_update.get("fixed_input_state_each_call") is not True:
        _fail(f"K={k}: actor input state is not fixed across calls")
    if actor_update.get("synchronization") != "every call blocks every JAX array leaf":
        _fail(f"K={k}: timed calls are not declared synchronized")
    timing_median = _verify_timing(result, k)
    absolute_peak, adjusted_peak, memory_source, memory_scope = _verify_memory(result, k)
    process = result.get("process")
    provenance = result.get("provenance")
    if not isinstance(process, dict) or not isinstance(provenance, dict):
        _fail(f"K={k}: process and provenance must be objects")
    pid = process.get("pid")
    parent_pid = process.get("parent_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        _fail(f"K={k}: invalid worker pid")
    if parent_pid != manifest.get("orchestrator", {}).get("pid"):
        _fail(f"K={k}: worker was not a direct fresh child of the orchestrator")
    token = process.get("process_token")
    if not isinstance(token, str) or not token:
        _fail(f"K={k}: missing process token")
    if provenance.get("visible_device_count") != 1:
        _fail(f"K={k}: exactly one JAX device must be visible")
    device = provenance.get("device")
    if not isinstance(device, dict):
        _fail(f"K={k}: missing device identity")
    platform = config.get("platform")
    if platform not in {"gpu", "cpu"}:
        _fail(f"K={k}: unsupported configured platform {platform!r}")
    if device.get("platform") != platform:
        _fail(f"K={k}: configured and observed device platforms differ")
    if platform == "cpu" and not config.get("allow_cpu_smoke"):
        _fail(f"K={k}: unlabeled CPU result")
    return {
        "config_invariant": invariant,
        "batch_invariant": batch,
        "hash_invariant": {
            name: hashes.get(name)
            for name in (
                "invariant_config_sha256",
                "code_sha256",
                "source_files",
                "dataset_sha256",
                "batch_sha256",
                "normalization_sha256",
                "run_context_sha256",
            )
        },
        "provenance_invariant": provenance,
        "pid": pid,
        "process_token": token,
        "median": timing_median,
        "absolute": absolute_peak,
        "adjusted": adjusted_peak,
        "memory_source": memory_source,
        "memory_scope": memory_scope,
    }


def _expected_ratio(value: int | float, baseline: int | float) -> float | None:
    return float(value) / float(baseline) if baseline > 0 else None


def verify_directory(
    directory: Path,
    *,
    source_repo: Path | None = None,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    root = directory.expanduser().resolve()
    repository = (REPO_ROOT if source_repo is None else source_repo).expanduser().resolve()
    manifest = _load_object(root / "MANIFEST.json")
    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        _fail("manifest schema mismatch")
    if manifest.get("artifact") != "actor_cost_manifest":
        _fail("not an actor-cost manifest")
    if manifest.get("exact_k_set") != list(EXPECTED_KS):
        _fail("manifest must declare exactly K=1,2,3,4")
    if manifest.get("only_swept_field") != "k":
        _fail("manifest must declare k as the only swept field")

    common = manifest.get("common_config")
    if not isinstance(common, dict):
        _fail("manifest common_config must be an object")
    platform = common.get("platform")
    fallback = common.get("memory_fallback")
    if platform not in {"gpu", "cpu"}:
        _fail(f"unsupported configured platform {platform!r}")
    if fallback not in {"none", "process-rss"}:
        _fail(f"unsupported memory fallback {fallback!r}")
    expected_status = (
        "cpu_smoke"
        if platform == "cpu"
        else "fallback_smoke"
        if fallback == "process-rss"
        else "measured"
    )
    if manifest.get("status") != expected_status:
        _fail(f"manifest status must be {expected_status!r}")
    if manifest.get("accelerator_evidence") is not (expected_status == "measured"):
        _fail("manifest accelerator_evidence label disagrees with status")
    if manifest.get("invariant_config_sha256") != _digest(common):
        _fail("manifest invariant config hash mismatch")

    git = manifest.get("git")
    if not isinstance(git, Mapping):
        _fail("manifest Git provenance must be an object")
    if expected_status == "measured":
        if git.get("available") is not True or git.get("dirty") is not False:
            _fail("release measurement requires an available clean Git revision")
        revision = git.get("revision")
        if not isinstance(revision, str) or GIT_REVISION_PATTERN.fullmatch(revision) is None:
            _fail("release measurement requires a full hexadecimal Git revision")

    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("sha256") != _digest(source.get("files")):
        _fail("manifest source hash bundle mismatch")
    _require_sha256(source.get("sha256"), "source bundle hash")
    _verify_source_binding(
        source,
        git,
        repository,
        measured=expected_status == "measured",
    )

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        _fail("manifest dataset must be an object")
    recorded_dataset_hash = _require_sha256(dataset.get("sha256"), "dataset hash")
    recorded_dataset_path = dataset.get("path")
    if not isinstance(recorded_dataset_path, str) or not recorded_dataset_path:
        _fail("manifest dataset path is missing")
    if common.get("dataset_path") != recorded_dataset_path:
        _fail("manifest dataset path differs from the invariant config")
    actual_dataset_path = (
        Path(recorded_dataset_path) if dataset_path is None else dataset_path
    ).expanduser().resolve()
    if not actual_dataset_path.is_file():
        _fail(f"dataset is unavailable for verification: {actual_dataset_path}")
    if _file_sha256(actual_dataset_path) != recorded_dataset_hash:
        _fail("dataset file hash differs from the recorded dataset hash")

    run_context_reference = manifest.get("run_context")
    if not isinstance(run_context_reference, dict):
        _fail("manifest RUN_CONTEXT reference must be an object")
    if run_context_reference.get("file") != "RUN_CONTEXT.json":
        _fail("manifest must use the canonical RUN_CONTEXT.json filename")
    context_sha256 = _require_sha256(
        run_context_reference.get("sha256"), "RUN_CONTEXT.json hash"
    )
    context_path = root / "RUN_CONTEXT.json"
    if _file_sha256(context_path) != context_sha256:
        _fail("RUN_CONTEXT.json file hash mismatch")
    context = _load_object(context_path)
    if context.get("schema_version") != EXPECTED_SCHEMA:
        _fail("RUN_CONTEXT.json schema mismatch")
    if context.get("common_config") != common:
        _fail("RUN_CONTEXT.json config differs from the manifest")
    if context.get("source") != source:
        _fail("RUN_CONTEXT.json source bundle differs from the manifest")
    context_dataset = context.get("dataset")
    if not isinstance(context_dataset, dict) or context_dataset != {
        "path": recorded_dataset_path,
        "sha256": recorded_dataset_hash,
    }:
        _fail("RUN_CONTEXT.json dataset binding differs from the manifest")
    context_fields = (
        "git",
        "host",
        "python",
        "driver_inventory",
        "selected_driver_record",
        "parent_environment",
        "exclusivity",
        "orchestrator",
    )
    for field in context_fields:
        if context.get(field) != manifest.get(field):
            _fail(f"RUN_CONTEXT.json {field} differs from the manifest")
    orchestrator = context.get("orchestrator")
    if not isinstance(orchestrator, dict) or (
        orchestrator.get("fresh_process_per_k") is not True
        or orchestrator.get("sequential_workers") is not True
        or orchestrator.get("exact_k_set") != list(EXPECTED_KS)
    ):
        _fail("RUN_CONTEXT.json does not declare the fresh sequential K protocol")

    manifest_hashes = manifest.get("hashes")
    expected_hash_names = {
        "code_sha256",
        "invariant_config_sha256",
        "dataset_sha256",
        "batch_sha256",
        "normalization_sha256",
        "run_context_sha256",
    }
    if not isinstance(manifest_hashes, dict) or set(manifest_hashes) != expected_hash_names:
        _fail("manifest must contain the exact source/config/data/context hash set")
    for name, value in manifest_hashes.items():
        _require_sha256(value, f"manifest {name}")
    if manifest_hashes["code_sha256"] != source["sha256"]:
        _fail("manifest code hash differs from source bundle")
    if manifest_hashes["invariant_config_sha256"] != _digest(common):
        _fail("manifest config hashes disagree")
    if manifest_hashes["dataset_sha256"] != recorded_dataset_hash:
        _fail("manifest dataset hashes disagree")
    if manifest_hashes["run_context_sha256"] != context_sha256:
        _fail("manifest RUN_CONTEXT hashes disagree")

    summaries = manifest.get("results")
    if not isinstance(summaries, list) or len(summaries) != len(EXPECTED_KS):
        _fail("manifest must contain four result summaries")
    summary_by_k: dict[int, dict[str, Any]] = {}
    for summary in summaries:
        if not isinstance(summary, dict) or isinstance(summary.get("k"), bool):
            _fail("invalid result summary")
        k = summary.get("k")
        if not isinstance(k, int) or k in summary_by_k:
            _fail("duplicate or invalid K in manifest results")
        summary_by_k[k] = summary
    if tuple(sorted(summary_by_k)) != EXPECTED_KS:
        _fail("result summaries must have the exact K set {1,2,3,4}")
    expected_names = {f"K={k}.json" for k in EXPECTED_KS}
    actual_names = {path.name for path in root.glob("K=*.json")}
    if actual_names != expected_names:
        _fail(f"worker result files must be exactly {sorted(expected_names)}")

    checked: dict[int, dict[str, Any]] = {}
    for k in EXPECTED_KS:
        summary = summary_by_k[k]
        if summary.get("result_file") != f"K={k}.json":
            _fail(f"K={k}: noncanonical result filename")
        result_path = root / f"K={k}.json"
        if summary.get("result_sha256") != _file_sha256(result_path):
            _fail(f"K={k}: worker result file hash mismatch")
        checked[k] = _verify_result(_load_object(result_path), k, manifest)

    baseline = checked[1]
    pids = {record["pid"] for record in checked.values()}
    tokens = {record["process_token"] for record in checked.values()}
    if len(pids) != len(EXPECTED_KS) or len(tokens) != len(EXPECTED_KS):
        _fail("fresh subprocess requirement failed: worker pid/token values are not unique")
    for k, record in checked.items():
        if record["config_invariant"] != common:
            _fail(f"K={k}: K is the only config field allowed to vary")
        for field in ("batch_invariant", "hash_invariant", "provenance_invariant"):
            if record[field] != baseline[field]:
                _fail(f"K={k}: non-K experimental invariant changed: {field}")

    expected_manifest_hashes = {
        "code_sha256": baseline["hash_invariant"]["code_sha256"],
        "invariant_config_sha256": baseline["hash_invariant"]["invariant_config_sha256"],
        "dataset_sha256": baseline["hash_invariant"]["dataset_sha256"],
        "batch_sha256": baseline["hash_invariant"]["batch_sha256"],
        "normalization_sha256": baseline["hash_invariant"]["normalization_sha256"],
        "run_context_sha256": baseline["hash_invariant"]["run_context_sha256"],
    }
    if manifest_hashes != expected_manifest_hashes:
        _fail("manifest hashes differ from verified worker invariants")

    batch = baseline["batch_invariant"]
    expected_dataset_fields = {
        "path": recorded_dataset_path,
        "sha256": recorded_dataset_hash,
        "rows_after_transition_filter": batch["dataset_rows_after_transition_filter"],
        "observation_shape": batch["observation_shape"],
        "action_shape": batch["action_shape"],
    }
    if dataset != expected_dataset_fields:
        _fail("manifest dataset dimensions differ from the verified workers")

    provenance = baseline["provenance_invariant"]
    if manifest.get("git") != provenance.get("git"):
        _fail("manifest Git provenance differs from workers")
    if context.get("packages_before_worker") != provenance.get("packages"):
        _fail("RUN_CONTEXT package versions differ from worker package versions")
    provenance_fields = {
        "python": "python",
        "packages": "packages",
        "backend_platform_version": "backend_platform_version",
        "device": "device",
        "device_binding": "device_binding",
        "worker_environment": "environment",
        "selected_driver_record": "selected_driver_record",
    }
    for manifest_field, worker_field in provenance_fields.items():
        if manifest.get(manifest_field) != provenance.get(worker_field):
            _fail(f"manifest {manifest_field} differs from workers")

    if manifest.get("memory_scope") != baseline["memory_scope"]:
        _fail("manifest memory scope differs from workers")
    exclusivity = manifest.get("exclusivity")
    if not isinstance(exclusivity, dict):
        _fail("manifest exclusivity must be an object")
    if exclusivity.get("operator_confirmed") is not True:
        _fail("exclusive accelerator was not operator-confirmed")
    if not isinstance(exclusivity.get("lock_path"), str) or not exclusivity["lock_path"]:
        _fail("exclusive accelerator lock provenance is missing")

    device = provenance.get("device")
    environment = provenance.get("environment")
    binding = provenance.get("device_binding")
    if not isinstance(device, dict) or not isinstance(environment, dict) or not isinstance(
        binding, dict
    ):
        _fail("worker device, environment, and binding provenance must be objects")
    if device.get("platform") != platform:
        _fail("configured and observed device platforms differ")
    if platform == "gpu":
        driver_record = manifest.get("selected_driver_record")
        selector = str(common.get("device_selector"))
        if not isinstance(driver_record, Mapping) or not driver_record.get("uuid"):
            _fail("GPU selector lacks a resolved driver/UUID record")
        resolved_uuid = str(driver_record["uuid"])
        if GPU_UUID_PATTERN.fullmatch(resolved_uuid) is None:
            _fail("resolved NVIDIA GPU UUID has an unsupported format")
        if not driver_record.get("name") or not driver_record.get("driver_version"):
            _fail("resolved GPU record lacks device-name or driver-version provenance")
        if selector not in (str(driver_record.get("index")), resolved_uuid):
            _fail("resolved driver record does not match the fixed GPU selector")
        inventory = manifest.get("driver_inventory")
        inventory_records = inventory.get("records") if isinstance(inventory, Mapping) else None
        if (
            not isinstance(inventory, Mapping)
            or inventory.get("provider") != "nvidia-smi"
            or not isinstance(inventory_records, list)
            or driver_record not in inventory_records
        ):
            _fail("resolved GPU record is not present in the nvidia-smi inventory")
        if device.get("selector") != common.get("device_selector"):
            _fail("worker device identity does not retain the requested selector")
        if environment.get("CUDA_VISIBLE_DEVICES") != resolved_uuid:
            _fail("worker CUDA visibility is not bound to the resolved GPU UUID")
        if environment.get("ROCR_VISIBLE_DEVICES") != "" or environment.get(
            "HIP_VISIBLE_DEVICES"
        ) != "":
            _fail("non-CUDA GPU visibility was not disabled for the resolved NVIDIA GPU")
        if environment.get("JAX_PLATFORMS") != "gpu":
            _fail("worker JAX platform selection is not GPU-only")
        if environment.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
            _fail("worker JAX GPU preallocation was not disabled")
        observed_uuid = device.get("uuid")
        observed_source = device.get("uuid_source")
        if observed_uuid is None:
            if observed_source is not None:
                _fail("missing JAX UUID must not claim an observation source")
        elif observed_uuid != resolved_uuid or observed_source not in {
            "jax.device.uuid",
            "jax.device.device_uuid",
        }:
            _fail("JAX-observed UUID/source does not match the resolved GPU")
        expected_binding = {
            "method": "resolved_uuid_visibility",
            "resolved_uuid": resolved_uuid,
            "cuda_visible_devices": resolved_uuid,
            "jax_device_uuid": observed_uuid,
            "jax_device_uuid_source": observed_source,
        }
        if binding != expected_binding:
            _fail("worker GPU environment/device binding is invalid")
    else:
        if common.get("allow_cpu_smoke") is not True:
            _fail("CPU result is not explicitly enabled as smoke-only")
        if manifest.get("selected_driver_record") is not None:
            _fail("CPU smoke must not claim a selected GPU record")
        if environment.get("CUDA_VISIBLE_DEVICES") != "" or environment.get(
            "JAX_PLATFORMS"
        ) != "cpu":
            _fail("CPU smoke worker environment is not CPU-only")
        if environment.get("ROCR_VISIBLE_DEVICES") != "" or environment.get(
            "HIP_VISIBLE_DEVICES"
        ) != "":
            _fail("CPU smoke worker left an accelerator visible")
        if environment.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
            _fail("CPU smoke worker preallocation provenance is invalid")
        if binding.get("method") != "cpu_smoke_visibility":
            _fail("CPU smoke device binding is mislabeled")

    any_fallback = any(
        record["memory_source"] != "device.memory_stats" for record in checked.values()
    )
    if expected_status == "measured" and any_fallback:
        _fail("release measurement cannot use a memory fallback")

    for k in EXPECTED_KS:
        summary = summary_by_k[k]
        record = checked[k]
        result = _load_object(root / f"K={k}.json")
        if summary.get("raw_trial_ms") != result["timing"]["raw_trial_ms"]:
            _fail(f"K={k}: manifest raw trials differ from worker result")
        _close(summary.get("median_actor_update_ms"), record["median"], f"K={k} summary median")
        _close(
            summary.get("time_ratio_to_k1"),
            _expected_ratio(record["median"], baseline["median"]),
            f"K={k} time ratio",
        )
        if summary.get("absolute_peak_bytes") != record["absolute"]:
            _fail(f"K={k}: manifest absolute peak mismatch")
        _close(
            summary.get("absolute_peak_ratio_to_k1"),
            _expected_ratio(record["absolute"], baseline["absolute"]),
            f"K={k} absolute memory ratio",
        )
        if summary.get("baseline_adjusted_peak_bytes") != record["adjusted"]:
            _fail(f"K={k}: manifest adjusted peak mismatch")
        _close(
            summary.get("baseline_adjusted_peak_ratio_to_k1"),
            _expected_ratio(record["adjusted"], baseline["adjusted"]),
            f"K={k} adjusted memory ratio",
        )
    return {
        "status": "verified",
        "directory": str(root),
        "k_values": list(EXPECTED_KS),
        "evidence_status": manifest["status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=REPO_ROOT,
        help="Git repository containing the recorded source revision (default: this checkout).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Override the recorded dataset path with an identical local copy.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            verify_directory(
                args.directory,
                source_repo=args.source_repo,
                dataset_path=args.dataset_path,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
