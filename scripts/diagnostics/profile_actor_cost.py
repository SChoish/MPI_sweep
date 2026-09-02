#!/usr/bin/env python3
"""Reproducible actor-only wall-time and accelerator-memory profiler.

Running this file without ``--execute`` is deliberately side-effect free: it
prints the planned K=1..4 experiment without importing JAX, opening a dataset,
creating an output directory, or launching a worker.  An executed sweep uses a
fresh, sequential subprocess for every K and exposes the same single device to
all four workers.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bar-actor-cost-v1"
EXACT_K_SET = (1, 2, 3, 4)
REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILER_PATH = Path(__file__).resolve()
TRAIN_PATH = REPO_ROOT / "train_td3bc.py"
VERIFIER_PATH = PROFILER_PATH.with_name("verify_actor_cost.py")
SOURCE_PATHS = (
    "d4rl_data.py",
    "scripts/diagnostics/profile_actor_cost.py",
    "scripts/diagnostics/verify_actor_cost.py",
    "train_td3bc.py",
)
DEFAULT_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "JAX_PLATFORMS",
    "JAX_PLATFORM_NAME",
    "JAX_ENABLE_X64",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_text(command: Sequence[str], cwd: Path = REPO_ROOT) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _run_bytes(command: Sequence[str], cwd: Path = REPO_ROOT) -> bytes | None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _git_provenance() -> dict[str, Any]:
    revision = _run_text(("git", "rev-parse", "HEAD"))
    status = _run_text(("git", "status", "--porcelain=v1", "--untracked-files=all"))
    if revision is None or status is None:
        return {
            "available": False,
            "revision": None,
            "dirty": None,
            "status_sha256": None,
        }
    return {
        "available": True,
        "revision": revision,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _source_hashes() -> dict[str, Any]:
    files = {relative: _file_sha256(REPO_ROOT / relative) for relative in SOURCE_PATHS}
    return {
        "repository_path": str(REPO_ROOT),
        "files": files,
        "sha256": _json_sha256(files),
    }


def _git_source_hashes(revision: str) -> dict[str, str]:
    files: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        blob = _run_bytes(("git", "cat-file", "blob", f"{revision}:{relative}"))
        if blob is None:
            raise RuntimeError(
                f"cannot read committed source blob {revision}:{relative}"
            )
        files[relative] = hashlib.sha256(blob).hexdigest()
    return files


def _require_source_snapshot(expected: Mapping[str, Any], phase: str) -> None:
    actual = _source_hashes()
    if actual.get("files") != expected.get("files") or actual.get("sha256") != expected.get(
        "sha256"
    ):
        raise RuntimeError(f"source files changed {phase}")


def _environment_snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in DEFAULT_ENV_KEYS}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("jax", "jaxlib", "numpy", "flax", "optax"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _driver_inventory() -> dict[str, Any]:
    query = _run_text(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    if query is None:
        return {"provider": None, "records": []}
    records = []
    for line in query.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            records.append(
                {
                    "index": fields[0],
                    "uuid": fields[1],
                    "name": fields[2],
                    "driver_version": fields[3],
                }
            )
    return {"provider": "nvidia-smi", "records": records}


def _resolved_gpu_record(
    inventory: Mapping[str, Any], selector: str
) -> dict[str, Any] | None:
    records = inventory.get("records", [])
    for record in records:
        if selector in (str(record.get("index")), str(record.get("uuid"))):
            return dict(record)
    return None


def _public_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the benchmark. Without this flag only a side-effect-free plan is printed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit alias for the default plan-only behavior.",
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--accelerator",
        help="One physical GPU index or UUID, passed to all four fresh workers.",
    )
    parser.add_argument("--platform", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--confirm-exclusive-accelerator",
        action="store_true",
        help=(
            "Attest that no unrelated workload uses the selected device; "
            "a harness lock is also held."
        ),
    )
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="Permit a clearly labeled CPU smoke run; never valid as accelerator evidence.",
    )
    parser.add_argument(
        "--memory-fallback",
        choices=("none", "process-rss"),
        default="none",
        help="Default is fail-closed when device.memory_stats() lacks required keys.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=20260902)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=12.0)
    parser.add_argument("--polyak", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--normalization-epsilon", type=float, default=1e-3)
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--calls-per-trial", type=int, default=10)
    return parser


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    return parser


def _common_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "allow_cpu_smoke": bool(args.allow_cpu_smoke),
        "batch_size": int(args.batch_size),
        "calls_per_trial": int(args.calls_per_trial),
        "dataset_path": (
            str(args.dataset_path.expanduser().resolve())
            if args.dataset_path is not None
            else None
        ),
        "device_selector": args.accelerator,
        "init_seed": int(args.init_seed),
        "integrator": "implicit",
        "learning_rate": float(args.learning_rate),
        "memory_fallback": args.memory_fallback,
        "method": "bar",
        "normalization_epsilon": float(args.normalization_epsilon),
        "platform": args.platform,
        "polyak": float(args.polyak),
        "production_actor_functions": [
            "train_td3bc.update_first_actor",
            "train_td3bc.update_actor_hop",
        ],
        "sample_seed": int(args.sample_seed),
        "scale_norm": True,
        "tau": float(args.tau),
        "trials": int(args.trials),
        "warmup_calls": int(args.warmup_calls),
    }


def _plan_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "plan",
        "side_effects": {
            "imports_jax": False,
            "opens_or_downloads_dataset": False,
            "launches_workers": False,
            "writes_output": False,
        },
        "k_values": list(EXACT_K_SET),
        "fresh_process_per_k": True,
        "fixed_single_visible_device": True,
        "actor_scope": "implicit BAR actor update only; critic update excluded",
        "compile_and_warmup_excluded_from_trials": True,
        "memory_policy": {
            "primary": "jax Device.memory_stats() peak_bytes_in_use",
            "unavailable_default": "fail_closed",
            "selected_fallback": args.memory_fallback,
            "fallback_is_accelerator_memory": False,
        },
        "requested_config": _common_config(args),
        "execute_requirements": [
            "--dataset-path must name an existing local HDF5 dataset",
            "--output-dir must be empty or absent",
            "--accelerator must select one fixed device",
            "--confirm-exclusive-accelerator must be supplied",
        ],
    }


def _validate_execute_args(args: argparse.Namespace) -> None:
    if args.dry_run:
        raise ValueError("--execute and --dry-run are mutually exclusive")
    if args.dataset_path is None:
        raise ValueError("--execute requires --dataset-path; datasets are never downloaded")
    dataset = args.dataset_path.expanduser().resolve()
    if not dataset.is_file():
        raise ValueError(f"dataset does not exist or is not a file: {dataset}")
    if args.output_dir is None:
        raise ValueError("--execute requires --output-dir")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"--output-dir must be absent or empty: {output}")
    if args.platform == "gpu" and not args.accelerator:
        raise ValueError("GPU execution requires --accelerator INDEX_OR_UUID")
    if args.platform == "cpu" and not args.allow_cpu_smoke:
        raise ValueError("CPU execution requires --allow-cpu-smoke and is not release evidence")
    if not args.confirm_exclusive_accelerator:
        raise ValueError("execution requires --confirm-exclusive-accelerator")
    positive_ints = {
        "batch-size": args.batch_size,
        "warmup-calls": args.warmup_calls,
        "trials": args.trials,
        "calls-per-trial": args.calls_per_trial,
    }
    invalid = [name for name, value in positive_ints.items() if value < 1]
    if invalid:
        raise ValueError("positive integer required for " + ", ".join(invalid))
    if args.tau <= 0 or args.learning_rate <= 0:
        raise ValueError("--tau and --learning-rate must be positive")
    if not 0 < args.polyak <= 1:
        raise ValueError("--polyak must lie in (0, 1]")
    if args.normalization_epsilon <= 0:
        raise ValueError("--normalization-epsilon must be positive")


class _AcceleratorLock:
    def __init__(self, selector: str) -> None:
        token = hashlib.sha256(selector.encode("utf-8")).hexdigest()[:20]
        self.path = Path(tempfile.gettempdir()) / f"bar-actor-cost-{token}.lock"
        self._handle: Any = None

    def __enter__(self) -> "_AcceleratorLock":
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another actor-cost harness holds the accelerator lock: {self.path}"
            ) from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()} started={_utc_now()}\n")
        self._handle.flush()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()


def _worker_environment(
    config: Mapping[str, Any], selected_driver_record: Mapping[str, Any] | None
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if config["platform"] == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["ROCR_VISIBLE_DEVICES"] = ""
        environment["HIP_VISIBLE_DEVICES"] = ""
        environment["JAX_PLATFORMS"] = "cpu"
    else:
        if not isinstance(selected_driver_record, Mapping) or not selected_driver_record.get(
            "uuid"
        ):
            raise RuntimeError("GPU worker environment requires a resolved device UUID")
        resolved_uuid = str(selected_driver_record["uuid"])
        environment["CUDA_VISIBLE_DEVICES"] = resolved_uuid
        environment["ROCR_VISIBLE_DEVICES"] = ""
        environment["HIP_VISIBLE_DEVICES"] = ""
        environment["JAX_PLATFORMS"] = "gpu"
    return environment


def _worker_command(context: Path, result: Path, k: int) -> list[str]:
    return [
        sys.executable,
        str(PROFILER_PATH),
        "_worker",
        "--context",
        str(context),
        "--result",
        str(result),
        "--k",
        str(k),
    ]


def _create_context(args: argparse.Namespace) -> dict[str, Any]:
    config = _common_config(args)
    dataset = Path(str(config["dataset_path"]))
    sources = _source_hashes()
    driver = _driver_inventory()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "orchestrator": {
            "pid": os.getpid(),
            "invocation_id": str(uuid.uuid4()),
            "fresh_process_per_k": True,
            "sequential_workers": True,
            "exact_k_set": list(EXACT_K_SET),
        },
        "common_config": config,
        "dataset": {"path": str(dataset), "sha256": _file_sha256(dataset)},
        "source": sources,
        "git": _git_provenance(),
        "host": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "system": platform.platform(),
        },
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages_before_worker": _package_versions(),
        "driver_inventory": driver,
        "selected_driver_record": (
            _resolved_gpu_record(driver, str(config["device_selector"]))
            if config["platform"] == "gpu"
            else None
        ),
        "parent_environment": _environment_snapshot(),
        "exclusivity": {
            "operator_confirmed": True,
            "scope": "operator attestation plus cross-run harness file lock",
        },
    }


def _validate_run_context(context: Mapping[str, Any]) -> None:
    config = context["common_config"]
    if config["platform"] != "gpu":
        return
    record = context.get("selected_driver_record")
    selector = str(config["device_selector"])
    if not isinstance(record, Mapping) or not record.get("uuid"):
        raise RuntimeError(
            "GPU selector did not resolve to an nvidia-smi device record; "
            "the fixed accelerator UUID and driver cannot be proven"
        )
    if selector not in (str(record.get("index")), str(record.get("uuid"))):
        raise RuntimeError("resolved GPU record does not match the requested selector")
    if config["memory_fallback"] == "none":
        git = context.get("git")
        if (
            not isinstance(git, Mapping)
            or git.get("available") is not True
            or not isinstance(git.get("revision"), str)
            or not git["revision"]
        ):
            raise RuntimeError("release GPU measurement requires an available Git revision")
        if git.get("dirty") is not False:
            raise RuntimeError(
                "release GPU measurement requires a clean Git worktree; use an "
                "explicit smoke fallback only for non-release diagnostics"
            )
        committed = _git_source_hashes(str(git["revision"]))
        if committed != context.get("source", {}).get("files"):
            raise RuntimeError(
                "release GPU measurement source hashes do not match the recorded Git revision"
            )


def _run_sweep(args: argparse.Namespace) -> int:
    _validate_execute_args(args)
    output = args.output_dir.expanduser().resolve()
    selector = args.accelerator or "cpu-smoke"
    with _AcceleratorLock(selector) as lock:
        context = _create_context(args)
        _validate_run_context(context)
        output.mkdir(parents=True, exist_ok=True)
        context["exclusivity"]["lock_path"] = str(lock.path)
        context_path = output / "RUN_CONTEXT.json"
        _write_json_atomic(context_path, context)
        context_sha256 = _file_sha256(context_path)
        environment = _worker_environment(
            context["common_config"], context.get("selected_driver_record")
        )
        result_paths = []
        for k in EXACT_K_SET:
            result_path = output / f"K={k}.json"
            completed = subprocess.run(
                _worker_command(context_path, result_path, k),
                cwd=REPO_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(f"K={k} worker failed: {detail}")
            if not result_path.is_file():
                raise RuntimeError(f"K={k} worker returned without {result_path}")
            result_paths.append(result_path)
        manifest = _aggregate_manifest(context, context_path, context_sha256, result_paths)
        _write_json_atomic(output / "MANIFEST.json", manifest)
    print(json.dumps({"status": "complete", "output_dir": str(output)}, sort_keys=True))
    return 0


def _aggregate_manifest(
    context: Mapping[str, Any],
    context_path: Path,
    context_sha256: str,
    result_paths: Sequence[Path],
) -> dict[str, Any]:
    if _file_sha256(context_path) != context_sha256:
        raise RuntimeError("RUN_CONTEXT.json changed while workers were running")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    by_k = {int(result["k"]): result for result in results}
    path_by_k = {
        int(result["k"]): path for result, path in zip(results, result_paths, strict=True)
    }
    if tuple(sorted(by_k)) != EXACT_K_SET or len(results) != len(EXACT_K_SET):
        raise RuntimeError("worker results do not contain exactly K=1,2,3,4")
    base_time = float(by_k[1]["timing"]["median_actor_update_ms"])
    base_absolute = int(by_k[1]["memory"]["absolute_peak_bytes"])
    base_adjusted = int(by_k[1]["memory"]["baseline_adjusted_peak_bytes"])
    memory_sources = {result["memory"]["source"] for result in results}
    if len(memory_sources) != 1:
        raise RuntimeError("worker results disagree on the memory measurement source")
    for result in results:
        if result.get("hashes", {}).get("run_context_sha256") != context_sha256:
            raise RuntimeError("worker result is not bound to RUN_CONTEXT.json")
    evidence_status = (
        "cpu_smoke"
        if context["common_config"]["platform"] == "cpu"
        else "fallback_smoke"
        if context["common_config"]["memory_fallback"] != "none"
        else "measured"
    )

    def ratio(value: float, baseline: float) -> float | None:
        return value / baseline if baseline > 0 else None

    summaries = []
    for k in EXACT_K_SET:
        result = by_k[k]
        timing = float(result["timing"]["median_actor_update_ms"])
        absolute = int(result["memory"]["absolute_peak_bytes"])
        adjusted = int(result["memory"]["baseline_adjusted_peak_bytes"])
        summaries.append(
            {
                "k": k,
                "result_file": f"K={k}.json",
                "result_sha256": _file_sha256(path_by_k[k]),
                "raw_trial_ms": result["timing"]["raw_trial_ms"],
                "median_actor_update_ms": timing,
                "time_ratio_to_k1": ratio(timing, base_time),
                "absolute_peak_bytes": absolute,
                "absolute_peak_ratio_to_k1": ratio(absolute, base_absolute),
                "baseline_adjusted_peak_bytes": adjusted,
                "baseline_adjusted_peak_ratio_to_k1": ratio(adjusted, base_adjusted),
            }
        )
    common_config = dict(context["common_config"])
    dataset_record = dict(context["dataset"])
    dataset_record.update(
        {
            "rows_after_transition_filter": by_k[1]["dataset_batch"][
                "dataset_rows_after_transition_filter"
            ],
            "observation_shape": by_k[1]["dataset_batch"]["observation_shape"],
            "action_shape": by_k[1]["dataset_batch"]["action_shape"],
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "actor_cost_manifest",
        "created_utc": _utc_now(),
        "status": evidence_status,
        "accelerator_evidence": evidence_status == "measured",
        "exact_k_set": list(EXACT_K_SET),
        "only_swept_field": "k",
        "common_config": common_config,
        "invariant_config_sha256": _json_sha256(common_config),
        "run_context": {"file": "RUN_CONTEXT.json", "sha256": context_sha256},
        "hashes": {
            "code_sha256": by_k[1]["hashes"]["code_sha256"],
            "invariant_config_sha256": by_k[1]["hashes"]["invariant_config_sha256"],
            "dataset_sha256": by_k[1]["hashes"]["dataset_sha256"],
            "batch_sha256": by_k[1]["hashes"]["batch_sha256"],
            "normalization_sha256": by_k[1]["hashes"]["normalization_sha256"],
            "run_context_sha256": context_sha256,
        },
        "source": context["source"],
        "dataset": dataset_record,
        "git": context["git"],
        "host": context["host"],
        "python": context["python"],
        "packages": by_k[1]["provenance"]["packages"],
        "backend_platform_version": by_k[1]["provenance"]["backend_platform_version"],
        "device": by_k[1]["provenance"]["device"],
        "device_binding": by_k[1]["provenance"]["device_binding"],
        "worker_environment": by_k[1]["provenance"]["environment"],
        "driver_inventory": context["driver_inventory"],
        "selected_driver_record": context["selected_driver_record"],
        "parent_environment": context["parent_environment"],
        "exclusivity": context["exclusivity"],
        "orchestrator": context["orchestrator"],
        "memory_scope": by_k[1]["memory"]["peak_counter_scope"],
        "results": summaries,
    }


def _current_rss_bytes() -> int:
    statm = Path("/proc/self/statm")
    if statm.is_file():
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return int(peak * scale)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return int(peak * scale)


def _memory_probe(device: Any, fallback: str) -> dict[str, Any]:
    error_detail = None
    try:
        stats = device.memory_stats()
    except Exception as error:  # backend-specific API failures are handled below
        error_detail = f"{type(error).__name__}: {error}"
        stats = None
    if isinstance(stats, Mapping) and {
        "bytes_in_use",
        "peak_bytes_in_use",
    }.issubset(stats):
        return {
            "source": "device.memory_stats",
            "current_bytes": int(stats["bytes_in_use"]),
            "peak_bytes": int(stats["peak_bytes_in_use"]),
            "raw_stats": {
                str(key): int(value)
                if isinstance(value, (int, bool))
                else float(value)
                if isinstance(value, float) and math.isfinite(value)
                else str(value)
                for key, value in stats.items()
            },
        }
    available = sorted(stats) if isinstance(stats, Mapping) else []
    unavailable = error_detail or f"available keys: {available}"
    if fallback != "process-rss":
        raise RuntimeError(
            "device.memory_stats() did not provide bytes_in_use and "
            f"peak_bytes_in_use ({unavailable}); rerun only with "
            "the explicitly labeled --memory-fallback process-rss for a smoke "
            "test that is not accelerator-memory evidence"
        )
    return {
        "source": "process_rss_fallback_not_accelerator_memory",
        "current_bytes": _current_rss_bytes(),
        "peak_bytes": _peak_rss_bytes(),
        "raw_stats": None,
        "device_memory_stats_unavailable": unavailable,
    }


def _hash_array_bundle(arrays: Mapping[str, Any], np_module: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np_module.ascontiguousarray(np_module.asarray(arrays[name]))
        header = {
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
        digest.update(_canonical_bytes(header))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _device_identity(device: Any, selector: str | None) -> dict[str, Any]:
    identity = {
        "selector": selector,
        "platform": str(getattr(device, "platform", "unknown")),
        "device_kind": str(getattr(device, "device_kind", "unknown")),
        "id": int(getattr(device, "id", 0)),
        "local_hardware_id": int(getattr(device, "local_hardware_id", 0)),
        "process_index": int(getattr(device, "process_index", 0)),
        "uuid": None,
        "uuid_source": None,
    }
    for attribute in ("uuid", "device_uuid"):
        value = getattr(device, attribute, None)
        if value:
            identity["uuid"] = str(value)
            identity["uuid_source"] = f"jax.device.{attribute}"
            break
    return identity


def _block_until_ready(tree: Any, jax_module: Any) -> None:
    for leaf in jax_module.tree_util.tree_leaves(tree):
        ready = getattr(leaf, "block_until_ready", None)
        if ready is not None:
            ready()


def _worker_main(argv: Sequence[str]) -> int:
    args = _worker_parser().parse_args(argv)
    if args.k not in EXACT_K_SET:
        raise ValueError(f"worker K must be one of {EXACT_K_SET}")
    context_bytes = args.context.read_bytes()
    context_sha256 = hashlib.sha256(context_bytes).hexdigest()
    context = json.loads(context_bytes)
    config = dict(context["common_config"])
    config["k"] = int(args.k)
    _require_source_snapshot(context["source"], "before worker imports")

    # Heavy imports occur only in an explicitly executed worker, after device
    # visibility has been fixed in the child environment.
    import jax
    import jax.numpy as jnp
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT))
    from train_td3bc import (
        Transition,
        create_train_state,
        qlearning_from_hdf5,
        update_actor_hop,
        update_first_actor,
    )

    _require_source_snapshot(context["source"], "during worker imports")

    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one visible JAX device, found {len(devices)}")
    device = devices[0]
    if config["platform"] == "gpu" and device.platform != "gpu":
        raise RuntimeError(
            f"GPU benchmark selected but JAX resolved platform {device.platform!r}"
        )
    if config["platform"] == "cpu" and device.platform != "cpu":
        raise RuntimeError(
            f"CPU smoke selected but JAX resolved platform {device.platform!r}"
        )
    if device.platform == "cpu" and not config["allow_cpu_smoke"]:
        raise RuntimeError("CPU worker is permitted only as an explicitly labeled smoke run")

    phases: dict[str, Any] = {}
    phases["runtime_baseline"] = _memory_probe(device, config["memory_fallback"])
    dataset_path = Path(config["dataset_path"])
    if _file_sha256(dataset_path) != context["dataset"]["sha256"]:
        raise RuntimeError("dataset hash changed after the run context was created")
    raw = qlearning_from_hdf5(dataset_path)
    observation_shape = list(raw["observations"].shape)
    action_shape = list(raw["actions"].shape)
    epsilon = float(config["normalization_epsilon"])
    mean = np.asarray(raw["observations"], dtype=np.float32).mean(axis=0)
    std = np.asarray(raw["observations"], dtype=np.float32).std(axis=0) + epsilon
    raw["observations"] = (raw["observations"] - mean) / std
    raw["next_observations"] = (raw["next_observations"] - mean) / std
    sample_rng = np.random.Generator(np.random.PCG64(int(config["sample_seed"])))
    indices = sample_rng.integers(
        0,
        raw["observations"].shape[0],
        size=int(config["batch_size"]),
        endpoint=False,
        dtype=np.int64,
    )
    batch_np = {name: np.asarray(values[indices]) for name, values in raw.items()}
    batch_hash = _hash_array_bundle({**batch_np, "indices": indices}, np)
    normalization_hash = _hash_array_bundle({"mean": mean, "std": std}, np)
    batch = Transition(**{name: jnp.asarray(value) for name, value in batch_np.items()})
    max_action = float(np.max(np.abs(np.asarray(raw["actions"]))))
    max_action = 1.0 if max_action <= 1.0 + 1e-5 else max_action
    state = create_train_state(
        jax.random.PRNGKey(int(config["init_seed"])),
        batch.observations[:1],
        batch.actions[:1],
        max_action=max_action,
        lr=float(config["learning_rate"]),
        policy_noise=0.2 * max_action,
        noise_clip=0.5 * max_action,
        mpi_steps=int(args.k),
    )
    tau_step = float(config["tau"]) / float(args.k)

    def actor_only(actor_state: Any, fixed_batch: Any) -> tuple[Any, tuple[Any, ...]]:
        actor_state, first_loss = update_first_actor(
            actor_state,
            fixed_batch,
            tau_step,
            float(config["polyak"]),
            bool(config["scale_norm"]),
        )
        losses = [first_loss]
        for actor_index in range(1, args.k):
            actor_state, loss = update_actor_hop(
                actor_state,
                fixed_batch,
                actor_index,
                tau_step,
                bool(config["scale_norm"]),
            )
            losses.append(loss)
        return actor_state, tuple(losses)

    phases["before_compile"] = _memory_probe(device, config["memory_fallback"])
    jitted = jax.jit(actor_only)
    lower_started = time.perf_counter_ns()
    lowered = jitted.lower(state, batch)
    lowering_ms = (time.perf_counter_ns() - lower_started) / 1e6
    compile_started = time.perf_counter_ns()
    compiled = lowered.compile()
    compile_ms = (time.perf_counter_ns() - compile_started) / 1e6
    phases["after_compile"] = _memory_probe(device, config["memory_fallback"])

    warmup_ms = []
    for _ in range(int(config["warmup_calls"])):
        started = time.perf_counter_ns()
        warm_result = compiled(state, batch)
        _block_until_ready(warm_result, jax)
        warmup_ms.append((time.perf_counter_ns() - started) / 1e6)
        del warm_result
    gc.collect()
    phases["after_warmup"] = _memory_probe(device, config["memory_fallback"])

    raw_trial_ms = []
    raw_call_ms = []
    for _ in range(int(config["trials"])):
        calls = []
        for _ in range(int(config["calls_per_trial"])):
            started = time.perf_counter_ns()
            trial_result = compiled(state, batch)
            _block_until_ready(trial_result, jax)
            calls.append((time.perf_counter_ns() - started) / 1e6)
            del trial_result
        raw_call_ms.append(calls)
        raw_trial_ms.append(statistics.fmean(calls))
    gc.collect()
    phases["after_trials"] = _memory_probe(device, config["memory_fallback"])

    sources = {phase["source"] for phase in phases.values()}
    if len(sources) != 1:
        raise RuntimeError(f"memory source changed during worker: {sorted(sources)}")
    absolute_peak = max(int(phase["peak_bytes"]) for phase in phases.values())
    baseline_current = int(phases["runtime_baseline"]["current_bytes"])
    adjusted_peak = max(0, absolute_peak - baseline_current)
    device_identity = _device_identity(device, config["device_selector"])
    selected_record = context.get("selected_driver_record")
    if config["platform"] == "gpu":
        if not isinstance(selected_record, Mapping) or not selected_record.get("uuid"):
            raise RuntimeError("GPU worker lacks a resolved driver UUID record")
        resolved_uuid = str(selected_record["uuid"])
        environment = _environment_snapshot()
        if environment.get("CUDA_VISIBLE_DEVICES") != resolved_uuid:
            raise RuntimeError("GPU worker CUDA visibility is not bound to the resolved UUID")
        if environment.get("JAX_PLATFORMS") != "gpu":
            raise RuntimeError("GPU worker JAX platform selection is not fail-closed")
        if environment.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
            raise RuntimeError("GPU worker must disable JAX preallocation")
        if device_identity["uuid"] is not None and device_identity["uuid"] != resolved_uuid:
            raise RuntimeError("JAX-observed device UUID differs from the resolved GPU UUID")
        device_binding = {
            "method": "resolved_uuid_visibility",
            "resolved_uuid": resolved_uuid,
            "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
            "jax_device_uuid": device_identity["uuid"],
            "jax_device_uuid_source": device_identity["uuid_source"],
        }
    else:
        device_binding = {
            "method": "cpu_smoke_visibility",
            "resolved_uuid": None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jax_device_uuid": device_identity["uuid"],
            "jax_device_uuid_source": device_identity["uuid_source"],
        }
    if _file_sha256(dataset_path) != context["dataset"]["sha256"]:
        raise RuntimeError("dataset changed while the worker was reading or measuring it")
    _require_source_snapshot(context["source"], "while the worker was running")
    backend_version = str(getattr(getattr(device, "client", None), "platform_version", "unknown"))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "actor_cost_worker_result",
        "k": int(args.k),
        "config": config,
        "hashes": {
            "config_sha256": _json_sha256(config),
            "invariant_config_sha256": _json_sha256(context["common_config"]),
            "code_sha256": context["source"]["sha256"],
            "source_files": context["source"]["files"],
            "dataset_sha256": context["dataset"]["sha256"],
            "batch_sha256": batch_hash,
            "normalization_sha256": normalization_hash,
            "run_context_sha256": context_sha256,
        },
        "dataset_batch": {
            "dataset_path": str(dataset_path),
            "dataset_rows_after_transition_filter": int(raw["observations"].shape[0]),
            "observation_shape": observation_shape,
            "action_shape": action_shape,
            "batch_size": int(config["batch_size"]),
            "sample_rng": {"algorithm": "numpy.PCG64", "seed": int(config["sample_seed"])},
            "indices_sha256": _hash_array_bundle({"indices": indices}, np),
            "normalization": "full filtered dataset mean/std plus epsilon",
        },
        "actor_update": {
            "scope": "actor_only_no_critic_update",
            "production_functions": config["production_actor_functions"],
            "tau_step": tau_step,
            "fixed_input_state_each_call": True,
            "synchronization": "every call blocks every JAX array leaf",
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "lowering_ms": lowering_ms,
            "compile_ms": compile_ms,
            "warmup_ms": warmup_ms,
            "raw_call_ms": raw_call_ms,
            "raw_trial_ms": raw_trial_ms,
            "trial_statistic": "arithmetic mean of synchronized calls in that trial",
            "median_actor_update_ms": statistics.median(raw_trial_ms),
        },
        "memory": {
            "source": next(iter(sources)),
            "fallback_selected": config["memory_fallback"],
            "is_accelerator_memory": next(iter(sources)) == "device.memory_stats",
            "phases": phases,
            "phase_order": [
                "runtime_baseline",
                "before_compile",
                "after_compile",
                "after_warmup",
                "after_trials",
            ],
            "peak_counter_scope": (
                "fresh worker process-lifetime backend high-water mark through dataset/batch "
                "transfer, actor/critic state initialization, lowering/compile, warmup, and "
                "timed trials; not an isolated steady-state actor-call peak"
            ),
            "baseline_phase": "runtime_baseline",
            "absolute_peak_bytes": absolute_peak,
            "baseline_current_bytes": baseline_current,
            "baseline_adjusted_peak_bytes": adjusted_peak,
            "baseline_adjusted_formula": "max(0, absolute_peak_bytes - baseline_current_bytes)",
            "before_compile_resident_bytes": int(phases["before_compile"]["current_bytes"]),
            "after_warmup_resident_bytes": int(phases["after_warmup"]["current_bytes"]),
            "after_trials_resident_bytes": int(phases["after_trials"]["current_bytes"]),
        },
        "process": {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "process_token": str(uuid.uuid4()),
            "started_utc": _utc_now(),
        },
        "provenance": {
            "git": context["git"],
            "python": context["python"],
            "packages": _package_versions(),
            "backend_platform_version": backend_version,
            "environment": _environment_snapshot(),
            "device": device_identity,
            "device_binding": device_binding,
            "visible_device_count": len(devices),
            "selected_driver_record": context["selected_driver_record"],
        },
    }
    _write_json_atomic(args.result, payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_worker":
        return _worker_main(arguments[1:])
    args = _public_parser().parse_args(arguments)
    if not args.execute:
        print(json.dumps(_plan_payload(args), indent=2, sort_keys=True))
        return 0
    return _run_sweep(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
