"""Best-effort provenance capture for MPI_sweep training and orchestration.

Provenance retention is forward-only: it records what produced *new* runs and
never blocks or crashes training. Every collector degrades to a recorded error
string rather than raising, so a missing tool (git, nvidia-smi) or an
unimportable package only annotates the failure.

See ``scripts/PROVENANCE.md`` for the on-disk contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Packages worth pinning for reproducibility of a JAX/Flax offline-RL run.
_KEY_PACKAGES = ("jax", "jaxlib", "flax", "optax", "numpy")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str | None:
    """Return the hex SHA-256 of a file, or ``None`` if it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_revision(root: str | Path) -> dict[str, Any]:
    """Best-effort Git revision + dirty flag. Records failure clearly."""
    root = str(root)

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", root, *args],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as error:  # git missing / timeout / OS error
            return f"__error__:{type(error).__name__}: {error}"
        if out.returncode != 0:
            return f"__error__:git {' '.join(args)} rc={out.returncode}"
        return out.stdout.strip()

    revision = _run(["rev-parse", "HEAD"])
    status = _run(["status", "--porcelain"])
    info: dict[str, Any] = {}
    if isinstance(revision, str) and revision.startswith("__error__:"):
        info["revision"] = None
        info["error"] = revision[len("__error__:") :]
    else:
        info["revision"] = revision
    if isinstance(status, str) and not status.startswith("__error__:"):
        info["dirty"] = bool(status)
    else:
        info["dirty"] = None
        info.setdefault("error", "status unavailable")
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    info["branch"] = None if (branch or "").startswith("__error__:") else branch
    return info


def package_versions() -> dict[str, str | None]:
    """Return versions of key packages that are importable, else ``None``."""
    from importlib import metadata

    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in _KEY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def gpu_info() -> dict[str, Any]:
    """Best-effort accelerator inventory from env + nvidia-smi."""
    info: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": None,
    }
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as error:
        info["error"] = f"{type(error).__name__}: {error}"
        return info
    if out.returncode != 0:
        info["error"] = f"nvidia-smi rc={out.returncode}"
        return info
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": parts[0], "uuid": parts[1], "name": parts[2]})
    info["gpus"] = gpus
    return info


def file_digests(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    """Return ``{label: {path, exists, size, sha256}}`` for source files."""
    result: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        p = Path(path)
        entry: dict[str, Any] = {"path": str(p), "exists": p.is_file()}
        if entry["exists"]:
            try:
                entry["size"] = p.stat().st_size
            except OSError:
                entry["size"] = None
            entry["sha256"] = sha256_file(p)
        result[label] = entry
    return result


def dataset_identity(path: str | Path) -> dict[str, Any]:
    """Cheap dataset identity: path + size + mtime (no multi-GB rehash).

    Full SHA-256 of the HDF5 file is intentionally skipped every run; hashing
    multi-GB datasets on each launch is not worth the cost and no cached hash
    helper exists here. Size + mtime pin the local file adequately.
    """
    p = Path(path)
    entry: dict[str, Any] = {
        "path": str(p),
        "exists": p.is_file(),
        "sha256": None,
        "hash_skipped": "multi-GB hdf5; recorded path+size+mtime instead",
    }
    if entry["exists"]:
        stat = p.stat()
        entry["size"] = stat.st_size
        entry["mtime"] = datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat()
    return entry


def base_provenance(
    root: str | Path,
    source_files: dict[str, str | Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the static provenance block available before data load."""
    provenance: dict[str, Any] = {
        "schema": "mpi_sweep.provenance/1",
        "captured_at": _utc_now(),
        "git": git_revision(root),
        "source_files": file_digests(source_files),
        "packages": package_versions(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "accelerator": gpu_info(),
    }
    if extra:
        provenance.update(extra)
    return provenance


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Atomically write a JSON provenance file. Never raises on best-effort use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def update_json(path: str | Path, updates: dict[str, Any]) -> None:
    """Merge ``updates`` into an existing provenance JSON (best-effort)."""
    path = Path(path)
    current: dict[str, Any] = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(updates)
    current["updated_at"] = _utc_now()
    write_json(path, current)
