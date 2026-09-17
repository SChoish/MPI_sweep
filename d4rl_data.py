"""D4RL dataset names and dependency-free download helper."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

D4RL_V2_BASE = "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2"

DATASET_FILES = {
    "halfcheetah-medium-v2": "halfcheetah_medium-v2.hdf5",
    "halfcheetah-medium-replay-v2": "halfcheetah_medium_replay-v2.hdf5",
    "halfcheetah-expert-v2": "halfcheetah_expert-v2.hdf5",
    "hopper-medium-v2": "hopper_medium-v2.hdf5",
    "hopper-medium-replay-v2": "hopper_medium_replay-v2.hdf5",
    "hopper-expert-v2": "hopper_expert-v2.hdf5",
    "walker2d-medium-v2": "walker2d_medium-v2.hdf5",
    "walker2d-medium-replay-v2": "walker2d_medium_replay-v2.hdf5",
    "walker2d-expert-v2": "walker2d_expert-v2.hdf5",
    # AntMaze (sparse). Local name matches ~/.d4rl/datasets convention.
    "antmaze-umaze-diverse-v2": "antmaze-umaze-diverse-v2.hdf5",
}

# Non-mujoco_v2 URLs (filename on the server may differ from local alias).
DATASET_URLS = {
    "antmaze-umaze-diverse-v2": (
        "http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/"
        "Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5"
    ),
}

D4RL_CACHE_DIRS = (
    Path.home() / ".d4rl" / "datasets",
    Path(os.environ["D4RL_DATASET_DIR"]) / "datasets"
    if os.environ.get("D4RL_DATASET_DIR")
    else None,
)


def dataset_path(env_name: str, data_dir: Path) -> Path:
    try:
        filename = DATASET_FILES[env_name]
    except KeyError as error:
        raise ValueError(f"unsupported D4RL environment: {env_name}") from error
    return data_dir / filename


def _existing_cache_copy(env_name: str) -> Path | None:
    filename = DATASET_FILES[env_name]
    for cache in D4RL_CACHE_DIRS:
        if cache is None:
            continue
        candidate = cache / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def download_dataset(env_name: str, data_dir: Path) -> Path:
    """Return a local dataset path, downloading atomically when absent."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_path(env_name, data_dir)
    if path.is_file() and path.stat().st_size > 0:
        return path
    cached = _existing_cache_copy(env_name)
    if cached is not None:
        try:
            path.symlink_to(cached)
            print(f"[data] linked {path} -> {cached}", flush=True)
            return path
        except OSError:
            import shutil

            print(f"[data] copying {cached} -> {path}", flush=True)
            shutil.copy2(cached, path)
            return path
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    url = DATASET_URLS.get(env_name, f"{D4RL_V2_BASE}/{path.name}")
    print(f"[data] downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, temporary_path)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path
