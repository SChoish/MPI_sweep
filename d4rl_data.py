"""D4RL locomotion dataset names and dependency-free download helper."""

from __future__ import annotations

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
}


def dataset_path(env_name: str, data_dir: Path) -> Path:
    try:
        filename = DATASET_FILES[env_name]
    except KeyError as error:
        raise ValueError(f"unsupported D4RL environment: {env_name}") from error
    return data_dir / filename


def download_dataset(env_name: str, data_dir: Path) -> Path:
    """Return a local dataset path, downloading atomically when absent."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_path(env_name, data_dir)
    if path.is_file() and path.stat().st_size > 0:
        return path
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    url = f"{D4RL_V2_BASE}/{path.name}"
    print(f"[data] downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, temporary_path)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path
