"""Prefer LAB_ROOT over this repo when importing train_td3bc.

Lab checkpoints use actor_params / actor2_params; the release train_td3bc
uses actors_params. Host dumps must import the tree that wrote the ckpts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_train_import_path() -> Path:
    lab = os.environ.get("LAB_ROOT", "").strip()
    prefer = Path(lab).expanduser().resolve() if lab else REPO_ROOT
    prefer_s = str(prefer)
    if prefer_s not in sys.path:
        sys.path.insert(0, prefer_s)
    return prefer
