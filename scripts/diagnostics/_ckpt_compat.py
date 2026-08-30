"""Normalize release checkpoints to the lab-style keys dump scripts expect.

Release ``train_td3bc.py`` stores ``actors_params`` (tuple of hop actors).
Lab dumps look up ``actor_params`` / ``actor2_params`` / ….
"""

from __future__ import annotations

from typing import Any, Mapping


_ACTOR_KEYS = (
    "actor_params",
    "actor2_params",
    "actor3_params",
    "actor4_params",
    "actor5_params",
)


def normalize_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if "actor_params" not in out and "actors_params" in out:
        actors = out["actors_params"]
        for key, params in zip(_ACTOR_KEYS, actors, strict=False):
            out[key] = params
    # Release / lab both use critic_params; keep a defensive alias.
    if "critic_params" not in out and "critic_params" in out:
        out["critic_params"] = out["critic_params"]
    return out
