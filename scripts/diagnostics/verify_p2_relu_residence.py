#!/usr/bin/env python3
"""Independent full-recomputation verifier for the P2 ReLU residence pilot.

This module intentionally does not import ``run_p2_relu_residence``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from decimal import Decimal, localcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROTOCOL = "p2_relu_residence_v1"
ENVIRONMENT = "halfcheetah-medium-v2"
SEED = 0
TIMES = (0.025, 0.05, 0.1, 0.2)
SUBSTEPS = (1, 2, 4, 8, 16)
N_PILOT = 64
PILOT_SAMPLE_SEED = 20260902
ACTION_ELIGIBILITY = 0.95
BOX = 1.0
ROUNDING_SAFETY = 32.0
TIME_ATOL = 1e-12
TIME_RTOL = 1e-9
FD_DELTAS = tuple(2.0**-power for power in range(10, 31))
FIRST_NONE = 0
FIRST_RELU = 1
FIRST_BOX = 2
FIRST_TIE = 3
PINNED_INPUTS = {
    "checkpoint": {
        "semantic_id": "halfcheetah-medium-v2_tau1_seed0_step1000000_qlearning_checkpoint",
        "basename": "params_1000000.pkl",
        "sha256": "cafaabea01bfb25af09e6b21f64d7953cc7aa76c6d4e486cd9075db239445ab1",
    },
    "config": {
        "semantic_id": "halfcheetah-medium-v2_tau1_seed0_training_config",
        "basename": "config.json",
        "sha256": "de316e5ab27bc2c7b09ef5ec12a93bcdfe9ccbffedf9bbeaf1a6ec46c8c923c9",
    },
    "dataset": {
        "semantic_id": "d4rl_halfcheetah_medium_v2_hdf5",
        "basename": "halfcheetah_medium-v2.hdf5",
        "sha256": "41324e8487a9556d8cfa4538827f8937f4d63dae3030825ebc9f3a726ed08ca0",
    },
}
PINNED_EXCLUSIONS = (
    {
        "semantic_id": "archived_fixed_operator_halfcheetah_medium_v2_seed0_indices",
        "basename": "halfcheetah-medium-v2_seed0.npy",
        "sha256": "ddc575b77e76797d7b021e8f122cb1345747c277b66f8a64e66ac334b1d397d0",
        "n_indices": 512,
    },
    {
        "semantic_id": "archived_fixed_operator_halfcheetah_medium_v2_seed1_indices",
        "basename": "halfcheetah-medium-v2_seed1.npy",
        "sha256": "8e80d56920b1d4c1360ac408863024677a0027457361c4dacd6b3485f16414ae",
        "n_indices": 512,
    },
)
PINNED_EXCLUSION_UNION = 1022
SOURCE_SNAPSHOT_DIR = "SOURCE_SNAPSHOT"
AFFINE_RESIDUAL_ULPS = 4096.0
CATEGORIES = (
    "resident",
    "relu_cross",
    "box_cross",
    "tie_cross",
    "boundary_ambiguous",
    "boundary_touch",
    "nonfinite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="recompute read-only and require the existing VERIFY.json to agree",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return sha256_bytes(array.view(np.uint8).tobytes())


def write_json_create(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_artifact_inventory(
    out_dir: Path, artifacts: Any, expected_paths: set[str]
) -> None:
    if not isinstance(artifacts, dict):
        raise AssertionError("manifest artifact inventory missing")
    if set(artifacts) != expected_paths:
        raise AssertionError("manifest artifact inventory mismatch")
    for name, digest in artifacts.items():
        if sha256_file(out_dir / name) != digest:
            raise AssertionError(f"artifact hash mismatch: {name}")


def verify_git_provenance(provenance: Any) -> None:
    if not isinstance(provenance, dict) or not isinstance(
        provenance.get("git_dirty"), bool
    ):
        raise AssertionError("git dirty provenance missing")
    if len(str(provenance.get("git_revision", ""))) != 40:
        raise AssertionError("git revision provenance missing")
    if provenance.get("source_snapshot_is_durable_provenance") is not True:
        raise AssertionError("durable source provenance declaration missing")


def verify_stored_report(verify_path: Path, recomputed: dict[str, Any]) -> None:
    stored = json.loads(verify_path.read_text())
    if not isinstance(stored.get("written_at"), str):
        raise AssertionError("stored VERIFY timestamp missing")
    stored_comparable = dict(stored)
    recomputed_comparable = dict(recomputed)
    stored_comparable.pop("written_at")
    recomputed_comparable.pop("written_at")
    if canonical_bytes(stored_comparable) != canonical_bytes(
        recomputed_comparable
    ):
        raise AssertionError("stored VERIFY does not match read-only recomputation")


def gamma_n(n_terms: int) -> float:
    eps = np.finfo(np.float64).eps
    product = float(n_terms) * eps
    return product / (1.0 - product)


def affine_tol(abs_sum: np.ndarray, n_terms: int) -> np.ndarray:
    return ROUNDING_SAFETY * gamma_n(n_terms) * np.maximum(1.0, abs_sum)


def time_tol(value: float) -> float:
    return max(TIME_ATOL, TIME_RTOL * max(1.0, abs(float(value))))


def load_rows(dataset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(dataset, "r") as handle:
        observations = np.array(handle["observations"], dtype=np.float32)
        actions = np.array(handle["actions"], dtype=np.float32)
        timeouts = np.array(handle["timeouts"], dtype=np.float32)
    keep = ~timeouts[:-1].astype(bool)
    observations = observations[:-1][keep]
    actions = actions[:-1][keep]
    mean = observations.mean(axis=0)
    std = observations.std(axis=0) + np.float32(1e-3)
    return observations, actions, mean, std


def extract_q1(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    tree = payload["critic_params"]
    if isinstance(tree, dict) and set(tree) == {"params"}:
        tree = tree["params"]
    return {
        "w1": np.asarray(tree["q1_l0"]["kernel"], dtype=np.float64),
        "b1": np.asarray(tree["q1_l0"]["bias"], dtype=np.float64),
        "w2": np.asarray(tree["q1_l1"]["kernel"], dtype=np.float64),
        "b2": np.asarray(tree["q1_l1"]["bias"], dtype=np.float64),
        "w3": np.asarray(tree["q1_out"]["kernel"], dtype=np.float64),
        "b3": np.asarray(tree["q1_out"]["bias"], dtype=np.float64),
    }


def direct_forward(
    params: dict[str, np.ndarray], states: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.concatenate((states, actions), axis=-1)
    z1 = np.einsum("ni,ij->nj", inputs, params["w1"]) + params["b1"]
    h1 = np.where(z1 > 0.0, z1, 0.0)
    z2 = np.einsum("ni,ij->nj", h1, params["w2"]) + params["b2"]
    h2 = np.where(z2 > 0.0, z2, 0.0)
    q = np.einsum("ni,ij->nj", h2, params["w3"]) + params["b3"]
    return q[:, 0], z1, z2


def independent_geometry(
    params: dict[str, np.ndarray], states: np.ndarray, actions: np.ndarray, c_ref: float
) -> dict[str, np.ndarray]:
    q, z1, z2 = direct_forward(params, states, actions)
    inputs = np.concatenate((states, actions), axis=-1)
    active1 = z1 > 0.0
    active2 = z2 > 0.0
    output_weight = params["w3"][:, 0]
    upstream2 = active2 * output_weight
    upstream1 = np.einsum("nj,ij->ni", upstream2, params["w2"]) * active1
    state_dim = states.shape[-1]
    action_kernel = params["w1"][state_dim:, :]
    grad = np.einsum("nh,dh->nd", upstream1, action_kernel)
    velocity = actions.shape[-1] * grad / c_ref
    dz1 = np.einsum("nd,dh->nh", velocity, action_kernel)
    dz2 = np.einsum("nh,hj->nj", dz1 * active1, params["w2"])
    z1_scale = np.einsum(
        "ni,ij->nj", np.abs(inputs), np.abs(params["w1"])
    ) + np.abs(params["b1"])
    h1 = np.where(z1 > 0.0, z1, 0.0)
    z2_scale = np.einsum(
        "ni,ij->nj", np.abs(h1), np.abs(params["w2"])
    ) + np.abs(params["b2"])
    tol1 = affine_tol(z1_scale, inputs.shape[-1] + 1)
    tol2_local = affine_tol(z2_scale, h1.shape[-1] + 1)
    propagated = np.einsum("nh,hj->nj", tol1, np.abs(params["w2"]))
    tol2 = tol2_local + (1.0 + ROUNDING_SAFETY * gamma_n(h1.shape[-1] + 1)) * propagated

    n = len(actions)
    relu = np.full(n, np.inf)
    box = np.full(n, np.inf)
    first = np.full(n, np.inf)
    code = np.zeros(n, dtype=np.int8)
    layer_out = np.full(n, -1, dtype=np.int16)
    unit_out = np.full(n, -1, dtype=np.int16)
    ambiguous = np.zeros(n, dtype=bool)
    nonfinite = np.zeros(n, dtype=bool)
    scaled_margin = np.full(n, np.inf)
    for row in range(n):
        values_to_check = (q[row], z1[row], z2[row], grad[row], velocity[row])
        if not all(np.all(np.isfinite(value)) for value in values_to_check):
            nonfinite[row] = True
            continue
        scaled_margin[row] = min(
            float(np.min(np.abs(z1[row]) / tol1[row])),
            float(np.min(np.abs(z2[row]) / tol2[row])),
        )
        if np.any(np.abs(z1[row]) <= tol1[row]) or np.any(
            np.abs(z2[row]) <= tol2[row]
        ):
            ambiguous[row] = True
        relu_candidates: list[tuple[float, int, int]] = []
        for layer, values, slopes, tolerances in (
            (1, z1[row], dz1[row], tol1[row]),
            (2, z2[row], dz2[row], tol2[row]),
        ):
            for unit in range(len(values)):
                value = float(values[unit])
                slope = float(slopes[unit])
                if abs(value) <= float(tolerances[unit]) or slope == 0.0:
                    continue
                if (value > 0.0 > slope) or (value < 0.0 < slope):
                    candidate = -value / slope
                    if candidate > 0.0 and math.isfinite(candidate):
                        relu_candidates.append((candidate, layer, unit))
        relu_candidates.sort()
        if relu_candidates:
            relu[row] = relu_candidates[0][0]
        box_candidates = []
        for action, direction in zip(actions[row], velocity[row], strict=True):
            if direction > 0:
                candidate = (BOX - action) / direction
            elif direction < 0:
                candidate = (-BOX - action) / direction
            else:
                continue
            if candidate > 0 and math.isfinite(candidate):
                box_candidates.append(float(candidate))
        if box_candidates:
            box[row] = min(box_candidates)
        first[row] = min(relu[row], box[row])
        if not math.isfinite(first[row]):
            continue
        tolerance = time_tol(first[row])
        near_relu = [
            item for item in relu_candidates if abs(item[0] - first[row]) <= tolerance
        ]
        near_box = [
            item for item in box_candidates if abs(item - first[row]) <= tolerance
        ]
        if len(near_relu) + len(near_box) > 1:
            code[row] = FIRST_TIE
        elif near_relu:
            code[row] = FIRST_RELU
        else:
            code[row] = FIRST_BOX
        if near_relu:
            layer_out[row], unit_out[row] = near_relu[0][1], near_relu[0][2]
    return {
        "q_anchor": q,
        "grad": grad,
        "velocity": velocity,
        "relu_hit_time": relu,
        "box_hit_time": box,
        "first_hit_time": first,
        "first_hit_code": code,
        "first_layer": layer_out,
        "first_unit": unit_out,
        "boundary_ambiguous": ambiguous,
        "nonfinite": nonfinite,
        "anchor_min_scaled_margin": scaled_margin,
    }


def classify(geometry: dict[str, np.ndarray], row: int, horizon: float) -> str:
    if geometry["nonfinite"][row]:
        return "nonfinite"
    if geometry["boundary_ambiguous"][row]:
        return "boundary_ambiguous"
    hit = float(geometry["first_hit_time"][row])
    if not math.isfinite(hit) or horizon < hit - time_tol(hit):
        return "resident"
    if abs(horizon - hit) <= time_tol(hit):
        return "boundary_touch"
    return {
        FIRST_RELU: "relu_cross",
        FIRST_BOX: "box_cross",
        FIRST_TIE: "tie_cross",
    }[int(geometry["first_hit_code"][row])]


def counts(geometry: dict[str, np.ndarray], horizon: float) -> dict[str, int]:
    result = {name: 0 for name in CATEGORIES}
    for row in range(len(geometry["first_hit_time"])):
        result[classify(geometry, row, horizon)] += 1
    return result


def pattern(
    params: dict[str, np.ndarray], state: np.ndarray, action: np.ndarray
) -> tuple[bytes, bytes]:
    _, z1, z2 = direct_forward(params, state[None, :], action[None, :])
    return np.packbits(z1[0] > 0.0).tobytes(), np.packbits(z2[0] > 0.0).tobytes()


def finite_difference_checks(
    params: dict[str, np.ndarray],
    states: np.ndarray,
    actions: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> tuple[int, float]:
    checked = 0
    max_error = 0.0
    eps = np.finfo(np.float64).eps
    for row in range(len(actions)):
        if geometry["nonfinite"][row] or geometry["boundary_ambiguous"][row]:
            continue
        base_pattern = pattern(params, states[row], actions[row])
        for coordinate in range(actions.shape[-1]):
            found = False
            for delta in FD_DELTAS:
                plus = actions[row].copy()
                minus = actions[row].copy()
                plus[coordinate] += delta
                minus[coordinate] -= delta
                if np.max(np.abs(plus)) >= BOX or np.max(np.abs(minus)) >= BOX:
                    continue
                if pattern(params, states[row], plus) != base_pattern:
                    continue
                if pattern(params, states[row], minus) != base_pattern:
                    continue
                q_plus = direct_forward(
                    params, states[row : row + 1], plus[None, :]
                )[0][0]
                q_minus = direct_forward(
                    params, states[row : row + 1], minus[None, :]
                )[0][0]
                estimate = (q_plus - q_minus) / (2.0 * delta)
                target = float(geometry["grad"][row, coordinate])
                error = abs(float(estimate) - target)
                tolerance = max(
                    1e-7,
                    256.0
                    * eps
                    * max(1.0, abs(float(q_plus)), abs(float(q_minus)))
                    / delta,
                )
                if error > tolerance:
                    raise AssertionError(
                        f"finite difference mismatch row={row} dim={coordinate}: "
                        f"error={error} tolerance={tolerance}"
                    )
                max_error = max(max_error, error)
                checked += 1
                found = True
                break
            if not found:
                raise AssertionError(
                    f"no within-pattern finite difference row={row} dim={coordinate}"
                )
    return checked, max_error


def exit_bracket_checks(
    params: dict[str, np.ndarray],
    states: np.ndarray,
    actions: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> tuple[int, float]:
    checked = 0
    max_error = 0.0
    max_time = max(TIMES)
    for row in range(len(actions)):
        if geometry["nonfinite"][row] or geometry["boundary_ambiguous"][row]:
            continue
        relu_time = float(geometry["relu_hit_time"][row])
        base = pattern(params, states[row], actions[row])
        velocity = geometry["velocity"][row]
        if math.isfinite(relu_time) and relu_time <= max_time:
            high = relu_time + max(1e-10, 1e-5 * max(1.0, relu_time))
            high_pattern = pattern(
                params, states[row], actions[row] + high * velocity
            )
            if high_pattern == base:
                raise AssertionError(f"reported exit does not change pattern at row={row}")
            low = 0.0
            for _ in range(64):
                middle = 0.5 * (low + high)
                if pattern(
                    params, states[row], actions[row] + middle * velocity
                ) == base:
                    low = middle
                else:
                    high = middle
            estimate = 0.5 * (low + high)
            error = abs(estimate - relu_time)
            tolerance = max(2e-10, 5e-9 * max(1.0, relu_time))
            if error > tolerance:
                raise AssertionError(
                    f"exit-time mismatch row={row}: error={error} tolerance={tolerance}"
                )
            checked += 1
            max_error = max(max_error, error)
        elif not math.isfinite(relu_time) or relu_time > max_time:
            if pattern(
                params, states[row], actions[row] + max_time * velocity
            ) != base:
                raise AssertionError(f"unreported ReLU exit before Tmax at row={row}")
    return checked, max_error


def pattern_arrays(
    params: dict[str, np.ndarray], state: np.ndarray, action: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _, z1, z2 = direct_forward(params, state[None, :], action[None, :])
    return z1[0] > 0.0, z2[0] > 0.0


def box_and_combined_event_checks(
    params: dict[str, np.ndarray],
    states: np.ndarray,
    actions: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Directly witness finite box hits and earliest-event/tie ordering."""
    box_checked = 0
    combined_checked = 0
    ties_checked = 0
    max_box_error = 0.0
    for row in range(len(actions)):
        if geometry["nonfinite"][row] or geometry["boundary_ambiguous"][row]:
            continue
        action = actions[row]
        velocity = geometry["velocity"][row]
        box_candidates: list[tuple[float, int]] = []
        for coordinate, (anchor, direction) in enumerate(
            zip(action, velocity, strict=True)
        ):
            if direction > 0.0:
                candidate = (BOX - float(anchor)) / float(direction)
            elif direction < 0.0:
                candidate = (-BOX - float(anchor)) / float(direction)
            else:
                continue
            if candidate > 0.0 and math.isfinite(candidate):
                box_candidates.append((candidate, coordinate))
        direct_box = min((item[0] for item in box_candidates), default=math.inf)
        reported_box = float(geometry["box_hit_time"][row])
        if math.isfinite(direct_box):
            if not math.isclose(
                reported_box, direct_box, rel_tol=5e-12, abs_tol=1e-12
            ):
                raise AssertionError(f"box event formula mismatch row={row}")
            delta = max(1e-10, 256.0 * time_tol(direct_box))
            low = max(0.0, direct_box - delta)
            high = direct_box + delta
            low_action = action + low * velocity
            high_action = action + high * velocity
            low_radius = ROUNDING_SAFETY * gamma_n(2) * np.maximum(
                1.0, np.abs(action) + np.abs(low * velocity)
            )
            high_radius = ROUNDING_SAFETY * gamma_n(2) * np.maximum(
                1.0, np.abs(action) + np.abs(high * velocity)
            )
            if np.any(np.abs(low_action) + low_radius > BOX):
                raise AssertionError(f"box lower bracket is not certified inside row={row}")
            if not np.any(np.abs(high_action) - high_radius > BOX):
                raise AssertionError(f"box upper bracket is not certified outside row={row}")
            for _ in range(72):
                middle = 0.5 * (low + high)
                if np.max(np.abs(action + middle * velocity)) <= BOX:
                    low = middle
                else:
                    high = middle
            estimate = 0.5 * (low + high)
            error = abs(estimate - direct_box)
            tolerance = max(2e-12, 5e-10 * max(1.0, direct_box))
            if error > tolerance:
                raise AssertionError(f"direct box bracket mismatch row={row}")
            max_box_error = max(max_box_error, error)
            box_checked += 1
        elif math.isfinite(reported_box):
            raise AssertionError(f"spurious finite box event row={row}")

        relu_time = float(geometry["relu_hit_time"][row])
        direct_first = min(relu_time, direct_box)
        reported_first = float(geometry["first_hit_time"][row])
        if math.isfinite(direct_first):
            if not math.isclose(
                reported_first, direct_first, rel_tol=5e-12, abs_tol=1e-12
            ):
                raise AssertionError(f"combined first time mismatch row={row}")
            tolerance = time_tol(direct_first)
            relu_is_first = math.isfinite(relu_time) and abs(relu_time - direct_first) <= tolerance
            box_at_first = [
                item for item in box_candidates if abs(item[0] - direct_first) <= tolerance
            ]
            relu_flips = 0
            if relu_is_first:
                delta = max(1e-10, 256.0 * tolerance)
                before = max(0.0, direct_first - delta)
                after = direct_first + delta
                pattern_before = pattern_arrays(
                    params, states[row], action + before * velocity
                )
                pattern_after = pattern_arrays(
                    params, states[row], action + after * velocity
                )
                relu_flips = sum(
                    int(np.count_nonzero(left != right))
                    for left, right in zip(pattern_before, pattern_after, strict=True)
                )
                if relu_flips == 0:
                    raise AssertionError(f"first ReLU event has no direct pattern flip row={row}")
            event_count = relu_flips + len(box_at_first)
            if event_count > 1:
                expected_code = FIRST_TIE
                ties_checked += 1
            elif relu_is_first:
                expected_code = FIRST_RELU
            else:
                expected_code = FIRST_BOX
            if int(geometry["first_hit_code"][row]) != expected_code:
                raise AssertionError(f"combined first-event type mismatch row={row}")
        elif math.isfinite(reported_first) or int(geometry["first_hit_code"][row]) != FIRST_NONE:
            raise AssertionError(f"spurious combined first event row={row}")
        combined_checked += 1
    if combined_checked == 0 or box_checked == 0:
        raise AssertionError("event verifier performed zero learned checks")
    return {
        "finite_box_rows_checked": box_checked,
        "finite_box_max_abs_error": max_box_error,
        "combined_event_rows_checked": combined_checked,
        "combined_ties_checked": ties_checked,
    }


def quadratic_reference() -> dict[str, Any]:
    """High-precision oracle independent of the runner's NumPy step loops."""
    with localcontext() as context:
        context.prec = 60
        action0 = (Decimal("0.2"), Decimal("-0.3"))
        diagonal = (Decimal("-0.25"), Decimal("-0.5"))
        linear = (Decimal("0.1"), Decimal("0.05"))
        total = Decimal("0.2")
        ks = (4, 8, 16)
        rate = tuple(Decimal(2) * value for value in diagonal)
        exact_factor = tuple((value * total).exp() for value in rate)
        exact_decimal = tuple(
            factor * start + (factor - 1) * (Decimal(2) * bias) / rate_value
            for factor, start, bias, rate_value in zip(
                exact_factor, action0, linear, rate, strict=True
            )
        )

        rows: list[dict[str, Any]] = []
        errors_decimal: dict[str, list[Decimal]] = {}
        for scheme in ("explicit", "implicit"):
            scheme_errors = []
            for k in ks:
                step = total / Decimal(k)
                if scheme == "explicit":
                    factor = tuple(1 + step * value for value in rate)
                else:
                    factor = tuple(1 / (1 - step * value) for value in rate)
                factor_k = tuple(value**k for value in factor)
                endpoint = tuple(
                    power * start
                    + (power - 1) * (Decimal(2) * bias) / rate_value
                    for power, start, bias, rate_value in zip(
                        factor_k, action0, linear, rate, strict=True
                    )
                )
                error = (
                    sum(
                        (value - target) ** 2
                        for value, target in zip(endpoint, exact_decimal, strict=True)
                    )
                    / Decimal(2)
                ).sqrt()
                scheme_errors.append(error)
                rows.append({"scheme": scheme, "K": k, "error": float(error)})
            errors_decimal[scheme] = scheme_errors

        log_steps = [(total / Decimal(k)).ln() for k in ks]
        mean_steps = sum(log_steps) / Decimal(len(log_steps))
        denominator = sum((value - mean_steps) ** 2 for value in log_steps)
        slopes = {}
        for scheme, errors in errors_decimal.items():
            log_errors = [value.ln() for value in errors]
            mean_errors = sum(log_errors) / Decimal(len(log_errors))
            slopes[scheme] = float(
                sum(
                    (step - mean_steps) * (error - mean_errors)
                    for step, error in zip(log_steps, log_errors, strict=True)
                )
                / denominator
            )
    passed = all(0.9 <= value <= 1.1 for value in slopes.values())
    return {
        "exact": np.asarray([float(value) for value in exact_decimal]),
        "rows": rows,
        "slopes_K4_8_16": slopes,
        "pass": passed,
    }


def synthetic_fixture(case: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Construct the hand-derived ReLU oracles without importing the runner."""
    if case == "layer1_exit":
        w1, b1 = np.array([[1.0, 0.0]]), np.array([0.1, 1.0])
        w2, b2 = np.eye(2), np.array([1.0, 1.0])
        w3, action = np.array([[-1.0], [0.0]]), np.array([[0.0]])
    elif case == "layer2_exit":
        w1, b1 = np.array([[1.0, 0.0]]), np.array([2.0, 1.0])
        w2, b2 = np.eye(2), np.array([-1.9, 1.0])
        w3, action = np.array([[-1.0], [0.0]]), np.array([[0.0]])
    elif case == "box_exit":
        w1, b1 = np.array([[1.0, 0.0]]), np.array([2.0, 1.0])
        w2, b2 = np.eye(2), np.array([1.0, 1.0])
        w3, action = np.array([[1.0], [0.0]]), np.array([[0.9]])
    elif case == "simultaneous_exit":
        w1, b1 = np.array([[1.0, 1.0]]), np.array([0.1, 0.1])
        w2, b2 = np.eye(2), np.array([1.0, 1.0])
        w3, action = np.array([[-0.5], [-0.5]]), np.array([[0.0]])
    elif case == "anchor_boundary":
        w1, b1 = np.array([[1.0, 0.0]]), np.array([0.0, 1.0])
        w2, b2 = np.eye(2), np.array([1.0, 1.0])
        w3, action = np.array([[-1.0], [0.0]]), np.array([[0.0]])
    else:
        raise KeyError(case)
    return {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": np.array([0.0]),
    }, action


def verify_harness(harness: dict[str, Any]) -> None:
    """Independently recompute all numeric and categorical harness oracles."""
    if harness.get("protocol") != PROTOCOL:
        raise AssertionError("harness protocol mismatch")
    reported_smooth = harness["smooth_quadratic"]
    expected_smooth = quadratic_reference()
    if not np.allclose(
        np.asarray(reported_smooth["exact"], dtype=np.float64),
        expected_smooth["exact"],
        rtol=1e-13,
        atol=1e-15,
    ):
        raise AssertionError("quadratic exact endpoint mismatch")
    expected_rows = {
        (row["scheme"], row["K"]): row["error"] for row in expected_smooth["rows"]
    }
    reported_rows = {
        (row["scheme"], int(row["K"])): float(row["error"])
        for row in reported_smooth["rows"]
    }
    if set(reported_rows) != set(expected_rows) or len(reported_smooth["rows"]) != 6:
        raise AssertionError("quadratic harness row grid mismatch")
    for key, expected_error in expected_rows.items():
        if not math.isclose(
            reported_rows[key], expected_error, rel_tol=1e-11, abs_tol=1e-15
        ):
            raise AssertionError(f"quadratic error mismatch: {key}")
    reported_slopes = reported_smooth["slopes_K4_8_16"]
    for scheme, expected_slope in expected_smooth["slopes_K4_8_16"].items():
        if not math.isclose(
            float(reported_slopes[scheme]), expected_slope, rel_tol=1e-11, abs_tol=1e-13
        ):
            raise AssertionError(f"quadratic slope mismatch: {scheme}")
    if reported_smooth.get("pass") is not expected_smooth["pass"]:
        raise AssertionError("quadratic pass flag mismatch")

    reported_cases = harness["synthetic_relu"]
    expected_case_names = {
        "resident",
        "layer1_exit",
        "layer2_exit",
        "box_exit",
        "simultaneous_exit",
        "anchor_boundary",
    }
    if set(reported_cases) != expected_case_names:
        raise AssertionError("synthetic ReLU harness case grid mismatch")
    labels = {FIRST_NONE: "none", FIRST_RELU: "relu", FIRST_BOX: "box", FIRST_TIE: "tie"}
    oracles = {
        "layer1_exit": (0.1, FIRST_RELU, 1, False),
        "layer2_exit": (0.1, FIRST_RELU, 2, False),
        "box_exit": (0.1, FIRST_BOX, -1, False),
        "simultaneous_exit": (0.1, FIRST_TIE, 1, False),
        "anchor_boundary": (math.inf, FIRST_NONE, -1, True),
    }
    geometries = {}
    for case, (oracle_time, oracle_code, oracle_layer, oracle_boundary) in oracles.items():
        params, action = synthetic_fixture(case)
        geometry = independent_geometry(params, np.zeros((1, 0)), action, 1.0)
        geometries[case] = geometry
        computed_time = float(geometry["first_hit_time"][0])
        if math.isfinite(oracle_time):
            if not math.isclose(computed_time, oracle_time, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"independent synthetic time oracle failed: {case}")
        elif not math.isinf(computed_time):
            raise AssertionError(f"independent synthetic infinity oracle failed: {case}")
        if int(geometry["first_hit_code"][0]) != oracle_code:
            raise AssertionError(f"independent synthetic type oracle failed: {case}")
        if int(geometry["first_layer"][0]) != oracle_layer:
            raise AssertionError(f"independent synthetic layer oracle failed: {case}")
        if bool(geometry["boundary_ambiguous"][0]) is not oracle_boundary:
            raise AssertionError(f"independent synthetic boundary oracle failed: {case}")

        reported = reported_cases[case]
        reported_time = reported["first_hit_time"]
        if math.isfinite(computed_time):
            if reported_time is None or not math.isclose(
                float(reported_time), computed_time, rel_tol=0.0, abs_tol=1e-12
            ):
                raise AssertionError(f"synthetic reported time mismatch: {case}")
        elif reported_time is not None:
            raise AssertionError(f"synthetic reported infinity mismatch: {case}")
        expected_fields = {
            "first_hit_type": labels[int(geometry["first_hit_code"][0])],
            "first_layer": int(geometry["first_layer"][0]),
            "boundary_ambiguous": bool(geometry["boundary_ambiguous"][0]),
            "pass": True,
        }
        for field, expected_value in expected_fields.items():
            if reported.get(field) != expected_value:
                raise AssertionError(f"synthetic reported {field} mismatch: {case}")

    resident_horizon = 0.05
    resident_category = classify(geometries["layer1_exit"], 0, resident_horizon)
    resident = reported_cases["resident"]
    if resident_category != "resident":
        raise AssertionError("independent resident oracle failed")
    if not math.isclose(float(resident.get("horizon")), resident_horizon, abs_tol=0.0):
        raise AssertionError("synthetic resident horizon mismatch")
    if resident.get("category") != resident_category or resident.get("pass") is not True:
        raise AssertionError("synthetic resident report mismatch")
    if harness.get("pass") is not bool(expected_smooth["pass"]):
        raise AssertionError("harness aggregate pass mismatch")


def verify_input_lock(lock: dict[str, Any], design: dict[str, Any]) -> str:
    without_hash = dict(lock)
    claimed_hash = without_hash.pop("input_lock_sha256")
    if sha256_bytes(canonical_bytes(without_hash)) != claimed_hash:
        raise AssertionError("INPUT_LOCK canonical hash mismatch")
    if lock.get("protocol") != PROTOCOL:
        raise AssertionError("INPUT_LOCK protocol mismatch")
    if lock.get("status") != "inputs_locked_before_semantic_read":
        raise AssertionError("INPUT_LOCK sequence declaration mismatch")
    if lock.get("development_only") is not True:
        raise AssertionError("INPUT_LOCK development flag mismatch")
    if lock.get("scientific_admissible") is not False:
        raise AssertionError("INPUT_LOCK admissibility flag mismatch")
    if lock.get("inputs") != PINNED_INPUTS:
        raise AssertionError("INPUT_LOCK does not bind the pinned exposed inputs")
    expected_exclusions = sorted(PINNED_EXCLUSIONS, key=lambda row: row["basename"])
    if lock.get("exclusions") != expected_exclusions:
        raise AssertionError("INPUT_LOCK exclusion inventory mismatch")
    if lock.get("exclusion_union_n_indices") != PINNED_EXCLUSION_UNION:
        raise AssertionError("INPUT_LOCK exclusion union mismatch")

    contract = design["pilot_input_contract"]
    if contract.get("bound_by_input_lock_before_semantic_read") is not True:
        raise AssertionError("DESIGN_LOCK does not require pre-read input binding")
    if contract.get("inputs") != PINNED_INPUTS:
        raise AssertionError("DESIGN_LOCK input contract mismatch")
    if contract.get("exclusions") != list(PINNED_EXCLUSIONS):
        raise AssertionError("DESIGN_LOCK exclusion contract mismatch")
    if contract.get("exclusion_union_n_indices") != PINNED_EXCLUSION_UNION:
        raise AssertionError("DESIGN_LOCK exclusion union mismatch")
    if contract.get("reject_missing_extra_or_empty_exclusion_inventory") is not True:
        raise AssertionError("DESIGN_LOCK exclusion strictness missing")
    return claimed_hash


CELL_FIELDS = [
    "environment",
    "seed",
    "T",
    "K",
    "h",
    "n_states",
    *[f"full_{name}" for name in CATEGORIES],
    *[f"step_{name}" for name in CATEGORIES],
    "full_residence_fraction",
    "step_residence_fraction",
    "full_affine_q_max_abs_residual",
]
CELL_INTEGER_FIELDS = {
    "seed",
    "K",
    "n_states",
    *[f"full_{name}" for name in CATEGORIES],
    *[f"step_{name}" for name in CATEGORIES],
}
CELL_FLOAT_FIELDS = {
    "T",
    "h",
    "full_residence_fraction",
    "step_residence_fraction",
    "full_affine_q_max_abs_residual",
}


def assert_cell_matches(
    actual: dict[str, Any],
    expected: dict[str, Any],
    label: str,
    affine_comparison_tolerance: float,
) -> None:
    if set(actual) != set(CELL_FIELDS):
        raise AssertionError(f"{label} cell schema mismatch")
    for name in CELL_FIELDS:
        observed = actual[name]
        target = expected[name]
        if name == "environment":
            if observed != target:
                raise AssertionError(f"{label} environment mismatch")
        elif name in CELL_INTEGER_FIELDS:
            if isinstance(observed, bool) or int(observed) != int(target):
                raise AssertionError(f"{label} integer mismatch: {name}")
        elif name in CELL_FLOAT_FIELDS:
            if target is None:
                if observed not in (None, ""):
                    raise AssertionError(f"{label} null residual mismatch")
                continue
            value = float(observed)
            if not math.isfinite(value):
                raise AssertionError(f"{label} nonfinite float: {name}")
            if name == "full_affine_q_max_abs_residual":
                tolerance = affine_comparison_tolerance
            else:
                tolerance = 64.0 * np.finfo(np.float64).eps * max(
                    1.0, abs(value), abs(float(target))
                )
            if not math.isclose(value, float(target), rel_tol=0.0, abs_tol=tolerance):
                raise AssertionError(f"{label} float mismatch: {name}")
        else:
            raise AssertionError(f"untyped cell field: {name}")


def affine_residual_reference(
    params: dict[str, np.ndarray],
    states: np.ndarray,
    anchors: np.ndarray,
    geometry: dict[str, np.ndarray],
    total: float,
) -> tuple[float | None, float, float]:
    resident = np.asarray(
        [classify(geometry, row, total) == "resident" for row in range(len(anchors))]
    )
    if not np.any(resident):
        return None, 0.0, 0.0
    endpoint = anchors + total * geometry["velocity"]
    q_endpoint = direct_forward(params, states, endpoint)[0]
    linear_term = total * np.sum(
        geometry["grad"] * geometry["velocity"], axis=-1
    )
    q_affine = geometry["q_anchor"] + linear_term
    residual = float(np.max(np.abs(q_endpoint[resident] - q_affine[resident])))
    scale = float(
        np.max(
            np.abs(q_endpoint[resident])
            + np.abs(geometry["q_anchor"][resident])
            + abs(total)
            * np.sum(
                np.abs(geometry["grad"][resident] * geometry["velocity"][resident]),
                axis=-1,
            )
        )
    )
    integrity_tolerance = AFFINE_RESIDUAL_ULPS * np.finfo(np.float64).eps * max(1.0, scale)
    comparison_tolerance = 128.0 * np.finfo(np.float64).eps * max(1.0, scale)
    if residual > integrity_tolerance:
        raise AssertionError(
            f"same-region affine residual exceeds scale-aware bound T={total}: "
            f"{residual} > {integrity_tolerance}"
        )
    return residual, integrity_tolerance, comparison_tolerance


def verify_bundle(out_dir: Path, *, check_existing: bool = False) -> dict[str, Any]:
    required = (
        "DESIGN_LOCK.json",
        "HARNESS.json",
        "INPUT_LOCK.json",
        "PILOT_INPUTS.json",
        "pilot_state_indices.npy",
        "STATE_GEOMETRY.npz",
        "residence_cells.csv",
        "SUMMARY.json",
        "MANIFEST.json",
        "STATUS.json",
    )
    for name in required:
        if not (out_dir / name).is_file():
            raise FileNotFoundError(out_dir / name)
    verify_path = out_dir / "VERIFY.json"
    if check_existing and not verify_path.is_file():
        raise FileNotFoundError(verify_path)
    if not check_existing and verify_path.exists():
        raise FileExistsError(verify_path)
    design = json.loads((out_dir / "DESIGN_LOCK.json").read_text())
    design_without_hash = dict(design)
    claimed_design_hash = design_without_hash.pop("design_sha256")
    if sha256_bytes(canonical_bytes(design_without_hash)) != claimed_design_hash:
        raise AssertionError("DESIGN_LOCK canonical hash mismatch")
    if design["protocol"] != PROTOCOL:
        raise AssertionError("protocol mismatch")
    if design["estimand"]["learned_slope"] is not None:
        raise AssertionError("learned slope must be absent")
    if design["estimand"]["learned_support_gate"] is not None:
        raise AssertionError("learned support gate must be absent")
    if (
        design["estimand"].get("unique_horizon_estimand")
        != "empirical survival curve P(t_exit > t)"
    ):
        raise AssertionError("unique-horizon estimand missing")
    if "never independent evidence" not in design["estimand"].get(
        "cell_dependence", ""
    ):
        raise AssertionError("repeated-cell dependence contract missing")
    final = design["final_contract_not_executed"]
    if final["excluded_family"] != "halfcheetah" or final["n_runs"] != 12:
        raise AssertionError("final pilot-separation contract mismatch")
    if final.get("exclude_every_development_attempt_index_set") is not True:
        raise AssertionError("final contract omits development-attempt exclusions")
    if final.get("development_attempts_include_failed_and_superseded_bundles") is not True:
        raise AssertionError("final contract omits failed/superseded attempts")
    if final.get("primary_summary") != "unique-horizon empirical survival curve P(t_exit > t)":
        raise AssertionError("final unique-horizon summary contract missing")
    if "do not count as independent evidence" not in final.get("repeated_cell_policy", ""):
        raise AssertionError("final repeated-cell policy missing")

    input_lock = json.loads((out_dir / "INPUT_LOCK.json").read_text())
    claimed_input_lock_hash = verify_input_lock(input_lock, design)

    manifest = json.loads((out_dir / "MANIFEST.json").read_text())
    status = json.loads((out_dir / "STATUS.json").read_text())
    summary = json.loads((out_dir / "SUMMARY.json").read_text())
    runtime = manifest.get("runtime", {})
    if runtime.get("backend") != "cpu":
        raise AssertionError("manifest CPU runtime missing")
    if runtime.get("jax_enable_x64") is not True:
        raise AssertionError("manifest x64 runtime missing")
    if manifest["status"] != status["status"] or manifest["status"] != "pilot_complete":
        raise AssertionError("pilot status mismatch")
    for doc in (manifest, status, summary):
        if doc["scientific_admissible"] is not False:
            raise AssertionError("pilot must remain scientifically inadmissible")
        if doc["development_only"] is not True:
            raise AssertionError("pilot must remain development-only")
    if summary["coverage_gate"] is not None or summary["learned_slope"] is not None:
        raise AssertionError("pilot contains a forbidden learned outcome gate")
    if manifest["design_sha256"] != claimed_design_hash:
        raise AssertionError("manifest design hash mismatch")
    if manifest.get("input_lock_sha256") != claimed_input_lock_hash:
        raise AssertionError("manifest INPUT_LOCK hash mismatch")
    expected_artifacts = set(required) - {"MANIFEST.json", "STATUS.json"}
    expected_artifacts.update(
        f"{SOURCE_SNAPSHOT_DIR}/{name}" for name in design["source_sha256"]
    )
    verify_artifact_inventory(out_dir, manifest.get("artifacts"), expected_artifacts)
    if design["source_sha256"] != manifest["source_sha256"]:
        raise AssertionError("design/manifest source hashes differ")
    snapshots = manifest.get("source_snapshots", {})
    if set(snapshots) != set(manifest["source_sha256"]):
        raise AssertionError("source snapshot inventory mismatch")
    for name, digest in manifest["source_sha256"].items():
        record = snapshots[name]
        expected_path = f"{SOURCE_SNAPSHOT_DIR}/{name}"
        if record != {"path": expected_path, "sha256": digest}:
            raise AssertionError(f"source snapshot record mismatch: {name}")
        if sha256_file(out_dir / expected_path) != digest:
            raise AssertionError(f"source snapshot content mismatch: {name}")
    verifier_name = Path(__file__).name
    if sha256_file(Path(__file__).resolve()) != manifest["source_sha256"][verifier_name]:
        raise AssertionError("invoke the exact snapshotted verifier")
    verify_git_provenance(manifest.get("git_provenance"))
    verify_harness(json.loads((out_dir / "HARNESS.json").read_text()))

    inputs = json.loads((out_dir / "PILOT_INPUTS.json").read_text())
    if inputs.get("input_lock_sha256") != claimed_input_lock_hash:
        raise AssertionError("PILOT_INPUTS/INPUT_LOCK hash mismatch")
    checkpoint = Path(inputs["checkpoint"])
    config = Path(inputs["config"])
    dataset = Path(inputs["dataset"])
    for role, path, key in (
        ("checkpoint", checkpoint, "checkpoint_sha256"),
        ("config", config, "config_sha256"),
        ("dataset", dataset, "dataset_sha256"),
    ):
        spec = PINNED_INPUTS[role]
        if path.name != spec["basename"] or inputs[key] != spec["sha256"]:
            raise AssertionError(f"{role} does not match INPUT_LOCK identity")
        if sha256_file(path) != spec["sha256"]:
            raise AssertionError(f"input hash mismatch: {path}")
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - declared trusted local checkpoint
    observations, actions, mean, std = load_rows(dataset)
    if sha256_array(np.asarray(payload["mean"], dtype=np.float32)) != inputs[
        "normalization_mean_sha256"
    ]:
        raise AssertionError("normalization mean fingerprint mismatch")
    if sha256_array(np.asarray(payload["std"], dtype=np.float32)) != inputs[
        "normalization_std_sha256"
    ]:
        raise AssertionError("normalization std fingerprint mismatch")
    if not np.allclose(mean, payload["mean"], rtol=0.0, atol=2e-5):
        raise AssertionError("recomputed normalization mean mismatch")
    if not np.allclose(std, payload["std"], rtol=0.0, atol=2e-5):
        raise AssertionError("recomputed normalization std mismatch")

    excluded: set[int] = set()
    exclusion_records = inputs["excluded_archived_index_files"]
    expected_exclusions = {row["basename"]: row for row in PINNED_EXCLUSIONS}
    if len(exclusion_records) != len(expected_exclusions):
        raise AssertionError("PILOT_INPUTS exclusion inventory length mismatch")
    seen_exclusions = set()
    for record in exclusion_records:
        path = Path(record["path"])
        name = path.name
        if name in seen_exclusions or name not in expected_exclusions:
            raise AssertionError(f"unexpected or duplicate exclusion: {name}")
        seen_exclusions.add(name)
        spec = expected_exclusions[name]
        if record["sha256"] != spec["sha256"] or record["n_indices"] != spec["n_indices"]:
            raise AssertionError(f"exclusion record mismatch: {name}")
        if sha256_file(path) != spec["sha256"]:
            raise AssertionError(f"excluded index hash mismatch: {path}")
        values = np.asarray(np.load(path), dtype=np.int64)
        if len(values) != spec["n_indices"]:
            raise AssertionError(f"excluded index count mismatch: {name}")
        excluded.update(int(value) for value in values)
    if seen_exclusions != set(expected_exclusions) or len(excluded) != PINNED_EXCLUSION_UNION:
        raise AssertionError("canonical exclusion union mismatch")
    if inputs.get("excluded_union_n_indices") != PINNED_EXCLUSION_UNION:
        raise AssertionError("PILOT_INPUTS exclusion union field mismatch")
    eligible = np.flatnonzero(np.max(np.abs(actions), axis=-1) <= ACTION_ELIGIBILITY)
    eligible = np.array([value for value in eligible if int(value) not in excluded])
    regenerated = np.random.default_rng(PILOT_SAMPLE_SEED).choice(
        eligible, size=N_PILOT, replace=False
    ).astype(np.int64)
    selected = np.asarray(np.load(out_dir / "pilot_state_indices.npy"), dtype=np.int64)
    if not np.array_equal(regenerated, selected):
        raise AssertionError("selected indices do not regenerate")
    if excluded.intersection(int(value) for value in selected):
        raise AssertionError("selected pilot rows overlap archived rows")
    if sha256_array(selected) != inputs["selected_indices_sha256"]:
        raise AssertionError("selected index content hash mismatch")

    states = (
        observations[selected].astype(np.float64)
        - np.asarray(payload["mean"], dtype=np.float64)
    ) / np.asarray(payload["std"], dtype=np.float64)
    anchors = actions[selected].astype(np.float64)
    params = extract_q1(payload)
    q_anchor = direct_forward(params, states, anchors)[0]
    c_ref = float(np.mean(np.abs(q_anchor)) + 1e-6)
    geometry = independent_geometry(params, states, anchors, c_ref)
    stored = np.load(out_dir / "STATE_GEOMETRY.npz")
    expected_arrays = {
        "dataset_row": selected,
        "normalized_state": states,
        "anchor_action": anchors,
        **geometry,
    }
    expected_stored_names = set(expected_arrays) | {"grad_jax", "c_ref"}
    if set(stored.files) != expected_stored_names:
        raise AssertionError("STATE_GEOMETRY array schema mismatch")
    for name, expected in expected_arrays.items():
        actual = stored[name]
        if expected.dtype.kind in "biu":
            equal = np.array_equal(actual, expected)
        else:
            # This diagnostic ratio can be O(1e9); independent BLAS reduction
            # order may differ slightly without affecting any classification.
            rtol = 1e-9 if name == "anchor_min_scaled_margin" else 1e-11
            equal = np.allclose(actual, expected, rtol=rtol, atol=1e-11, equal_nan=True)
        if not equal:
            raise AssertionError(f"geometry mismatch: {name}")
    if not math.isclose(float(stored["c_ref"]), c_ref, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("C_ref mismatch")
    grad_jax = np.asarray(stored["grad_jax"], dtype=np.float64)
    if not np.allclose(grad_jax, geometry["grad"], rtol=1e-10, atol=1e-10):
        raise AssertionError("stored JAX gradient does not match independent gradient")
    analytic_jax_error = float(np.max(np.abs(grad_jax - geometry["grad"])))

    fd_checked, fd_max_error = finite_difference_checks(
        params, states, anchors, geometry
    )
    exit_checked, exit_max_error = exit_bracket_checks(
        params, states, anchors, geometry
    )
    if fd_checked == 0 or exit_checked == 0:
        raise AssertionError("verifier performed zero finite-difference/ReLU checks")
    event_report = box_and_combined_event_checks(params, states, anchors, geometry)

    with (out_dir / "residence_cells.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CELL_FIELDS:
            raise AssertionError("residence CSV header mismatch")
        rows = list(reader)
    if len(rows) != len(TIMES) * len(SUBSTEPS):
        raise AssertionError("residence cell count mismatch")
    keys = {(float(row["T"]), int(row["K"])) for row in rows}
    if keys != {(total, k) for total in TIMES for k in SUBSTEPS}:
        raise AssertionError("residence cell grid mismatch")
    by_key = {(float(row["T"]), int(row["K"])): row for row in rows}
    expected_cells: dict[tuple[float, int], tuple[dict[str, Any], float]] = {}
    affine_residuals_checked = 0
    affine_integrity_max = 0.0
    for total in TIMES:
        full_expected = counts(geometry, total)
        residual, integrity_tolerance, comparison_tolerance = affine_residual_reference(
            params, states, anchors, geometry, total
        )
        if residual is not None:
            affine_residuals_checked += 1
            affine_integrity_max = max(affine_integrity_max, integrity_tolerance)
        prior_step_resident = -1
        first_full = None
        for k in SUBSTEPS:
            row = by_key[(total, k)]
            step = total / k
            step_expected = counts(geometry, step)
            expected_row = {
                "environment": ENVIRONMENT,
                "seed": SEED,
                "T": total,
                "K": k,
                "h": step,
                "n_states": N_PILOT,
                **{f"full_{name}": full_expected[name] for name in CATEGORIES},
                **{f"step_{name}": step_expected[name] for name in CATEGORIES},
                "full_residence_fraction": full_expected["resident"] / N_PILOT,
                "step_residence_fraction": step_expected["resident"] / N_PILOT,
                "full_affine_q_max_abs_residual": residual,
            }
            assert_cell_matches(
                row,
                expected_row,
                f"CSV T={total} K={k}",
                comparison_tolerance,
            )
            expected_cells[(total, k)] = (expected_row, comparison_tolerance)
            if sum(int(row[f"full_{name}"]) for name in CATEGORIES) != N_PILOT:
                raise AssertionError("full categories are not exhaustive")
            if sum(int(row[f"step_{name}"]) for name in CATEGORIES) != N_PILOT:
                raise AssertionError("step categories are not exhaustive")
            current = int(row["step_resident"])
            if current < prior_step_resident:
                raise AssertionError("step residence is not monotone in K")
            prior_step_resident = current
            signature = tuple(int(row[f"full_{name}"]) for name in CATEGORIES)
            if first_full is None:
                first_full = signature
            elif signature != first_full:
                raise AssertionError("full residence changed with K")

    if summary.get("status") != "development_pilot_complete":
        raise AssertionError("summary status mismatch")
    if summary.get("n_states") != N_PILOT or summary.get("n_cells") != len(expected_cells):
        raise AssertionError("summary dimensions mismatch")
    if not math.isclose(float(summary.get("C_ref")), c_ref, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("summary C_ref mismatch")
    if not math.isclose(
        float(summary.get("analytic_vs_jax_grad_max_abs")),
        analytic_jax_error,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("summary analytic/JAX error mismatch")
    summary_cells = summary.get("cells", [])
    if len(summary_cells) != len(expected_cells):
        raise AssertionError("SUMMARY/CSV cell count mismatch")
    summary_by_key = {(float(row["T"]), int(row["K"])): row for row in summary_cells}
    if set(summary_by_key) != set(expected_cells):
        raise AssertionError("SUMMARY/CSV cell key mismatch")
    for key, (expected_row, comparison_tolerance) in expected_cells.items():
        assert_cell_matches(
            summary_by_key[key],
            expected_row,
            f"SUMMARY T={key[0]} K={key[1]}",
            comparison_tolerance,
        )

    report = {
        "protocol": PROTOCOL,
        "pass": True,
        "artifact_integrity_pass": True,
        "independent_geometry_recompute_pass": True,
        "finite_difference_pass": True,
        "finite_difference_coordinates_checked": fd_checked,
        "finite_difference_max_abs_error": fd_max_error,
        "exit_bracket_pass": True,
        "relu_exits_bracketed_within_Tmax": exit_checked,
        "exit_bracket_max_abs_error": exit_max_error,
        "finite_box_pass": True,
        "finite_box_rows_checked": event_report["finite_box_rows_checked"],
        "finite_box_max_abs_error": event_report["finite_box_max_abs_error"],
        "combined_event_pass": True,
        "combined_event_rows_checked": event_report["combined_event_rows_checked"],
        "combined_ties_checked": event_report["combined_ties_checked"],
        "derived_cells_recomputed_pass": True,
        "affine_residuals_checked": affine_residuals_checked,
        "affine_integrity_max_tolerance": affine_integrity_max,
        "development_only": True,
        "scientific_admissible": False,
        "manifest_sha256": sha256_file(out_dir / "MANIFEST.json"),
        "design_sha256": claimed_design_hash,
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "written_at": now_iso(),
    }
    if check_existing:
        verify_stored_report(verify_path, report)
    else:
        write_json_create(verify_path, report)
    return report


def main() -> int:
    args = parse_args()
    try:
        report = verify_bundle(args.out_dir, check_existing=args.check_existing)
    except Exception as exc:
        print(f"FAIL P2 ReLU residence pilot: {type(exc).__name__}: {exc}")
        return 1
    print(
        "PASS P2 ReLU residence pilot: independent geometry, finite differences, "
        f"and exits verified ({report['finite_difference_coordinates_checked']} FD coordinates)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
