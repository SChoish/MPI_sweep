#!/usr/bin/env python3
"""Fixed-operator convergence-order audit (TODO.md P0-A).

CPU-only. Freezes one critic and batch scale, then measures explicit / backward
Euler global error versus an independent RK4 reference as K increases.

Phases:
  harness  — analytic d=2 validation (no checkpoints)
  resolve  — fingerprint datasets + locate the 18 T=1 critics
  protocol — write a draft, or freeze only after all fingerprints resolve
  audit    — full 18-run primary audit (requires resolved checkpoints)
  all      — harness → resolve → protocol → audit-if-ready
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

from _lab_import import REPO_ROOT as ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from train_td3bc import (  # noqa: E402
    DATASET_FILES,
    TwinCritic,
    load_checkpoint,
    load_transition,
)

ENVIRONMENTS = list(
    json.loads(
        (
            ROOT
            / "sweep_results"
            / "diagnostics"
            / "frozen_critic_small_step"
            / "MANIFEST.json"
        ).read_text(encoding="utf-8")
    )["environments"]
)
SEEDS = (0, 1)
TOTAL_TIMES = (0.025, 0.05, 0.1, 0.2)
SUBSTEPS = (1, 2, 4, 8, 16)
EPS = 1e-6
BOX = 1.0
SAMPLE_BASE = 20260829
N_STATES = 512
MIN_MASK = 410
IMPLICIT_TOL = 1e-10
IMPLICIT_MAX_ITERS = 1000
RK4_ABS = 1e-10
RK4_REL = 1e-6
RK4_N_CANDIDATES = (256, 512, 1024, 2048, 4096)
DEFAULT_OUT = ROOT / "sweep_results" / "diagnostics" / "fixed_operator_order"
DEFAULT_CKPT_ROOTS = (
    "/home/ext_csv/mpi_sweep_lab/results_qnorm",
    str(ROOT / "results_qnorm"),
    str(Path.home() / "mpi_sweep_lab" / "results_qnorm"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=("harness", "resolve", "protocol", "audit", "all"),
        default="all",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument(
        "--checkpoint-roots",
        nargs="+",
        default=list(DEFAULT_CKPT_ROOTS),
        help="Directories containing {env}_tau1_seed{s}/params_1000000.pkl",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="If >0, audit only the first N env-seed runs (debug).",
    )
    return p.parse_args()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(blob)


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return value


def tree_params(payload: dict[str, Any], key: str) -> Any:
    value = payload[key]
    if isinstance(value, dict) and set(value) == {"params"}:
        return value
    return value


def dataset_path(data_dir: Path, env_name: str) -> Path:
    return data_dir / DATASET_FILES[env_name]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Analytic harness
# ---------------------------------------------------------------------------


def analytic_linear_field(b: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    b = np.asarray(b, dtype=np.float64)

    def f(a: np.ndarray) -> np.ndarray:
        # f = d * grad(b·a) / C with d=2, C=1 → 2b
        return np.broadcast_to(2.0 * b, a.shape)

    return f


def analytic_quadratic_field(A: np.ndarray, b: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def f(a: np.ndarray) -> np.ndarray:
        # Q = a^T A a / 2 + b^T a; grad = A a + b; f = 2 * grad (d=2,C=1)
        return 2.0 * (a @ A.T + b)

    return f


def explicit_euler(
    f: Callable[[np.ndarray], np.ndarray], a0: np.ndarray, dt: float, k: int
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    a = np.array(a0, dtype=np.float64, copy=True)
    oob = np.zeros(a.shape[:-1], dtype=bool)
    actions = []
    step_oob = []
    nonfinite = []
    for _ in range(k):
        nxt = a + dt * f(a)
        current_oob = np.any(np.abs(nxt) > BOX, axis=-1)
        current_nonfinite = ~np.all(np.isfinite(nxt), axis=-1)
        oob |= current_oob
        a = nxt
        actions.append(a.copy())
        step_oob.append(current_oob)
        nonfinite.append(current_nonfinite)
    trace = {
        "actions": np.stack(actions),
        "oob": np.stack(step_oob),
        "nonfinite": np.stack(nonfinite),
    }
    return a, oob, trace


def backward_euler(
    f: Callable[[np.ndarray], np.ndarray],
    a0: np.ndarray,
    dt: float,
    k: int,
    tol: float = IMPLICIT_TOL,
    max_iters: int = IMPLICIT_MAX_ITERS,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    a = np.array(a0, dtype=np.float64, copy=True)
    oob = np.zeros(a.shape[:-1], dtype=bool)
    conv = np.ones(a.shape[:-1], dtype=bool)
    resid = np.zeros(a.shape[:-1], dtype=np.float64)
    iters = np.zeros(a.shape[:-1], dtype=np.int32)
    path_actions = []
    path_residual = []
    path_delta = []
    path_iterations = []
    path_converged = []
    path_oob = []
    path_nonfinite = []
    for _ in range(k):
        prev = a
        cand = prev.copy()
        active = np.ones(a.shape[:-1], dtype=bool)
        ok = np.zeros_like(active)
        last_res = np.full(a.shape[:-1], np.inf, dtype=np.float64)
        last_delta = np.full(a.shape[:-1], np.inf, dtype=np.float64)
        state_iters = np.zeros(a.shape[:-1], dtype=np.int32)
        failed_nonfinite = np.zeros(a.shape[:-1], dtype=bool)
        for n_it in range(1, max_iters + 1):
            # Fixed-point map: G(a)=a_prev + dt*f(a); converge a = G(a).
            proposal = prev + dt * f(cand)
            delta = np.max(np.abs(proposal - cand), axis=-1)
            res = np.max(
                np.abs(proposal - (prev + dt * f(proposal))), axis=-1
            )
            finite = (
                np.all(np.isfinite(proposal), axis=-1)
                & np.isfinite(delta)
                & np.isfinite(res)
            )
            cand = np.where(active[..., None], proposal, cand)
            last_res = np.where(active, res, last_res)
            last_delta = np.where(active, delta, last_delta)
            state_iters = np.where(active, n_it, state_iters)
            newly_ok = active & finite & (res <= tol) & (delta <= tol)
            ok |= newly_ok
            failed_nonfinite |= active & ~finite
            active &= finite & ~newly_ok
            if not bool(np.any(active)):
                break
        a = cand
        current_oob = np.any(np.abs(a) > BOX, axis=-1)
        oob |= current_oob
        conv &= ok
        resid = np.maximum(resid, last_res)
        iters = np.maximum(iters, state_iters)
        path_actions.append(a.copy())
        path_residual.append(last_res.copy())
        path_delta.append(last_delta.copy())
        path_iterations.append(state_iters.copy())
        path_converged.append(ok.copy())
        path_oob.append(current_oob)
        path_nonfinite.append(failed_nonfinite | ~np.all(np.isfinite(a), axis=-1))
    trace = {
        "actions": np.stack(path_actions),
        "residual": np.stack(path_residual),
        "delta": np.stack(path_delta),
        "iterations": np.stack(path_iterations),
        "converged": np.stack(path_converged),
        "oob": np.stack(path_oob),
        "nonfinite": np.stack(path_nonfinite),
    }
    return a, oob, conv, resid, iters, trace


def rk4_step(f: Callable[[np.ndarray], np.ndarray], a: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    k1 = f(a)
    k2 = f(a + 0.5 * dt * k1)
    k3 = f(a + 0.5 * dt * k2)
    k4 = f(a + dt * k3)
    nxt = a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    stage_pts = (
        a,
        a + 0.5 * dt * k1,
        a + 0.5 * dt * k2,
        a + dt * k3,
        nxt,
    )
    oob = np.zeros(a.shape[:-1], dtype=bool)
    for pt in stage_pts:
        oob |= np.any(np.abs(pt) > BOX, axis=-1)
    return nxt, oob


def rk4_integrate(
    f: Callable[[np.ndarray], np.ndarray], a0: np.ndarray, t: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    a = np.array(a0, dtype=np.float64, copy=True)
    oob = np.zeros(a.shape[:-1], dtype=bool)
    dt = t / n
    for _ in range(n):
        a, step_oob = rk4_step(f, a, dt)
        oob |= step_oob
    return a, oob


def accept_rk4_reference(
    f: Callable[[np.ndarray], np.ndarray], a0: np.ndarray, t: float
) -> dict[str, Any]:
    a_d = np.asarray(a0, dtype=np.float64)
    chosen = None
    for n in RK4_N_CANDIDATES:
        a_n, oob_n = rk4_integrate(f, a_d, t, n)
        a_2n, oob_2n = rk4_integrate(f, a_d, t, 2 * n)
        r_n = float(np.sqrt(np.mean(np.square(a_n - a_2n))))
        m_2n = float(np.sqrt(np.mean(np.square(a_2n - a_d))))
        thr = max(RK4_ABS, RK4_REL * m_2n)
        record = {
            "n": n,
            "R_N": r_n,
            "M_2N": m_2n,
            "threshold": thr,
            "accepted": r_n <= thr,
            "oob_any": bool(np.any(oob_n) or np.any(oob_2n)),
            "endpoint": a_2n,
            "oob": oob_2n,
        }
        if record["accepted"]:
            chosen = record
            break
        chosen = record
    if chosen is None or not chosen["accepted"]:
        return {
            "stable": False,
            "n_micro": int(2 * chosen["n"]) if chosen else None,
            "R_N": chosen["R_N"] if chosen else None,
            "M_2N": chosen["M_2N"] if chosen else None,
            "endpoint": chosen["endpoint"] if chosen else None,
            "oob": chosen["oob"] if chosen else None,
        }
    return {
        "stable": True,
        "n_micro": int(2 * chosen["n"]),
        "R_N": chosen["R_N"],
        "M_2N": chosen["M_2N"],
        "endpoint": chosen["endpoint"],
        "oob": chosen["oob"],
    }


def rms_per_dim(err: np.ndarray) -> float:
    # mean(square) already averages over both states and action coordinates,
    # which equals mean_i(||err_i||^2 / d).
    return float(np.sqrt(np.mean(np.square(err))))


def fit_log_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or any(y <= 0 for y in ys):
        return None
    x = np.log(np.asarray(xs, dtype=np.float64))
    y = np.log(np.asarray(ys, dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])


def run_harness(out_dir: Path) -> dict[str, Any]:
    a0 = np.asarray([0.2, -0.3], dtype=np.float64)
    t = 0.2
    # Linear Q = b^T a
    b = np.asarray([0.1, -0.2], dtype=np.float64)
    f_lin = analytic_linear_field(b)
    exact_lin = a0 + 2.0 * t * b
    lin_rows = []
    for scheme in ("explicit", "implicit"):
        for k in (1, 2, 4, 8, 16):
            dt = t / k
            if scheme == "explicit":
                end, oob, _trace = explicit_euler(f_lin, a0, dt, k)
            else:
                end, oob, *_ = backward_euler(f_lin, a0, dt, k)
            err = float(np.max(np.abs(end - exact_lin)))
            lin_rows.append(
                {
                    "scheme": scheme,
                    "K": k,
                    "max_abs_err": err,
                    "pass": err <= 1e-10,
                    "oob": bool(np.any(oob)),
                }
            )
    lin_ok = all(r["pass"] and not r["oob"] for r in lin_rows)

    # Quadratic
    A = np.diag([-0.25, -0.5])
    bq = np.asarray([0.1, 0.05], dtype=np.float64)
    f_q = analytic_quadratic_field(A, bq)
    # exact: exp(2 A T) a0 + (2A)^{-1}(exp(2 A T)-I) 2 b
    two_a = 2.0 * A
    exp_mat = np.diag(np.exp(np.diag(two_a) * t))
    inv_two_a = np.diag(1.0 / np.diag(two_a))
    exact_q = exp_mat @ a0 + inv_two_a @ (exp_mat - np.eye(2)) @ (2.0 * bq)
    ref = accept_rk4_reference(f_q, a0, t)
    quad_rows = []
    slopes = {}
    for scheme in ("explicit", "implicit"):
        abs_errs = {}
        for k in SUBSTEPS:
            dt = t / k
            if scheme == "explicit":
                end, oob, _trace = explicit_euler(f_q, a0, dt, k)
                conv = True
            else:
                end, oob, conv_arr, *_ = backward_euler(f_q, a0, dt, k)
                conv = bool(np.all(conv_arr))
            e_abs = rms_per_dim(end - exact_q)
            abs_errs[k] = e_abs
            quad_rows.append(
                {
                    "scheme": scheme,
                    "K": k,
                    "E_abs_vs_exact": e_abs,
                    "converged": conv,
                    "oob": bool(np.any(oob)),
                }
            )
        ks = [4, 8, 16]
        slope = fit_log_slope([t / k for k in ks], [abs_errs[k] for k in ks])
        slopes[scheme] = slope
    quad_ok = all(
        slopes[s] is not None and 0.9 <= slopes[s] <= 1.1 for s in ("explicit", "implicit")
    ) and bool(ref["stable"])

    report = {
        "linear_Q": {"rows": lin_rows, "pass": lin_ok, "exact": exact_lin.tolist()},
        "quadratic_Q": {
            "rows": quad_rows,
            "slopes_K4_8_16": slopes,
            "rk4_vs_exact_stable": ref["stable"],
            "rk4_n_micro": ref["n_micro"],
            "exact": exact_q.tolist(),
            "pass": quad_ok,
        },
        "pass": bool(lin_ok and quad_ok),
        "written_at": now_iso(),
    }
    (out_dir / "HARNESS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"harness_pass": report["pass"], "slopes": slopes}, indent=2), flush=True)
    if not report["pass"]:
        raise SystemExit("analytic harness failed")
    return report


# ---------------------------------------------------------------------------
# Resolve checkpoints + sample indices
# ---------------------------------------------------------------------------


def resolve_checkpoints(out_dir: Path, data_dir: Path, roots: list[str]) -> dict[str, Any]:
    entries = []
    missing = []
    for j, env_name in enumerate(ENVIRONMENTS):
        ds = dataset_path(data_dir, env_name)
        if not ds.is_file():
            raise FileNotFoundError(ds)
        data, mean, std = load_transition(env_name, data_dir, normalize=True)
        actions = np.asarray(data.actions)
        observations = np.asarray(data.observations)
        interior = np.all(np.abs(actions) <= 0.95, axis=-1)
        candidates = np.flatnonzero(interior)
        if candidates.size < N_STATES:
            raise RuntimeError(f"{env_name}: only {candidates.size} interior rows")
        ds_hash = sha256_file(ds)
        mean_hash = sha256_bytes(np.asarray(mean, dtype=np.float64).tobytes())
        std_hash = sha256_bytes(np.asarray(std, dtype=np.float64).tobytes())
        norm_hash = sha256_bytes(
            np.asarray(mean, dtype=np.float64).tobytes()
            + np.asarray(std, dtype=np.float64).tobytes()
        )
        for seed in SEEDS:
            rng = np.random.default_rng(SAMPLE_BASE + 100 * j + seed)
            selected = rng.choice(candidates, size=N_STATES, replace=False)
            # Preserve the exact RNG order used by the archived audit.
            selected = selected.astype(np.int64)
            idx_hash = sha256_bytes(selected.tobytes())
            expected_name = f"{env_name}_tau1_seed{seed}"
            found = None
            for root in roots:
                cand = Path(root) / expected_name / "params_1000000.pkl"
                if cand.is_file():
                    found = cand.resolve()
                    break
            # also honor archived CSV absolute path if present locally
            if found is None:
                archived = Path(
                    f"/home/ext_csv/mpi_sweep_lab/results_qnorm/{expected_name}/params_1000000.pkl"
                )
                if archived.is_file():
                    found = archived.resolve()
            rec: dict[str, Any] = {
                "environment": env_name,
                "environment_index": j,
                "seed": seed,
                "update": 1_000_000,
                "expected_run_dir": expected_name,
                "dataset_identifier": env_name,
                "dataset_path": str(ds.resolve()),
                "dataset_sha256": ds_hash,
                "state_mean_sha256": mean_hash,
                "state_std_sha256": std_hash,
                "normalization_statistics_sha256": norm_hash,
                "state_indices_sha256": idx_hash,
                "n_states": N_STATES,
                "state_indices_path": str(
                    (out_dir / "state_indices" / f"{env_name}_seed{seed}.npy").resolve()
                ),
            }
            (out_dir / "state_indices").mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "state_indices" / f"{env_name}_seed{seed}.npy", selected)
            if found is None:
                rec["resolved"] = False
                rec["resolution_error"] = "checkpoint missing"
                rec["checkpoint_path"] = None
                rec["weights_sha256"] = None
                rec["config_sha256"] = None
                missing.append(expected_name)
            else:
                payload = load_checkpoint(found)
                if not np.allclose(np.asarray(payload["mean"]), mean):
                    raise ValueError(f"mean mismatch: {found}")
                if not np.allclose(np.asarray(payload["std"]), std):
                    raise ValueError(f"std mismatch: {found}")
                rec["checkpoint_path"] = str(found)
                rec["weights_sha256"] = sha256_file(found)
                config = payload.get("config")
                config_path = found.parent / "config.json"
                if config is None and config_path.is_file():
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                if config is None:
                    rec["resolved"] = False
                    rec["resolution_error"] = "config metadata missing"
                    rec["config"] = None
                    rec["config_sha256"] = None
                    missing.append(expected_name)
                else:
                    config = jsonable(config)
                    config_env = config.get("env")
                    config_seed = config.get("seed")
                    config_tau = config.get("tau")
                    step = int(payload.get("step", 0))
                    mismatches = []
                    if config_env is not None and config_env != env_name:
                        mismatches.append(f"env={config_env}")
                    if config_seed is not None and int(config_seed) != seed:
                        mismatches.append(f"seed={config_seed}")
                    if config_tau is not None and float(config_tau) != 1.0:
                        mismatches.append(f"tau={config_tau}")
                    if step != 1_000_000:
                        mismatches.append(f"step={step}")
                    if mismatches:
                        raise ValueError(
                            f"checkpoint identity mismatch {found}: "
                            + ", ".join(mismatches)
                        )
                    rec["resolved"] = True
                    rec["resolution_error"] = None
                    rec["config"] = config
                    rec["config_sha256"] = sha256_json(config)
                    rec["config_path"] = (
                        str(config_path.resolve()) if config_path.is_file() else None
                    )
                    rec["config_file_sha256"] = (
                        sha256_file(config_path) if config_path.is_file() else None
                    )
            entries.append(rec)
        del data, observations, actions

    doc = {
        "experiment": "fixed_operator_order",
        "source_manifest": "sweep_results/diagnostics/frozen_critic_small_step/MANIFEST.json",
        "checkpoint_rule": "canonical TD3+BC T=1 final critic, seeds 0-1, nine envs",
        "n_expected": 18,
        "n_resolved": sum(1 for e in entries if e["resolved"]),
        "checkpoint_roots_tried": roots,
        "written_at": now_iso(),
        "entries": entries,
        "missing_run_dirs": missing,
    }
    out_name = "CHECKPOINTS.json" if not missing else "CHECKPOINTS_UNRESOLVED.json"
    path = out_dir / out_name
    path.write_text(json.dumps(doc, indent=2) + "\n")
    stale = out_dir / (
        "CHECKPOINTS_UNRESOLVED.json" if not missing else "CHECKPOINTS.json"
    )
    if stale.is_file():
        stale.unlink()
    # Always also write a pointer STATUS
    status = {
        "checkpoints_ready": not missing,
        "n_resolved": doc["n_resolved"],
        "n_expected": 18,
        "missing_run_dirs": missing,
        "artifact": str(path.resolve()),
        "blocker": (
            None
            if not missing
            else (
                "The 18 canonical T=1 critics live under "
                "/home/ext_csv/mpi_sweep_lab/results_qnorm/ on ext_csv and are "
                "not present on this host. Sync or pass --checkpoint-roots."
            )
        ),
        "written_at": now_iso(),
    }
    (out_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)
    return doc


def protocol_document(
    out_dir: Path,
    harness: dict[str, Any] | None,
    *,
    frozen: bool,
) -> dict[str, Any]:
    checkpoint_manifest = out_dir / "CHECKPOINTS.json"
    unresolved_manifest = out_dir / "CHECKPOINTS_UNRESOLVED.json"
    inventory_path = (
        checkpoint_manifest
        if checkpoint_manifest.is_file()
        else unresolved_manifest
    )
    code_path = ROOT / "scripts" / "diagnostics" / "run_fixed_operator_order.py"
    protocol = {
        "name": "fixed_operator_order_v2",
        "written_at": now_iso(),
        "locked": frozen,
        "note": (
            "Immutable after creation; do not edit after reading learned-critic errors."
            if frozen
            else "Draft only: checkpoints are unresolved, so this is not the frozen protocol."
        ),
        "grid": {
            "T": list(TOTAL_TIMES),
            "K": list(SUBSTEPS),
            "schemes": ["explicit", "implicit"],
            "n_states": N_STATES,
            "seeds": list(SEEDS),
            "environments": ENVIRONMENTS,
        },
        "operator": {
            "f_s(a)": "d * grad_a Q_ref(s,a) / C_ref",
            "C_ref": "mean_i(|Q_ref(s_i,a_D_i)|) + epsilon",
            "epsilon": EPS,
            "action_box": [-BOX, BOX],
            "precision": "float64",
            "projection_in_primary": False,
        },
        "sampling": {
            "rng": "numpy.random.default_rng(20260829 + 100*j + z)",
            "eligibility": "max(abs(a_D)) <= 0.95",
            "without_replacement": True,
        },
        "rk4": {
            "independent_code_path": True,
            "N_candidates": list(RK4_N_CANDIDATES),
            "maximum_accepted_microsteps": 8192,
            "accept": "R_N <= max(1e-10, 1e-6 * M_2N); use 2N endpoint",
        },
        "implicit_solver": {
            "equation": "a_next = a_prev + dt * f_s(a_next)",
            "init": "a_prev only",
            "tol_inf": IMPLICIT_TOL,
            "max_iters": IMPLICIT_MAX_ITERS,
        },
        "mask": {
            "I": "intersection over all K, both schemes, and RK4 reference",
            "requires": "finite, implicit converged, no projection/OOB",
            "min_|I|": MIN_MASK,
            "min_eligible_T_per_run": 3,
        },
        "errors": {
            "E_abs": "sqrt(mean_I ||a-a_ref||_2^2 / d)",
            "M_ref": "sqrt(mean_I ||a_ref-a_D||_2^2 / d)",
            "E_rel": "E_abs / max(M_ref, 1e-8)",
            "slope_fit": "log(E_abs) ~ log(T/K) over K in {4,8,16}",
        },
        "expected_counts": {
            "endpoint_cells": 18 * 4 * 5 * 2,
            "raw_state_rows": 18 * 4 * 5 * 2 * 512,
            "implicit_substep_records": 18 * 4 * 512 * (1 + 2 + 4 + 8 + 16),
        },
        "output_schema": {
            "endpoint_errors.csv": [
                "environment",
                "seed",
                "T",
                "K",
                "scheme",
                "E_abs",
                "E_rel",
                "M_ref",
                "ratio_to_2K",
                "slope_K4_8_16",
                "slope_status",
                "|I|",
                "primary_cell",
                "reference_stable",
                "failure_reason",
                "C_ref",
            ],
            "raw_state_diagnostics.npz": {
                "rows": "720*512",
                "arrays": [
                    "cell_index",
                    "state_index",
                    "error_sq_per_dim",
                    "finite",
                    "oob",
                    "implicit_converged",
                    "common_mask",
                ],
            },
            "implicit_substeps.npz": {
                "rows": "18*4*512*31",
                "arrays": [
                    "run_index",
                    "T_index",
                    "K",
                    "substep",
                    "state_index",
                    "residual",
                    "delta",
                    "iterations",
                    "converged",
                    "oob",
                    "nonfinite",
                    "action",
                ],
            },
            "euler_paths.npz": {
                "rows": "2*18*4*512*31",
                "description": "Unprojected action at every Euler substep.",
            },
            "masks.npz": {
                "shape": [72, 512],
                "description": "Realized common mask for every env-seed-T cell.",
            },
            "per_t_slopes.json": "Per env-seed-T-scheme slope or below-floor/failure status.",
            "SUMMARY.json": "18 run summaries, nine two-seed task means, task-equal aggregate, gates.",
            "RESULT_MANIFEST.json": "Counts, realized mask metadata, and artifact SHA-256 values.",
        },
        "scientific_gates": {
            "aggregate_median_slope": [0.75, 1.25],
            "run_slopes_in_[0.5,1.5]": ">=14/18",
            "complete_runs": ">=15/18",
        },
        "harness_pass": None if harness is None else bool(harness.get("pass")),
        "harness_sha256": (
            sha256_file(out_dir / "HARNESS.json")
            if (out_dir / "HARNESS.json").is_file()
            else None
        ),
        "checkpoint_inventory_sha256": (
            sha256_file(inventory_path)
            if inventory_path.is_file()
            else None
        ),
        "checkpoint_inventory_status": (
            "resolved" if checkpoint_manifest.is_file() else "unresolved"
        ),
        "code_path": str(code_path.resolve()),
        "code_sha256": sha256_file(code_path),
    }
    protocol["protocol_sha256"] = sha256_json(
        {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    )
    return protocol


def write_protocol(out_dir: Path, harness: dict[str, Any] | None) -> dict[str, Any]:
    checkpoint_manifest = out_dir / "CHECKPOINTS.json"
    if not checkpoint_manifest.is_file():
        raise RuntimeError("cannot freeze protocol before all 18 checkpoints resolve")
    if harness is None or not harness.get("pass"):
        raise RuntimeError("cannot freeze protocol before the analytic harness passes")
    protocol = protocol_document(out_dir, harness, frozen=True)
    path = out_dir / "FROZEN_PROTOCOL.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(
                "FROZEN_PROTOCOL.json already exists and differs; refusing overwrite"
            )
        return existing
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        f"[protocol] froze FROZEN_PROTOCOL.json "
        f"sha={protocol['protocol_sha256'][:12]}",
        flush=True,
    )
    return protocol


def write_draft_protocol(
    out_dir: Path, harness: dict[str, Any] | None
) -> dict[str, Any]:
    protocol = protocol_document(out_dir, harness, frozen=False)
    path = out_dir / "PROTOCOL_DRAFT.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        f"[protocol] wrote draft only sha={protocol['protocol_sha256'][:12]}",
        flush=True,
    )
    return protocol


# ---------------------------------------------------------------------------
# Learned-critic audit
# ---------------------------------------------------------------------------


def make_critic_field(
    critic_params: Any, states: np.ndarray, c_ref: float, action_dim: int
) -> Callable[[np.ndarray], np.ndarray]:
    critic = TwinCritic()
    states_j = jnp.asarray(states, dtype=jnp.float64)

    @jax.jit
    def grad_q(actions):
        actions = jnp.asarray(actions, dtype=jnp.float64)

        def q_sum(a):
            q1, _ = critic.apply(critic_params, states_j, a)
            return jnp.sum(jnp.squeeze(q1, axis=-1))

        return jax.grad(q_sum)(actions)

    def f(actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        g = np.asarray(grad_q(actions), dtype=np.float64)
        return (action_dim * g) / c_ref

    return f


def compute_c_ref(critic_params: Any, states: np.ndarray, actions: np.ndarray) -> float:
    critic = TwinCritic()
    states_j = jnp.asarray(states, dtype=jnp.float64)
    actions_j = jnp.asarray(actions, dtype=jnp.float64)
    q1, _ = critic.apply(critic_params, states_j, actions_j)
    q = np.asarray(jnp.squeeze(q1, axis=-1), dtype=np.float64)
    return float(np.mean(np.abs(q)) + EPS)


def run_audit(out_dir: Path, data_dir: Path, max_runs: int) -> dict[str, Any]:
    ckpt_path = out_dir / "CHECKPOINTS.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"{ckpt_path} missing; resolve checkpoints on a host that has them"
        )
    doc = json.loads(ckpt_path.read_text(encoding="utf-8"))
    entries = [e for e in doc["entries"] if e["resolved"]]
    if len(entries) != 18:
        raise RuntimeError(f"need 18 resolved critics, found {len(entries)}")
    harness = json.loads((out_dir / "HARNESS.json").read_text(encoding="utf-8"))
    if not harness.get("pass"):
        raise RuntimeError("analytic harness did not pass")
    protocol_path = out_dir / "FROZEN_PROTOCOL.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("FROZEN_PROTOCOL.json must be frozen before audit")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    code_path = ROOT / "scripts" / "diagnostics" / "run_fixed_operator_order.py"
    if protocol["checkpoint_inventory_sha256"] != sha256_file(ckpt_path):
        raise RuntimeError("checkpoint inventory changed after protocol freeze")
    if protocol["code_sha256"] != sha256_file(code_path):
        raise RuntimeError("audit code changed after protocol freeze")
    max_action_dim = 6
    for run_index, entry in enumerate(entries):
        if sha256_file(Path(entry["checkpoint_path"])) != entry["weights_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {entry['checkpoint_path']}")
        idx_path = Path(entry["state_indices_path"])
        indices = np.load(idx_path)
        if sha256_bytes(np.asarray(indices, dtype=np.int64).tobytes()) != entry[
            "state_indices_sha256"
        ]:
            raise RuntimeError(f"state-index hash mismatch: {idx_path}")
    if max_runs > 0:
        entries = entries[:max_runs]

    endpoint_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    masks_meta: list[dict[str, Any]] = []
    realized_masks: list[np.ndarray] = []
    per_t_slopes: list[dict[str, Any]] = []
    raw_parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "cell_index",
            "state_index",
            "error_sq_per_dim",
            "finite",
            "oob",
            "implicit_converged",
            "common_mask",
        )
    }
    implicit_parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "run_index",
            "T_index",
            "K",
            "substep",
            "state_index",
            "residual",
            "delta",
            "iterations",
            "converged",
            "oob",
            "nonfinite",
            "action_dim",
            "action",
        )
    }
    path_parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "run_index",
            "T_index",
            "scheme",
            "K",
            "substep",
            "state_index",
            "action_dim",
            "action",
            "oob",
            "nonfinite",
        )
    }

    for entry in entries:
        env_name = entry["environment"]
        seed = entry["seed"]
        print(f"[audit] {env_name} seed={seed}", flush=True)
        data, mean, std = load_transition(env_name, data_dir, normalize=True)
        idx = np.load(entry["state_indices_path"])
        states = np.asarray(data.observations)[idx].astype(np.float64)
        anchors = np.asarray(data.actions)[idx].astype(np.float64)
        payload = load_checkpoint(Path(entry["checkpoint_path"]))
        critic_params = tree_params(payload, "critic_params")
        c_ref = compute_c_ref(critic_params, states, anchors)
        d = int(anchors.shape[-1])
        f = make_critic_field(critic_params, states, c_ref, d)

        complete_T = 0
        slopes_by_scheme: dict[str, list[float]] = {"explicit": [], "implicit": []}
        for t_index, t in enumerate(TOTAL_TIMES):
            ref = accept_rk4_reference(f, anchors, t)
            reference_stable = bool(ref["stable"] and ref["endpoint"] is not None)
            a_ref = (
                np.asarray(ref["endpoint"], dtype=np.float64)
                if ref["endpoint"] is not None
                else np.full_like(anchors, np.nan)
            )
            # eligibility buffers per (scheme,K)
            finite_ok = np.full(N_STATES, reference_stable, dtype=bool)
            finite_ok &= np.all(np.isfinite(a_ref), axis=-1)
            reference_oob = (
                np.asarray(ref["oob"], dtype=bool)
                if ref["oob"] is not None
                else np.ones(N_STATES, dtype=bool)
            )
            finite_ok &= ~reference_oob

            scheme_data: dict[tuple[str, int], dict[str, Any]] = {}
            for scheme in ("explicit", "implicit"):
                for k in SUBSTEPS:
                    dt = t / k
                    if scheme == "explicit":
                        end, oob, trace = explicit_euler(f, anchors, dt, k)
                        conv = np.ones(N_STATES, dtype=bool)
                        resid = np.zeros(N_STATES)
                        iters = np.ones(N_STATES, dtype=np.int32)
                    else:
                        end, oob, conv, resid, iters, trace = backward_euler(
                            f, anchors, dt, k
                        )
                    finite = np.all(np.isfinite(end), axis=-1)
                    scheme_data[(scheme, k)] = {
                        "end": end,
                        "oob": oob,
                        "conv": conv,
                        "resid": resid,
                        "iters": iters,
                        "finite": finite,
                        "trace": trace,
                    }

                    n_trace = k * N_STATES
                    padded = np.full(
                        (n_trace, max_action_dim), np.nan, dtype=np.float64
                    )
                    padded[:, :d] = trace["actions"].reshape(n_trace, d)
                    path_parts["run_index"].append(
                        np.full(n_trace, run_index, dtype=np.int16)
                    )
                    path_parts["T_index"].append(
                        np.full(n_trace, t_index, dtype=np.int8)
                    )
                    path_parts["scheme"].append(
                        np.full(
                            n_trace,
                            0 if scheme == "explicit" else 1,
                            dtype=np.int8,
                        )
                    )
                    path_parts["K"].append(np.full(n_trace, k, dtype=np.int8))
                    path_parts["substep"].append(
                        np.repeat(np.arange(1, k + 1, dtype=np.int8), N_STATES)
                    )
                    path_parts["state_index"].append(np.tile(idx, k))
                    path_parts["action_dim"].append(
                        np.full(n_trace, d, dtype=np.int8)
                    )
                    path_parts["action"].append(padded)
                    path_parts["oob"].append(trace["oob"].reshape(-1))
                    path_parts["nonfinite"].append(
                        trace["nonfinite"].reshape(-1)
                    )

                    if scheme == "implicit":
                        for key, values in (
                            ("run_index", np.full(n_trace, run_index, dtype=np.int16)),
                            ("T_index", np.full(n_trace, t_index, dtype=np.int8)),
                            ("K", np.full(n_trace, k, dtype=np.int8)),
                            (
                                "substep",
                                np.repeat(
                                    np.arange(1, k + 1, dtype=np.int8),
                                    N_STATES,
                                ),
                            ),
                            ("state_index", np.tile(idx, k)),
                            ("residual", trace["residual"].reshape(-1)),
                            ("delta", trace["delta"].reshape(-1)),
                            ("iterations", trace["iterations"].reshape(-1)),
                            ("converged", trace["converged"].reshape(-1)),
                            ("oob", trace["oob"].reshape(-1)),
                            ("nonfinite", trace["nonfinite"].reshape(-1)),
                            ("action_dim", np.full(n_trace, d, dtype=np.int8)),
                            ("action", padded),
                        ):
                            implicit_parts[key].append(values)

            mask = finite_ok.copy()
            for pack in scheme_data.values():
                mask &= pack["finite"]
                mask &= pack["conv"]
                mask &= ~pack["oob"]
            n_mask = int(np.sum(mask))
            mask_sha = sha256_bytes(np.packbits(mask).tobytes())
            realized_masks.append(mask.copy())
            explicit_nonfinite = np.logical_or.reduce(
                [
                    ~scheme_data[("explicit", k)]["finite"]
                    for k in SUBSTEPS
                ]
            )
            explicit_oob = np.logical_or.reduce(
                [scheme_data[("explicit", k)]["oob"] for k in SUBSTEPS]
            )
            implicit_nonfinite = np.logical_or.reduce(
                [
                    ~scheme_data[("implicit", k)]["finite"]
                    for k in SUBSTEPS
                ]
            )
            implicit_oob = np.logical_or.reduce(
                [scheme_data[("implicit", k)]["oob"] for k in SUBSTEPS]
            )
            implicit_failed = np.logical_or.reduce(
                [~scheme_data[("implicit", k)]["conv"] for k in SUBSTEPS]
            )
            masks_meta.append(
                {
                    "environment": env_name,
                    "seed": seed,
                    "T": t,
                    "reference_unstable": not reference_stable,
                    "|I|": n_mask,
                    "mask_sha256": mask_sha,
                    "rk4_n_micro": ref["n_micro"],
                    "rk4_R_N": ref["R_N"],
                    "rk4_M_2N": ref["M_2N"],
                    "C_ref": c_ref,
                    "exclusion_counts": {
                        "reference_nonfinite": int(
                            np.sum(~np.all(np.isfinite(a_ref), axis=-1))
                        ),
                        "reference_oob": int(np.sum(reference_oob)),
                        "explicit_nonfinite": int(np.sum(explicit_nonfinite)),
                        "explicit_oob": int(np.sum(explicit_oob)),
                        "implicit_nonfinite": int(np.sum(implicit_nonfinite)),
                        "implicit_oob": int(np.sum(implicit_oob)),
                        "implicit_not_converged": int(np.sum(implicit_failed)),
                    },
                }
            )
            primary = reference_stable and n_mask >= MIN_MASK
            if primary:
                complete_T += 1

            m_ref = (
                float(np.sqrt(np.mean(np.square(a_ref[mask] - anchors[mask]))))
                if n_mask
                else float("nan")
            )
            for scheme in ("explicit", "implicit"):
                e_by_k: dict[int, float] = {}
                state_error_by_k: dict[int, np.ndarray] = {}
                for k in SUBSTEPS:
                    end = scheme_data[(scheme, k)]["end"]
                    if n_mask:
                        e_abs = float(np.sqrt(np.mean(np.square(end[mask] - a_ref[mask]))))
                    else:
                        e_abs = float("nan")
                    e_by_k[k] = e_abs
                    state_error_by_k[k] = np.mean(
                        np.square(end - a_ref), axis=-1
                    )

                fitted = [e_by_k[k] for k in (4, 8, 16)]
                if not reference_stable:
                    slope = None
                    slope_status = "reference_unstable"
                elif not primary:
                    slope = None
                    slope_status = "mask_below_410"
                elif any(np.isfinite(value) and value < 1e-12 for value in fitted):
                    slope = None
                    slope_status = "below_floor"
                elif not all(np.isfinite(value) for value in fitted):
                    slope = None
                    slope_status = "nonfinite_error"
                else:
                    slope = fit_log_slope(
                        [t / k for k in (4, 8, 16)], fitted
                    )
                    slope_status = "ok" if slope is not None else "fit_failed"
                if slope_status == "ok" and slope is not None:
                    slopes_by_scheme[scheme].append(slope)
                per_t_slopes.append(
                    {
                        "environment": env_name,
                        "seed": seed,
                        "T": t,
                        "scheme": scheme,
                        "slope_K4_8_16": slope,
                        "slope_status": slope_status,
                    }
                )

                for k in SUBSTEPS:
                    e_abs = e_by_k[k]
                    e_rel = e_abs / max(m_ref, 1e-8) if np.isfinite(e_abs) else float("nan")
                    doubled = e_by_k.get(2 * k)
                    ratio = (
                        e_abs / doubled
                        if k in (1, 2, 4, 8)
                        and doubled is not None
                        and np.isfinite(e_abs)
                        and np.isfinite(doubled)
                        and doubled > 0
                        else None
                    )
                    failure_reasons = []
                    if not reference_stable:
                        failure_reasons.append("reference_unstable")
                    if n_mask < MIN_MASK:
                        failure_reasons.append("mask_below_410")
                    cell_index = len(endpoint_rows)
                    endpoint_rows.append(
                        {
                            "environment": env_name,
                            "seed": seed,
                            "T": t,
                            "K": k,
                            "scheme": scheme,
                            "E_abs": e_abs,
                            "E_rel": e_rel,
                            "M_ref": m_ref,
                            "ratio_to_2K": ratio,
                            "slope_K4_8_16": slope,
                            "slope_status": slope_status,
                            "|I|": n_mask,
                            "primary_cell": primary,
                            "reference_stable": reference_stable,
                            "failure_reason": ";".join(failure_reasons),
                            "C_ref": c_ref,
                        }
                    )
                    pack = scheme_data[(scheme, k)]
                    raw_parts["cell_index"].append(
                        np.full(N_STATES, cell_index, dtype=np.int16)
                    )
                    raw_parts["state_index"].append(idx.copy())
                    raw_parts["error_sq_per_dim"].append(
                        state_error_by_k[k]
                    )
                    raw_parts["finite"].append(pack["finite"].copy())
                    raw_parts["oob"].append(pack["oob"].copy())
                    raw_parts["implicit_converged"].append(
                        pack["conv"].copy()
                    )
                    raw_parts["common_mask"].append(mask.copy())

        run_sum = {
            "environment": env_name,
            "seed": seed,
            "complete": complete_T >= 3,
            "n_eligible_T": complete_T,
            "median_slope_explicit": (
                float(np.median(slopes_by_scheme["explicit"]))
                if slopes_by_scheme["explicit"]
                else None
            ),
            "median_slope_implicit": (
                float(np.median(slopes_by_scheme["implicit"]))
                if slopes_by_scheme["implicit"]
                else None
            ),
            "C_ref": c_ref,
            "checkpoint": entry["checkpoint_path"],
        }
        run_summaries.append(run_sum)
        del data

    import csv

    expected_endpoint = len(entries) * len(TOTAL_TIMES) * len(SUBSTEPS) * 2
    expected_raw = expected_endpoint * N_STATES
    expected_implicit = (
        len(entries) * len(TOTAL_TIMES) * N_STATES * sum(SUBSTEPS)
    )
    expected_paths = 2 * expected_implicit
    endpoint_keys = {
        (
            row["environment"],
            int(row["seed"]),
            float(row["T"]),
            int(row["K"]),
            row["scheme"],
        )
        for row in endpoint_rows
    }
    if len(endpoint_rows) != expected_endpoint or len(endpoint_keys) != expected_endpoint:
        raise AssertionError(
            f"endpoint key/count failure: rows={len(endpoint_rows)} "
            f"unique={len(endpoint_keys)} expected={expected_endpoint}"
        )

    def concatenate(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        if any(not values for values in parts.values()):
            missing_keys = [key for key, values in parts.items() if not values]
            raise AssertionError(f"missing diagnostic arrays: {missing_keys}")
        return {key: np.concatenate(values, axis=0) for key, values in parts.items()}

    raw_arrays = concatenate(raw_parts)
    implicit_arrays = concatenate(implicit_parts)
    path_arrays = concatenate(path_parts)
    if len(raw_arrays["cell_index"]) != expected_raw:
        raise AssertionError(
            f"raw-state rows={len(raw_arrays['cell_index'])}, expected={expected_raw}"
        )
    if len(implicit_arrays["run_index"]) != expected_implicit:
        raise AssertionError(
            f"implicit rows={len(implicit_arrays['run_index'])}, "
            f"expected={expected_implicit}"
        )
    if len(path_arrays["run_index"]) != expected_paths:
        raise AssertionError(
            f"path rows={len(path_arrays['run_index'])}, expected={expected_paths}"
        )

    csv_path = out_dir / "endpoint_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(endpoint_rows[0].keys()))
        writer.writeheader()
        writer.writerows(endpoint_rows)
    (out_dir / "masks.json").write_text(json.dumps(masks_meta, indent=2) + "\n")
    masks_path = out_dir / "masks.npz"
    np.savez_compressed(
        masks_path,
        common_mask=np.stack(realized_masks),
        run_index=np.repeat(
            np.arange(len(entries), dtype=np.int16), len(TOTAL_TIMES)
        ),
        T_index=np.tile(
            np.arange(len(TOTAL_TIMES), dtype=np.int8), len(entries)
        ),
    )
    (out_dir / "per_t_slopes.json").write_text(
        json.dumps(per_t_slopes, indent=2) + "\n"
    )
    raw_path = out_dir / "raw_state_diagnostics.npz"
    implicit_path = out_dir / "implicit_substeps.npz"
    paths_path = out_dir / "euler_paths.npz"
    np.savez_compressed(raw_path, **raw_arrays)
    np.savez_compressed(implicit_path, **implicit_arrays)
    np.savez_compressed(paths_path, **path_arrays)

    def scheme_gate(scheme: str) -> dict[str, Any]:
        slopes = [
            r[f"median_slope_{scheme}"]
            for r in run_summaries
            if r[f"median_slope_{scheme}"] is not None
        ]
        complete = sum(1 for r in run_summaries if r["complete"])
        in_band = sum(1 for s in slopes if 0.5 <= s <= 1.5)
        agg = float(np.median(slopes)) if slopes else None
        support = (
            agg is not None
            and 0.75 <= agg <= 1.25
            and in_band >= 14
            and complete >= 15
        )
        task_means = {}
        for env_name in ENVIRONMENTS:
            values = [
                row[f"median_slope_{scheme}"]
                for row in run_summaries
                if row["environment"] == env_name
                and row[f"median_slope_{scheme}"] is not None
            ]
            if len(values) == 2:
                task_means[env_name] = float(np.mean(values))
        task_equal = (
            float(np.mean(list(task_means.values())))
            if len(task_means) == len(ENVIRONMENTS)
            else None
        )
        return {
            "aggregate_median_slope": agg,
            "n_slopes": len(slopes),
            "n_in_[0.5,1.5]": in_band,
            "n_complete_runs": complete,
            "two_seed_task_means": task_means,
            "task_equal_mean": task_equal,
            "pre_specified_support": support,
        }

    summary = {
        "n_runs": len(run_summaries),
        "n_endpoint_rows": len(endpoint_rows),
        "n_unique_endpoint_keys": len(endpoint_keys),
        "expected_endpoint_rows": expected_endpoint,
        "n_raw_state_rows": len(raw_arrays["cell_index"]),
        "expected_raw_state_rows": expected_raw,
        "n_implicit_substep_records": len(implicit_arrays["run_index"]),
        "expected_implicit_substep_records": expected_implicit,
        "n_euler_path_records": len(path_arrays["run_index"]),
        "expected_euler_path_records": expected_paths,
        "explicit": scheme_gate("explicit"),
        "implicit": scheme_gate("implicit"),
        "runs": run_summaries,
        "csv": str(csv_path.resolve()),
        "artifacts": {
            "endpoint_errors.csv": sha256_file(csv_path),
            "masks.json": sha256_file(out_dir / "masks.json"),
            "masks.npz": sha256_file(masks_path),
            "per_t_slopes.json": sha256_file(out_dir / "per_t_slopes.json"),
            "raw_state_diagnostics.npz": sha256_file(raw_path),
            "implicit_substeps.npz": sha256_file(implicit_path),
            "euler_paths.npz": sha256_file(paths_path),
        },
        "written_at": now_iso(),
    }
    summary_path = out_dir / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    result_manifest = {
        "experiment": "fixed_operator_order_v2",
        "protocol_sha256": protocol["protocol_sha256"],
        "checkpoint_inventory_sha256": sha256_file(ckpt_path),
        "code_sha256": sha256_file(code_path),
        "counts": {
            "endpoint_rows": len(endpoint_rows),
            "raw_state_rows": len(raw_arrays["cell_index"]),
            "implicit_substep_records": len(implicit_arrays["run_index"]),
            "euler_path_records": len(path_arrays["run_index"]),
        },
        "realized_masks": masks_meta,
        "artifacts": {
            **summary["artifacts"],
            "SUMMARY.json": sha256_file(summary_path),
        },
        "written_at": now_iso(),
    }
    (out_dir / "RESULT_MANIFEST.json").write_text(
        json.dumps(result_manifest, indent=2) + "\n"
    )
    print(json.dumps({k: summary[k] for k in ("n_runs", "explicit", "implicit")}, indent=2))
    return summary


def write_readme(out_dir: Path) -> None:
    text = """# Fixed-operator order audit

Replacement for the archived actor-path semigroup defect tables as manuscript
evidence for discretization / semigroup-error claims. Spec: repo `TODO.md` P0-A.

## Artifacts

| File | Role |
| --- | --- |
| `HARNESS.json` | Analytic d=2 validation |
| `PROTOCOL_DRAFT.json` | Pre-checkpoint draft formulas, schemas, gates |
| `FROZEN_PROTOCOL.json` | Written only after all 18 fingerprints resolve |
| `CHECKPOINTS.json` / `CHECKPOINTS_UNRESOLVED.json` | 18 T=1 critic fingerprints |
| `state_indices/*.npy` | Frozen 512-row index sets |
| `endpoint_errors.csv` | Compact 720-cell endpoint table (after audit) |
| `raw_state_diagnostics.npz` | 368,640 pre-mask state records |
| `implicit_substeps.npz` | 1,142,784 implicit state/substep diagnostics |
| `euler_paths.npz` | Unprojected explicit/implicit paths |
| `SUMMARY.json` | Run slopes and scientific gates |
| `RESULT_MANIFEST.json` | Realized mask and artifact hashes |
| `STATUS.json` | Host readiness / blockers |

## Run

```bash
cd /path/to/MPI_sweep
python scripts/diagnostics/run_fixed_operator_order.py --phase all \\
  --checkpoint-roots /path/to/results_qnorm
```

On ext_csh the 18 critics are currently unresolved (owned by ext_csv
`results_qnorm`). Harness + protocol draft + state indices can still be
produced. The script refuses to freeze or run the learned audit until every
checkpoint/config/weight fingerprint resolves.
"""
    (out_dir / "README.md").write_text(text)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_readme(out_dir)
    harness = None
    phase = args.phase

    if phase in ("harness", "all"):
        harness = run_harness(out_dir)
    elif (out_dir / "HARNESS.json").is_file():
        harness = json.loads((out_dir / "HARNESS.json").read_text())

    if phase in ("resolve", "all"):
        resolve_checkpoints(out_dir, args.data_dir, args.checkpoint_roots)

    if phase in ("protocol", "all"):
        if (out_dir / "CHECKPOINTS.json").is_file():
            write_protocol(out_dir, harness)
        else:
            write_draft_protocol(out_dir, harness)

    if phase == "audit" or (
        phase == "all" and (out_dir / "CHECKPOINTS.json").is_file()
    ):
        run_audit(out_dir, args.data_dir, args.max_runs)
    elif phase == "all":
        print(
            "[audit] skipped — checkpoints unresolved on this host",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
