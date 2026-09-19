"""Train/evaluate the shared-IQL-critic actor/geometry comparison in JAX."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from functools import partial
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

import provenance
from d4rl_data import DATASET_FILES, dataset_path, download_dataset
from iql_mpi import create_state, policy_stats, update_many
from iql_mpi_config import (SCHEMA, add_iql_args, canonical, config_from_args,
                            is_complete, run_name, run_signature, signature_hash)
from train_td3bc import (Transition, evaluate, install_stop_handler, latest_checkpoint,
                         qlearning_from_hdf5)

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ("iql_mpi.py", "iql_mpi_config.py", "train_iql_mpi.py", "train_td3bc.py", "d4rl_data.py")
EVAL_FIELDS = ("step", "variant", "policy", "hop", "K", "T", "h", "time",
               "eval_mode", "return", "d4rl_score")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=1.0, help="Total horizon T, including the first dataset-anchored step")
    parser.add_argument("--mpi-steps", type=int, default=4, help="Total steps K >= 1; K=1 is the base algorithm at T")
    parser.add_argument("--polyak", type=float, default=0.005)
    parser.add_argument("--max-timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-freq", type=int, default=1_000_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--updates-per-dispatch", type=int, default=64)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "results" / "iql")
    parser.add_argument("--compilation-cache-dir", type=Path, default=Path.home() / ".cache/mpi-sweep/jax")
    add_iql_args(parser)
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args):
    config_from_args(args)
    if args.env not in DATASET_FILES:
        raise ValueError(f"unsupported environment {args.env}")
    for name in ("batch_size", "max_timesteps", "eval_freq", "eval_episodes", "updates_per_dispatch"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.seed < 0 or args.save_interval < 0:
        raise ValueError("seed and save_interval must be nonnegative")
    if not math.isfinite(args.reward_scale) or args.reward_scale <= 0:
        raise ValueError("reward_scale must be finite and positive")


def normalize_iql_rewards(raw, mode="iql", reward_scale=1.0):
    """Official IQL / Park et al. locomotion return-range normalization.

    Use unnormalized observations to find skipped-timeout boundaries; terminal
    masks for Bellman targets remain unchanged. Include the last trajectory.
    """
    factor, return_range = 1.0, None
    if mode == "iql":
        boundary = np.asarray(raw["not_dones"]).reshape(-1) == 0
        boundary[:-1] |= np.linalg.norm(
            raw["observations"][1:] - raw["next_observations"][:-1], axis=-1) > 1e-6
        boundary[-1] = True
        starts = np.r_[0, np.flatnonzero(boundary)[:-1] + 1]
        returns = np.add.reduceat(raw["rewards"].reshape(-1).astype(np.float64), starts)
        return_range = float(np.max(returns) - np.min(returns))
        if not np.isfinite(return_range) or return_range <= 0:
            raise ValueError("IQL reward normalization needs a positive trajectory-return range; "
                             "use --iql-reward-normalization none for a deliberate raw-reward experiment")
        factor = 1000.0 / return_range
    elif mode != "none":
        raise ValueError(f"unknown reward normalization {mode}")
    rewards = (raw["rewards"] * (factor * reward_scale)).astype(np.float32)
    return rewards, {"mode": mode, "return_range": return_range, "factor": factor,
                     "extra_reward_scale": reward_scale}


def load_iql_transition(args):
    raw = qlearning_from_hdf5(download_dataset(args.env, args.data_dir))
    raw["rewards"], reward_info = normalize_iql_rewards(
        raw, args.iql_reward_normalization, args.reward_scale)
    # The official IQL and bottleneck loaders also clip dataset actions by eps.
    if np.max(np.abs(raw["actions"])) > 1.0 + 1e-5:
        raise ValueError("this locomotion protocol expects actions in [-1, 1]")
    raw["actions"] = np.clip(raw["actions"], -1. + 1e-5, 1. - 1e-5)
    mean = np.zeros(raw["observations"].shape[-1], dtype=np.float32)
    std = np.ones_like(mean)
    if args.iql_normalize_state:
        mean = raw["observations"].mean(axis=0)
        std = raw["observations"].std(axis=0) + 1e-3
        raw["observations"] = (raw["observations"] - mean) / std
        raw["next_observations"] = (raw["next_observations"] - mean) / std
    return Transition(**{k: jnp.asarray(v) for k, v in raw.items()}), mean, std, reward_info


def resolved_protocol(config, action_dim):
    return {"T": config.tau, "K": config.mpi_steps, "h": config.h,
            "chain_times": [k * config.h for k in range(1, config.mpi_steps + 1)],
            "q_scale_source": "pre-update full-T baseline; stopped and shared across the chain",
            "extra_baseline_updates_per_variant": int(config.mpi_steps > 1),
            "chain_actor_updates_per_variant": 1 + (config.mpi_steps - 1) * config.inner_updates,
            "variants": {v: {"metric_reduction": config.reduction(v),
                              "q_scale_norm": config.normalize_q(v),
                              "first_step_coefficients": config.coefficients(v, action_dim),
                              "baseline_coefficients": config.coefficients(v, action_dim, config.tau)}
                         for v in config.variants}}


def save_checkpoint(path, state, rng, step, mean, std, signature, sources, data_hash):
    payload = {"schema": SCHEMA, "step": int(step), "signature": signature,
               "sources": sources, "dataset_sha256": data_hash,
               "state": jax.device_get(flax.serialization.to_state_dict(state)),
               "rng": np.asarray(rng), "mean": np.asarray(mean), "std": np.asarray(std)}
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def restore_checkpoint(path, state, signature, sources, data_hash):
    # This entrypoint reads only its own run directory's trusted checkpoints.
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("schema") != SCHEMA or canonical(payload.get("signature")) != canonical(signature):
        raise ValueError("checkpoint algorithm/geometry/training/evaluation configuration mismatch")
    if payload.get("sources") != sources or payload.get("dataset_sha256") != data_hash:
        raise ValueError("checkpoint source or dataset changed; use a new save directory")
    state = flax.serialization.from_state_dict(state, payload["state"])
    state = jax.tree_util.tree_map(jnp.asarray, state)
    return state, jnp.asarray(payload["rng"]), int(payload["step"]), payload["mean"], payload["std"]


def evaluation_specs(config, args):
    policies = [("baseline", 1)]
    if config.mpi_steps > 1:
        hops = range(1, config.mpi_steps + 1) if args.eval_hops == "all" else (config.mpi_steps,)
        policies.extend(("mpi", hop) for hop in hops)
    for index, variant in enumerate(config.variants):
        modes = ("mean",) if variant == "qbc_deterministic_w2" else (
            ("mean", "sample") if args.eval_mode == "both" else (args.eval_mode,))
        for policy, hop in policies:
            for mode in modes:
                yield index, variant, policy, hop, mode


def evaluate_all(state, config, args, step, mean, std):
    rows = []
    for index, variant, policy_name, hop, mode in evaluation_specs(config, args):
        actor = state.baselines[index] if policy_name == "baseline" else state.actors[index][hop-1]
        k, h = (1, config.tau) if policy_name == "baseline" else (config.mpi_steps, config.h)
        gaussian = variant != "qbc_deterministic_w2"
        # Separate evaluation RNG: changing evaluation frequency cannot alter training.
        # Same standard-normal stream across Gaussian branches/hops at this step.
        eval_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), step)
        def action_fn(params, observation, key):
            m, sigma = policy_stats(actor, params, observation[None], gaussian)
            action = m[0]
            if mode == "sample":
                action = action + sigma[0] * jax.random.normal(key, action.shape)
            return jnp.clip(action, -1.0, 1.0)
        action_fn = jax.jit(action_fn)
        def policy(observation):
            nonlocal eval_key
            eval_key, key = jax.random.split(eval_key)
            return action_fn(actor.params, jnp.asarray(observation), key)
        ret, score = evaluate(policy, args.env, args.seed, mean, std, args.eval_episodes)
        if not np.isfinite([ret, score]).all():
            raise FloatingPointError(f"nonfinite evaluation for {variant}/{hop}/{mode}")
        rows.append(dict(zip(EVAL_FIELDS, (step, variant, policy_name, hop, k, config.tau,
                                          h, hop*h, mode, ret, score), strict=True)))
    return rows


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def record_evaluation(out_dir, state, config, args, step, mean, std):
    rows = [row for row in read_rows(out_dir / "eval.csv") if int(row["step"]) < step]
    new_rows = evaluate_all(state, config, args, step, mean, std)
    write_rows(out_dir / "eval.csv", rows + new_rows)
    for row in new_rows:
        print(f"[eval] step={step} {row['variant']} {row['policy']} hop={row['hop']} time={row['time']:g} "
              f"mode={row['eval_mode']} d4rl={row['d4rl_score']:.2f}", flush=True)


def mark_complete(out_dir, config, args, signature):
    expected = {(v, p, str(h), m) for _, v, p, h, m in evaluation_specs(config, args)}
    final_rows = [r for r in read_rows(out_dir / "eval.csv") if int(r["step"]) == args.max_timesteps]
    actual = {(r["variant"], r["policy"], r["hop"], r["eval_mode"]) for r in final_rows}
    if actual != expected or len(final_rows) != len(expected):
        raise ValueError("incomplete or duplicate final actor evaluations")
    checkpoint = out_dir / f"params_{args.max_timesteps}.pkl"
    if not checkpoint.is_file():
        raise ValueError("final checkpoint missing")
    provenance.write_json(out_dir / "COMPLETE.json", {
        "schema": SCHEMA, "step": args.max_timesteps, "signature": signature_hash(signature),
        "checkpoint_sha256": provenance.sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "eval_sha256": provenance.sha256_file(out_dir / "eval.csv"), "evaluation_rows": len(final_rows)})


def main(argv=None):
    args = parse_args(argv)
    config = config_from_args(args)
    signature = run_signature(args, config)
    out_dir = args.save_dir / run_name(args, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: provenance.sha256_file(ROOT / name) for name in SOURCE_NAMES}
    config_path = out_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if canonical(existing["signature"]) != canonical(signature) or existing["sources"] != sources:
            raise ValueError("existing run has different configuration or source; use a new save directory")
    elif any(out_dir.iterdir()):
        raise ValueError("nonempty run directory without config; use a new save directory")
    if not config_path.exists():
        provenance.write_json(config_path, {"signature": signature, "sources": sources, "args": vars(args)})
    if is_complete(out_dir, args.max_timesteps):
        print(f"[complete] {out_dir}")
        return 0
    (out_dir / "COMPLETE.json").unlink(missing_ok=True)
    args.compilation_cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache_dir))
    stop = install_stop_handler(out_dir)
    data, mean, std, reward_info = load_iql_transition(args)
    protocol = resolved_protocol(config, data.actions.shape[-1])
    provenance.write_json(config_path, {"signature": signature, "sources": sources,
                                       "args": vars(args), "resolved_protocol": protocol,
                                       "reward_normalization": reward_info})
    if float(jnp.max(jnp.abs(data.actions))) > 1.0 + 1e-5:
        raise ValueError("this locomotion protocol expects actions in [-1, 1]")
    data_hash = provenance.sha256_file(dataset_path(args.env, args.data_dir))
    rng, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
    state = create_state(init_key, data.observations[:1], data.actions[:1], config)
    step = 0
    checkpoint = latest_checkpoint(out_dir)
    if checkpoint is not None:
        state, rng, step, stored_mean, stored_std = restore_checkpoint(
            checkpoint, state, signature, sources, data_hash)
        if not np.array_equal(mean, stored_mean) or not np.array_equal(std, stored_std):
            raise ValueError("dataset normalization changed since checkpoint")
        if step > args.max_timesteps:
            raise ValueError("checkpoint is beyond requested max_timesteps")
        print(f"[resume] {checkpoint} step={step}", flush=True)
    # Trim uncheckpointed log rows after a crash; preserve valid earlier evaluations.
    write_rows(out_dir / "eval.csv", [r for r in read_rows(out_dir / "eval.csv") if int(r["step"]) <= step])
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        retained = []
        for line in metrics_path.read_text().splitlines():
            try:
                if json.loads(line)["step"] <= step:
                    retained.append(line)
            except (ValueError, KeyError):
                pass
        metrics_path.write_text("".join(line + "\n" for line in retained))
    if not (out_dir / "PROVENANCE.json").exists():
        provenance.write_json(out_dir / "PROVENANCE.json", provenance.base_provenance(
            ROOT, {name: ROOT / name for name in SOURCE_NAMES},
            extra={"signature": signature, "dataset_sha256": data_hash,
                   "dataset": provenance.dataset_identity(dataset_path(args.env, args.data_dir)),
                   "normalization": {"mean": mean.tolist(), "std": std.tolist()},
                   "reward_normalization": reward_info, "resolved_protocol": protocol,
                   "metric": "raw diagonal Gaussian; ambient FR^2 = 4*acos(BC)^2 (MPI A.3)",
                   "q_action_transform": config.q_action_transform,
                   "gaussian_qbc_base": "Park et al. 2024 Eq.6/C.6.1: -mean(Q(clipped mean)) + bc_coef*mean(NLL); raw Gaussian, sigma=1",
                   "deterministic_base": "TD3+BC actor loss: detached alpha/mean(abs(IQL Q)) and coordinate-mean MSE",
                   "gaussian_refinement": "MPI extension: sampled expected Q with learnable mean and std",
                   "evaluation_action_transform": "clip to [-1,1]"}))
    update_fns = {}
    while step < args.max_timesteps:
        count = min(args.updates_per_dispatch, args.max_timesteps - step,
                    args.eval_freq - step % args.eval_freq)
        if args.save_interval:
            count = min(count, args.save_interval - step % args.save_interval)
        if count not in update_fns:
            update_fns[count] = jax.jit(partial(update_many, config=config,
                                                batch_size=args.batch_size, count=count))
        state, rng, metrics = update_fns[count](state, data, rng)
        step += count
        metrics = {name: float(value) for name, value in metrics.items()}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"nonfinite training metric at step {step}")
        with metrics_path.open("a") as stream:
            stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
        evaluate_now = step % args.eval_freq == 0 or step == args.max_timesteps
        if evaluate_now or stop["flag"] or (args.save_interval and step % args.save_interval == 0):
            save_checkpoint(out_dir / f"params_{step}.pkl", state, rng, step, mean, std,
                            signature, sources, data_hash)
        if stop["flag"]:
            print(f"[stop] checkpoint saved at step {step}", flush=True)
            return 128 + int(stop["signum"] or 2)
        if evaluate_now:
            record_evaluation(out_dir, state, config, args, step, mean, std)
    # A crash during evaluation can leave a valid final checkpoint without rows.
    expected = {(v, p, str(h), m) for _, v, p, h, m in evaluation_specs(config, args)}
    actual = {(r["variant"], r["policy"], r["hop"], r["eval_mode"]) for r in read_rows(out_dir / "eval.csv")
              if int(r["step"]) == args.max_timesteps}
    if actual != expected:
        record_evaluation(out_dir, state, config, args, step, mean, std)
    mark_complete(out_dir, config, args, signature)
    print(f"[complete] {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
