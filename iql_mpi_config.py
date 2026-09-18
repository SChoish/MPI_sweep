"""Dependency-free CLI and run identity for the shared-critic IQL comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass

VARIANTS = ("awr_gaussian_fr", "qbc_deterministic_w2", "qbc_gaussian_w2")
SCHEMA = "iql_actor_geometry_v3_bottleneck_ddpgbc"


@dataclass(frozen=True)
class IQLConfig:
    variants: tuple[str, ...] = VARIANTS
    mpi_steps: int = 4
    tau: float = 1.0
    expectile: float = 0.7
    awr_beta: float = 3.0
    bc_coef: float = 1.0
    td3bc_alpha: float = 2.5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    discount: float = 0.99
    polyak: float = 0.005
    hidden_dims: tuple[int, ...] = (256, 256)
    log_std_init: float = -1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    mc_samples: int = 8
    inner_updates: int = 1
    metric_reduction: str = "sum"
    q_action_transform: str = "identity"
    iql_q_scale_norm: bool = False

    def __post_init__(self):
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must be nonempty and unique")
        if set(self.variants).difference(VARIANTS):
            raise ValueError(f"variants must be selected from {VARIANTS}")
        for name in ("tau", "awr_beta", "td3bc_alpha", "actor_lr", "critic_lr", "value_lr"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.bc_coef) or self.bc_coef < 0:
            raise ValueError("bc_coef must be finite and nonnegative")
        if not 0 < self.expectile < 1 or not 0 < self.polyak <= 1:
            raise ValueError("expectile must be in (0,1); polyak in (0,1]")
        if not 0 <= self.discount < 1:
            raise ValueError("discount must be in [0,1)")
        if self.mpi_steps < 0 or self.mc_samples < 1 or self.inner_updates < 1:
            raise ValueError("mpi_steps must be >=0; mc_samples and inner_updates >=1")
        if not self.hidden_dims or any(x < 1 for x in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not all(math.isfinite(x) for x in (self.log_std_min, self.log_std_init, self.log_std_max)):
            raise ValueError("log standard deviations must be finite")
        if not self.log_std_min < self.log_std_init < self.log_std_max:
            raise ValueError("require log_std_min < log_std_init < log_std_max")
        if "qbc_gaussian_w2" in self.variants and not self.log_std_min < 0 < self.log_std_max:
            raise ValueError("Gaussian DDPG+BC refinement bounds must contain log(sigma=1)=0")
        if self.metric_reduction not in ("sum", "mean"):
            raise ValueError("metric_reduction must be sum or mean")
        if self.q_action_transform not in ("identity", "clip"):
            raise ValueError("q_action_transform must be identity or clip")


def add_iql_args(parser):
    parser.add_argument("--variants", default=" ".join(VARIANTS))
    for name, default in (("expectile", 0.7), ("awr-beta", 3.0), ("bc-coef", 1.0), ("td3bc-alpha", 2.5),
                          ("actor-lr", 3e-4), ("critic-lr", 3e-4), ("value-lr", 3e-4),
                          ("discount", 0.99), ("log-std-init", -1.0),
                          ("log-std-min", -5.0), ("log-std-max", 2.0), ("reward-scale", 1.0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", default="256 256")
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--inner-updates", type=int, default=1)
    parser.add_argument("--metric-reduction", choices=("sum", "mean"), default="sum")
    parser.add_argument("--q-action-transform", choices=("identity", "clip"), default="identity",
                        help="MPI expected-Q action transform; DDPG+BC base always clips its mean.")
    parser.add_argument("--iql-q-scale-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--iql-normalize-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-mode", choices=("mean", "sample", "both"), default="both")
    parser.add_argument("--eval-hops", choices=("endpoints", "all"), default="endpoints")


def config_from_args(args, *, tau=None, hops=None):
    values = {name: getattr(args, name) for name in IQLConfig.__dataclass_fields__
              if hasattr(args, name)}
    selected = args.variants.replace(",", " ").split()
    if len(selected) != len(set(selected)) or set(selected).difference(VARIANTS):
        raise ValueError(f"select unique variants from {VARIANTS}")
    values["variants"] = tuple(v for v in VARIANTS if v in selected)
    values["hidden_dims"] = tuple(int(x) for x in args.hidden_dims.replace(",", " ").split())
    values["mpi_steps"] = args.mpi_steps if hops is None else hops
    values["tau"] = args.tau if tau is None else float(tau)
    return IQLConfig(**values)


def run_signature(args, config):
    return {"schema": SCHEMA, "algorithm": asdict(config), "env": args.env,
            "seed": args.seed, "batch_size": args.batch_size,
            "normalize_state": args.iql_normalize_state, "reward_scale": args.reward_scale,
            "eval_mode": args.eval_mode, "eval_hops": args.eval_hops,
            "eval_episodes": args.eval_episodes}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def signature_hash(signature):
    return hashlib.sha256(canonical(signature).encode()).hexdigest()[:12]


def run_name(args, config):
    return (f"{args.env}_iql_tau{config.tau:g}_mpi{config.mpi_steps}_seed{args.seed}_"
            f"{signature_hash(run_signature(args, config))}")


def is_complete(directory, max_timesteps):
    """A final eval row alone does not prove all variants/checkpoints completed."""
    try:
        done = json.loads((directory / "COMPLETE.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        checkpoint = directory / f"params_{max_timesteps}.pkl"
        return (done["schema"] == SCHEMA and done["step"] == max_timesteps
                and done["signature"] == signature_hash(config["signature"])
                and checkpoint.is_file() and checkpoint.stat().st_size > 0
                and checkpoint.stat().st_size == done["checkpoint_bytes"]
                and hashlib.sha256((directory / "eval.csv").read_bytes()).hexdigest() == done["eval_sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
