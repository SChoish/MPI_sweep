"""Dependency-free CLI and run identity for the shared-critic IQL comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass

VARIANTS = ("awr_gaussian_fr", "qbc_deterministic_w2", "qbc_gaussian_w2")
SCHEMA = "iql_actor_geometry_v4_total_horizon"
# DDPG+BC uses MART's small-T spacing over the requested [1/50, 5] range.
# Other families retain their existing defaults; these are experiment grids.
NATIVE_T_GRIDS = {"awr_gaussian_fr": (1., 3., 10.),
                  "qbc_gaussian_w2": (.02, .05, .1, .2, .4, .7, 1.5, 2.5, 4., 5.),
                  "qbc_deterministic_w2": (1.25,)}


@dataclass(frozen=True)
class IQLConfig:
    variants: tuple[str, ...] = VARIANTS
    mpi_steps: int = 4
    tau: float = 1.0
    expectile: float = 0.7
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
    metric_reduction: str = "native"
    q_action_transform: str = "identity"
    iql_q_scale_norm: bool | None = None

    def __post_init__(self):
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must be nonempty and unique")
        if set(self.variants).difference(VARIANTS):
            raise ValueError(f"variants must be selected from {VARIANTS}")
        for name in ("tau", "actor_lr", "critic_lr", "value_lr"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.expectile < 1 or not 0 < self.polyak <= 1:
            raise ValueError("expectile must be in (0,1); polyak in (0,1]")
        if not 0 <= self.discount < 1:
            raise ValueError("discount must be in [0,1)")
        if self.mpi_steps < 1 or self.mc_samples < 1 or self.inner_updates < 1:
            raise ValueError("mpi_steps must be >=1 (K=1 is base); mc_samples and inner_updates >=1")
        if not self.hidden_dims or any(x < 1 for x in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not all(math.isfinite(x) for x in (self.log_std_min, self.log_std_init, self.log_std_max)):
            raise ValueError("log standard deviations must be finite")
        if not self.log_std_min < self.log_std_init < self.log_std_max:
            raise ValueError("require log_std_min < log_std_init < log_std_max")
        if "qbc_gaussian_w2" in self.variants and not self.log_std_min < 0 < self.log_std_max:
            raise ValueError("Gaussian DDPG+BC refinement bounds must contain log(sigma=1)=0")
        if self.metric_reduction not in ("native", "sum", "mean"):
            raise ValueError("metric_reduction must be native, sum or mean")
        if self.q_action_transform not in ("identity", "clip"):
            raise ValueError("q_action_transform must be identity or clip")

    @property
    def h(self):
        return self.tau / self.mpi_steps

    def reduction(self, variant):
        if self.metric_reduction != "native":
            return self.metric_reduction
        return "mean" if variant == "qbc_deterministic_w2" else "sum"

    def normalize_q(self, variant):
        if self.iql_q_scale_norm is not None:
            return self.iql_q_scale_norm
        return variant == "qbc_deterministic_w2"

    def coefficients(self, variant, action_dim, horizon=None):
        """Coefficients in each native base loss; metric changes affect base too."""
        h = self.h if horizon is None else horizon
        divisor = action_dim if self.reduction(variant) == "mean" else 1
        if variant == "awr_gaussian_fr":
            return {"awr_beta": h * divisor}
        if variant == "qbc_gaussian_w2":
            return {"bc_coef": 1 / (h * divisor)}
        return {"td3bc_alpha": 2 * h * divisor / action_dim}


def default_t_grid(args):
    if (args.metric_reduction != "native" or args.iql_q_scale_norm is not None
            or args.iql_reward_normalization != "iql" or args.reward_scale != 1.):
        raise ValueError("use explicit --taus when overriding native geometry or Q/reward scale")
    variants = args.variants.replace(",", " ").split()
    if not variants or set(variants).difference(VARIANTS):
        raise ValueError(f"select variants from {VARIANTS}")
    return sorted({t for v in variants for t in NATIVE_T_GRIDS[v]})[:args.n_tau]


def add_iql_args(parser):
    parser.add_argument("--variants", default=" ".join(VARIANTS))
    for name, default in (("expectile", 0.7),
                          ("actor-lr", 3e-4), ("critic-lr", 3e-4), ("value-lr", 3e-4),
                          ("discount", 0.99), ("log-std-init", -1.0),
                          ("log-std-min", -5.0), ("log-std-max", 2.0), ("reward-scale", 1.0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", default="256 256")
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--inner-updates", type=int, default=1)
    parser.add_argument("--metric-reduction", choices=("native", "sum", "mean"), default="native")
    parser.add_argument("--q-action-transform", choices=("identity", "clip"), default="identity",
                        help="MPI expected-Q action transform; DDPG+BC base always clips its mean.")
    parser.add_argument("--iql-q-scale-norm", action=argparse.BooleanOptionalAction, default=None,
                        help="Default: only deterministic TD3+BC normalizes Q. Override applies to base AND MPI.")
    parser.add_argument("--iql-reward-normalization", choices=("iql", "none"), default="iql",
                        help="IQL locomotion rewards: multiply by 1000/(max return - min return).")
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
            "reward_normalization": args.iql_reward_normalization,
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
