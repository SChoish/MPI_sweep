#!/usr/bin/env python3
"""Run a resumable K-hop BAR sweep across one or more GPUs."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import provenance
from d4rl_data import DATASET_FILES, download_dataset
from tau_grids import MPI_TAU_GRID, mpi_tau_grid
import iql_mpi_config as iql_config

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Job:
    env_name: str
    tau: str
    seed: int
    tag: str


def _split(value: str) -> list[str]:
    return value.replace(",", " ").split()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("td3bc", "iql"), default="td3bc")
    iql_config.add_iql_args(parser)
    parser.add_argument("--hops", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=("bar", "mcep"),
        default="bar",
        help=(
            "BAR chain or two-actor policy-separation control "
            "(legacy mcep token; not published reproduction)"
        ),
    )
    parser.add_argument(
        "--integrator",
        choices=("implicit", "explicit"),
        default="implicit",
        help="proximal-loss hops or action-metric-matched projected-linearized hops",
    )
    parser.add_argument("--n-tau", type=int, default=len(MPI_TAU_GRID))
    parser.add_argument(
        "--taus",
        default="",
        help="Comma- or space-separated custom total tau values; overrides --n-tau",
    )
    parser.add_argument("--seeds", default="0 1")
    parser.add_argument("--gpus", default="0", help="Visible GPU IDs, comma or space separated")
    parser.add_argument(
        "--cpu-jobs",
        type=int,
        default=0,
        help="If >0, run this many concurrent CPU workers and ignore --gpus",
    )
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--domains", default="hopper halfcheetah walker2d")
    parser.add_argument("--datasets", default="medium medium-replay expert")
    parser.add_argument("--polyak", type=float, default=0.005)
    parser.add_argument("--max-timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1_000_000,
        help="Eval period. Default 1M so tau plots use a single final score.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--n-jitted-updates", type=int, default=8)
    parser.add_argument(
        "--updates-per-dispatch",
        type=int,
        default=64,
        help="Updates fused into each host-to-GPU dispatch",
    )
    parser.add_argument(
        "--compilation-cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "mpi-sweep" / "jax",
        help="Shared persistent JAX compilation cache",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Periodic checkpoint interval; 0 = only 1M + emergency save",
    )
    parser.add_argument("--save-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--cpu-affinity",
        action="store_true",
        help="Pin each worker with taskset (Linux only)",
    )
    parser.add_argument("--cpus-per-job", type=int, default=1)
    parser.add_argument(
        "--cpu-start",
        type=int,
        default=0,
        help="First host CPU id used by --cpu-affinity",
    )
    parser.add_argument(
        "--q-scale-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download all selected D4RL files before launching workers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job matrix without downloading data or starting workers",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.algorithm == "iql":
        if args.method != "bar" or args.integrator != "implicit":
            raise ValueError("IQL actor/geometry comparison supports only the implicit chain")
        for tau in selected_taus(args):
            iql_config.config_from_args(args, tau=tau, hops=args.hops)
        for name in ("batch_size", "max_timesteps", "eval_freq", "eval_episodes"):
            if getattr(args, name) < 1:
                raise ValueError(f"{name} must be positive")
        if args.save_interval < 0 or not math.isfinite(args.reward_scale) or args.reward_scale <= 0:
            raise ValueError("invalid save_interval or reward_scale")
        if any(int(s) < 0 for s in _split(args.seeds)):
            raise ValueError("seeds must be nonnegative")
    elif args.hops < 1:
        raise ValueError("--hops must be positive")
    if args.method == "mcep" and args.hops < 2:
        raise ValueError(
            "two-actor policy-separation control (legacy mcep token; not "
            "published reproduction) requires --hops >= 2"
        )
    if args.method == "mcep" and args.integrator != "implicit":
        raise ValueError(
            "two-actor policy-separation control (legacy mcep token; not "
            "published reproduction) supports only --integrator implicit"
        )
    if args.n_tau < 1:
        raise ValueError("--n-tau must be positive")
    if args.slots_per_gpu < 1:
        raise ValueError("--slots-per-gpu must be positive")
    if args.cpus_per_job < 1:
        raise ValueError("--cpus-per-job must be positive")
    if args.updates_per_dispatch < 1:
        raise ValueError("--updates-per-dispatch must be positive")
    if args.n_jitted_updates < 1:
        raise ValueError("--n-jitted-updates must be positive")
    if args.algorithm != "iql" and args.updates_per_dispatch % args.n_jitted_updates != 0:
        raise ValueError("--updates-per-dispatch must be divisible by --n-jitted-updates")
    if args.cpu_jobs < 0:
        raise ValueError("--cpu-jobs must be >= 0")
    if args.cpu_jobs == 0 and not _split(args.gpus):
        raise ValueError("--gpus must contain at least one GPU ID")
    if args.cpu_start < 0:
        raise ValueError("--cpu-start must be >= 0")
    if (args.cpu_affinity or args.cpu_jobs > 0) and shutil.which("taskset") is None:
        raise RuntimeError("CPU pinning requires the Linux taskset command")


def selected_envs(args: argparse.Namespace) -> list[str]:
    envs = [
        f"{domain}-{dataset}-v2"
        for domain, dataset in product(_split(args.domains), _split(args.datasets))
    ]
    unknown = sorted(set(envs).difference(DATASET_FILES))
    if unknown:
        raise ValueError(f"unsupported environment(s): {', '.join(unknown)}")
    return envs


def selected_taus(args: argparse.Namespace) -> list[str]:
    values = _split(args.taus) if args.taus else mpi_tau_grid(args.n_tau)
    if not values:
        raise ValueError("the tau grid is empty")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
        raise ValueError("all tau values must be positive")
    return [f"{float(value):g}" for value in values]


def is_complete(save_dir: Path, tag: str, max_timesteps: int) -> bool:
    eval_path = save_dir / tag / "eval.csv"
    if not eval_path.is_file():
        return False
    prefix = f"{max_timesteps},"
    return any(
        line.startswith(prefix)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
    )


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    method_tag = (
        "mcep"
        if args.method == "mcep"
        else ("mpi" if args.integrator == "implicit" else "exp")
    )
    for env_name, tau, seed_text in product(
        selected_envs(args), selected_taus(args), _split(args.seeds)
    ):
        seed = int(seed_text)
        if args.algorithm == "iql":
            cfg = iql_config.config_from_args(args, tau=tau, hops=args.hops)
            run_args = argparse.Namespace(**{**vars(args), "env": env_name, "seed": seed})
            tag = iql_config.run_name(run_args, cfg)
            if not iql_config.is_complete(args.save_dir / tag, args.max_timesteps):
                jobs.append(Job(env_name, tau, seed, tag))
            continue
        tag = f"{env_name}_tau{tau}_{method_tag}{args.hops}_seed{seed}"
        if not is_complete(args.save_dir, tag, args.max_timesteps):
            jobs.append(Job(env_name, tau, seed, tag))
    return jobs


def worker_environment(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "EIGEN_NUM_THREADS": "1",
        }
    )
    if gpu == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1 "
            "inter_op_parallelism_threads=1"
        )
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def worker_command(
    args: argparse.Namespace,
    job: Job,
    slot_index: int,
) -> list[str]:
    if args.algorithm == "iql":
        return iql_worker_command(args, job, slot_index)
    command = [
        args.python,
        "-u",
        str(ROOT / "train_td3bc.py"),
        "--env",
        job.env_name,
        "--tau",
        job.tau,
        "--mpi-steps",
        str(args.hops),
        "--method",
        args.method,
        "--integrator",
        args.integrator,
        "--polyak",
        str(args.polyak),
        "--seed",
        str(job.seed),
        "--max-timesteps",
        str(args.max_timesteps),
        "--eval-freq",
        str(args.eval_freq),
        "--eval-episodes",
        str(args.eval_episodes),
        "--n-jitted-updates",
        str(args.n_jitted_updates),
        "--updates-per-dispatch",
        str(args.updates_per_dispatch),
        "--compilation-cache-dir",
        str(args.compilation_cache_dir),
        "--save-interval",
        str(args.save_interval),
        "--data-dir",
        str(args.data_dir),
        "--save-dir",
        str(args.save_dir),
        "--q-scale-norm" if args.q_scale_norm else "--no-q-scale-norm",
    ]
    if args.cpu_affinity or args.cpu_jobs > 0:
        cpu_start = args.cpu_start + slot_index * args.cpus_per_job
        cpu_end = cpu_start + args.cpus_per_job - 1
        command = ["taskset", "-c", f"{cpu_start}-{cpu_end}", *command]
    return command


def iql_worker_command(args, job, slot_index):
    command = [args.python, "-u", str(ROOT / "train_iql_mpi.py"),
               "--env", job.env_name, "--tau", job.tau, "--mpi-steps", str(args.hops),
               "--seed", str(job.seed)]
    for name in ("polyak", "max_timesteps", "eval_freq", "eval_episodes", "updates_per_dispatch",
                 "compilation_cache_dir", "save_interval", "data_dir", "save_dir", "variants",
                 "expectile", "awr_beta", "bc_coef", "actor_lr", "critic_lr", "value_lr",
                 "discount", "hidden_dims", "log_std_init", "log_std_min", "log_std_max",
                 "mc_samples", "inner_updates", "metric_reduction", "q_action_transform",
                 "batch_size", "reward_scale", "eval_mode", "eval_hops"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    for name in ("iql_q_scale_norm", "iql_normalize_state"):
        command.append("--" + ("" if getattr(args, name) else "no-") + name.replace("_", "-"))
    if args.cpu_affinity or args.cpu_jobs > 0:
        start = args.cpu_start + slot_index * args.cpus_per_job
        command = ["taskset", "-c", f"{start}-{start + args.cpus_per_job - 1}", *command]
    return command


def terminate_workers(running: dict[int, subprocess.Popen]) -> None:
    for process in running.values():
        if process.poll() is None:
            process.terminate()
    for process in running.values():
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def write_sweep_provenance(args: argparse.Namespace, jobs: list[Job]) -> None:
    """Record orchestrator-level provenance once per sweep (best-effort)."""
    try:
        source_files = {
            "launch_mpi_sweep.py": ROOT / "launch_mpi_sweep.py",
            "train_td3bc.py": ROOT / "train_td3bc.py",
        }
        if args.algorithm == "iql":
            source_files.update({name: ROOT / name for name in
                                 ("train_iql_mpi.py", "iql_mpi.py", "iql_mpi_config.py")})
        payload = provenance.base_provenance(
            ROOT,
            source_files,
            extra={
                "role": "sweep_orchestrator",
                "sweep": {
                    "algorithm": args.algorithm,
                    "configuration": vars(args),
                    "method": args.method,
                    "integrator": args.integrator,
                    "hops": args.hops,
                    "taus": selected_taus(args),
                    "seeds": _split(args.seeds),
                    "envs": selected_envs(args),
                    "gpus": _split(args.gpus),
                    "slots_per_gpu": args.slots_per_gpu,
                    "max_timesteps": args.max_timesteps,
                    "pending_jobs": [job.tag for job in jobs],
                    "pending_job_count": len(jobs),
                },
            },
        )
        provenance.write_json(args.log_dir / "SWEEP_PROVENANCE.json", payload)
        print(
            f"[provenance] wrote {args.log_dir / 'SWEEP_PROVENANCE.json'}",
            flush=True,
        )
    except Exception as error:  # provenance must never break a sweep
        print(f"[provenance] sweep capture skipped: {error}", flush=True)


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    jobs = build_jobs(args)
    cpu_mode = args.cpu_jobs > 0
    gpus = ["cpu"] if cpu_mode else _split(args.gpus)
    concurrency = args.cpu_jobs if cpu_mode else len(gpus) * args.slots_per_gpu
    print(
        f"[sweep] algorithm={args.algorithm} method={args.method} integrator={args.integrator} hops={args.hops} "
        f"jobs={len(jobs)} device={'cpu' if cpu_mode else gpus} "
        f"slots/gpu={args.slots_per_gpu} concurrency={concurrency}",
        flush=True,
    )
    print(f"[sweep] tau_grid={' '.join(selected_taus(args))}", flush=True)

    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = "cpu" if cpu_mode else gpus[index % len(gpus)]
            command = worker_command(args, job, index % concurrency)
            print(f"[dry-run] gpu={gpu} {shlex.join(command)}")
        return 0

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    write_sweep_provenance(args, jobs)
    if args.prefetch:
        for env_name in selected_envs(args):
            path = download_dataset(env_name, args.data_dir)
            print(f"[data] ready {path}", flush=True)

    free_slots = list(range(concurrency))
    running: dict[int, subprocess.Popen] = {}
    metadata: dict[int, tuple[Job, int, object, float]] = {}
    next_job = 0
    failures = 0

    try:
        while next_job < len(jobs) or running:
            while next_job < len(jobs) and free_slots:
                slot_index = free_slots.pop(0)
                gpu = "cpu" if cpu_mode else gpus[slot_index // args.slots_per_gpu]
                job = jobs[next_job]
                next_job += 1
                log_handle = (args.log_dir / f"{job.tag}.log").open(
                    "a", encoding="utf-8"
                )
                print(
                    f"\n[launcher] start_or_resume {job.tag} "
                    f"at={time.strftime('%Y-%m-%d %H:%M:%S')}",
                    file=log_handle,
                    flush=True,
                )
                if cpu_mode and running:
                    time.sleep(25.0)
                process = subprocess.Popen(
                    worker_command(args, job, slot_index),
                    cwd=ROOT,
                    env=worker_environment(gpu),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                running[process.pid] = process
                metadata[process.pid] = (job, slot_index, log_handle, time.monotonic())
                print(f"[launch] gpu={gpu} {job.tag} pid={process.pid}", flush=True)

            if running:
                time.sleep(1.0)
            finished = [pid for pid, process in running.items() if process.poll() is not None]
            for pid in finished:
                process = running.pop(pid)
                job, slot_index, log_handle, started = metadata.pop(pid)
                log_handle.close()
                free_slots.append(slot_index)
                free_slots.sort()
                elapsed = (time.monotonic() - started) / 60.0
                completed = process.returncode == 0 and (
                    args.algorithm != "iql" or iql_config.is_complete(args.save_dir / job.tag, args.max_timesteps))
                if completed:
                    print(f"[ok] {job.tag} {elapsed:.1f}m", flush=True)
                else:
                    failures += 1
                    print(
                        f"[fail] {job.tag} rc={process.returncode} {elapsed:.1f}m",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("[sweep] interrupted; stopping workers", flush=True)
        terminate_workers(running)
        return 130
    finally:
        for _job, _slot, handle, _started in metadata.values():
            handle.close()

    print(f"[sweep] complete ok={len(jobs) - failures} fail={failures}", flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
