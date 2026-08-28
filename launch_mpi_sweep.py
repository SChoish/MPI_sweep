#!/usr/bin/env python3
"""Run a resumable K-hop MPI sweep across one or more GPUs."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from d4rl_data import DATASET_FILES, download_dataset
from tau_grids import MPI_TAU_GRID, mpi_tau_grid

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
    parser.add_argument("--hops", type=int, required=True)
    parser.add_argument(
        "--integrator",
        choices=("implicit", "explicit"),
        default="implicit",
        help="implicit JKO hops or matched-scale projected explicit hops",
    )
    parser.add_argument("--n-tau", type=int, default=len(MPI_TAU_GRID))
    parser.add_argument(
        "--taus",
        default="",
        help="Comma- or space-separated custom total tau values; overrides --n-tau",
    )
    parser.add_argument("--seeds", default="0 1")
    parser.add_argument("--gpus", default="0", help="Visible GPU IDs, comma or space separated")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--domains", default="hopper halfcheetah walker2d")
    parser.add_argument("--datasets", default="medium medium-replay expert")
    parser.add_argument("--polyak", type=float, default=0.005)
    parser.add_argument("--max-timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--n-jitted-updates", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=100_000)
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
    if args.hops < 1:
        raise ValueError("--hops must be positive")
    if args.n_tau < 1:
        raise ValueError("--n-tau must be positive")
    if args.slots_per_gpu < 1:
        raise ValueError("--slots-per-gpu must be positive")
    if args.cpus_per_job < 1:
        raise ValueError("--cpus-per-job must be positive")
    if not _split(args.gpus):
        raise ValueError("--gpus must contain at least one GPU ID")
    if args.cpu_affinity and shutil.which("taskset") is None:
        raise RuntimeError("--cpu-affinity requires the Linux taskset command")


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
    if any(float(value) <= 0.0 for value in values):
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
    method_tag = "mpi" if args.integrator == "implicit" else "exp"
    for env_name, tau, seed_text in product(
        selected_envs(args), selected_taus(args), _split(args.seeds)
    ):
        seed = int(seed_text)
        tag = f"{env_name}_tau{tau}_{method_tag}{args.hops}_seed{seed}"
        if not is_complete(args.save_dir, tag, args.max_timesteps):
            jobs.append(Job(env_name, tau, seed, tag))
    return jobs


def worker_environment(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def worker_command(
    args: argparse.Namespace,
    job: Job,
    slot_index: int,
) -> list[str]:
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
        "--save-interval",
        str(args.save_interval),
        "--data-dir",
        str(args.data_dir),
        "--save-dir",
        str(args.save_dir),
        "--q-scale-norm" if args.q_scale_norm else "--no-q-scale-norm",
    ]
    if args.cpu_affinity:
        cpu_start = slot_index * args.cpus_per_job
        cpu_end = cpu_start + args.cpus_per_job - 1
        command = ["taskset", "-c", f"{cpu_start}-{cpu_end}", *command]
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


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    jobs = build_jobs(args)
    gpus = _split(args.gpus)
    concurrency = len(gpus) * args.slots_per_gpu
    print(
        f"[sweep] integrator={args.integrator} hops={args.hops} "
        f"jobs={len(jobs)} gpus={gpus} "
        f"slots/gpu={args.slots_per_gpu} concurrency={concurrency}",
        flush=True,
    )
    print(f"[sweep] tau_grid={' '.join(selected_taus(args))}", flush=True)

    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            command = worker_command(args, job, index % concurrency)
            print(f"[dry-run] gpu={gpu} {shlex.join(command)}")
        return 0

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
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
                gpu = gpus[slot_index // args.slots_per_gpu]
                job = jobs[next_job]
                next_job += 1
                log_handle = (args.log_dir / f"{job.tag}.log").open(
                    "w", encoding="utf-8"
                )
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
                if process.returncode == 0:
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
