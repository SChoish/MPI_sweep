from pathlib import Path

from launch_mpi_sweep import build_jobs, is_complete, parse_args, run, worker_command


def small_args(tmp_path: Path):
    return parse_args(
        [
            "--hops",
            "4",
            "--taus",
            "0.1,0.4",
            "--seeds",
            "0",
            "--gpus",
            "0,1",
            "--domains",
            "hopper",
            "--datasets",
            "medium",
            "--save-dir",
            str(tmp_path / "results"),
            "--data-dir",
            str(tmp_path / "data"),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )


def test_job_matrix_and_worker_command(tmp_path: Path):
    args = small_args(tmp_path)
    jobs = build_jobs(args)
    assert [job.tag for job in jobs] == [
        "hopper-medium-v2_tau0.1_mpi4_seed0",
        "hopper-medium-v2_tau0.4_mpi4_seed0",
    ]
    command = worker_command(args, jobs[0], slot_index=0)
    assert command[0] == args.python
    assert command[command.index("--mpi-steps") + 1] == "4"
    assert command[command.index("--eval-freq") + 1] == "1000000"
    assert command[command.index("--save-interval") + 1] == "0"
    assert command[command.index("--updates-per-dispatch") + 1] == "64"
    assert "--compilation-cache-dir" in command
    assert "--q-scale-norm" in command


def test_completed_job_is_skipped(tmp_path: Path):
    args = small_args(tmp_path)
    tag = "hopper-medium-v2_tau0.1_mpi4_seed0"
    run_dir = args.save_dir / tag
    run_dir.mkdir(parents=True)
    (run_dir / "eval.csv").write_text(
        "step,return\n50000,1.0\n1000000,2.0\n", encoding="utf-8"
    )
    assert is_complete(args.save_dir, tag, 1_000_000)
    assert [job.tau for job in build_jobs(args)] == ["0.4"]


def test_explicit_jobs_use_separate_tags_and_worker_mode(tmp_path: Path):
    args = small_args(tmp_path)
    args.integrator = "explicit"
    jobs = build_jobs(args)
    assert jobs[0].tag == "hopper-medium-v2_tau0.1_exp4_seed0"
    command = worker_command(args, jobs[0], slot_index=0)
    assert command[command.index("--integrator") + 1] == "explicit"


def test_two_actor_legacy_token_tag_and_worker_mode(tmp_path: Path):
    args = small_args(tmp_path)
    args.method = "mcep"
    args.hops = 3
    jobs = build_jobs(args)
    assert jobs[0].tag == "hopper-medium-v2_tau0.1_mcep3_seed0"
    command = worker_command(args, jobs[0], slot_index=0)
    assert command[command.index("--method") + 1] == "mcep"
    assert command[command.index("--mpi-steps") + 1] == "3"


def test_dry_run_has_no_filesystem_side_effects(tmp_path: Path):
    args = small_args(tmp_path)
    args.dry_run = True
    assert run(args) == 0
    assert not args.data_dir.exists()
    assert not args.log_dir.exists()


def test_cpu_jobs_pin_and_ignore_gpus(tmp_path: Path):
    args = parse_args(
        [
            "--hops",
            "4",
            "--taus",
            "24",
            "--seeds",
            "2",
            "--cpu-jobs",
            "2",
            "--cpus-per-job",
            "4",
            "--cpu-start",
            "200",
            "--domains",
            "hopper",
            "--datasets",
            "medium",
            "--save-dir",
            str(tmp_path / "results"),
            "--data-dir",
            str(tmp_path / "data"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--dry-run",
        ]
    )
    jobs = build_jobs(args)
    command = worker_command(args, jobs[0], slot_index=1)
    assert command[:3] == ["taskset", "-c", "204-207"]
    assert run(args) == 0
