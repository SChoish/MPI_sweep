import argparse
import json

import pytest

import launch_mpi_sweep as launcher
import train_iql_mpi as trainer
from iql_mpi_config import config_from_args, is_complete, run_name


def test_worker_roundtrip_and_separate_identity(tmp_path):
    args = launcher.parse_args(["--algorithm", "iql", "--hops", "4", "--taus", "0.5 1",
                                "--seeds", "0", "--domains", "hopper", "--datasets", "medium",
                                "--save-dir", str(tmp_path), "--inner-updates", "2",
                                "--metric-reduction", "mean", "--iql-q-scale-norm"])
    launcher.validate_args(args)
    jobs = launcher.build_jobs(args)
    assert len(jobs) == 2
    for job in jobs:
        command = launcher.worker_command(args, job, 0)
        assert command[2].endswith("train_iql_mpi.py")
        worker_args = trainer.parse_args(command[3:])
        config = config_from_args(worker_args)
        assert config.inner_updates == 2 and config.iql_q_scale_norm
        assert config.metric_reduction == "mean"
        assert run_name(worker_args, config) == job.tag
    args.bc_coef = 2.
    assert launcher.build_jobs(args)[0].tag != jobs[0].tag


def test_iql_baseline_only_and_rejects_explicit():
    args = launcher.parse_args(["--algorithm", "iql", "--hops", "0"])
    launcher.validate_args(args)
    args.integrator = "explicit"
    with pytest.raises(ValueError, match="implicit"):
        launcher.validate_args(args)


def test_eval_alone_never_skips_iql_job(tmp_path):
    args = launcher.parse_args(["--algorithm", "iql", "--hops", "1", "--taus", "1",
                                "--domains", "hopper", "--datasets", "medium", "--seeds", "0",
                                "--save-dir", str(tmp_path)])
    job = launcher.build_jobs(args)[0]
    directory = tmp_path / job.tag
    directory.mkdir()
    (directory / "eval.csv").write_text("step,return\n1000000,20\n")
    assert len(launcher.build_jobs(args)) == 1
    assert not is_complete(directory, 1_000_000)


def test_training_cli_resume_final_eval_recovery(tmp_path, monkeypatch):
    import h5py
    import numpy as np
    from d4rl_data import dataset_path
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n = 32
    with h5py.File(dataset_path("hopper-medium-v2", data_dir), "w") as f:
        f["observations"] = np.random.default_rng(0).normal(size=(n, 3)).astype("float32")
        f["actions"] = np.zeros((n, 2), dtype="float32")
        f["rewards"] = np.ones(n, dtype="float32")
        f["terminals"] = np.zeros(n, dtype="float32")
        f["timeouts"] = np.zeros(n, dtype="float32")
    # Exercise actual policies (including Gaussian sampling), stub only simulator.
    def evaluate(policy, env, seed, mean, std, episodes):
        action = np.asarray(policy(np.zeros(3, dtype="float32")))
        assert action.shape == (2,) and np.isfinite(action).all()
        assert np.abs(action).max() <= 1.
        return 1., 2.
    monkeypatch.setattr(trainer, "evaluate", evaluate)
    monkeypatch.setattr(trainer, "install_stop_handler", lambda _: {"flag": False})
    argv = ["--data-dir", str(data_dir), "--save-dir", str(tmp_path / "runs"),
            "--max-timesteps", "2", "--eval-freq", "2", "--updates-per-dispatch", "2",
            "--mpi-steps", "1", "--hidden-dims", "8", "--mc-samples", "2",
            "--batch-size", "4", "--eval-episodes", "1"]
    assert trainer.main(argv) == 0
    args = trainer.parse_args(argv)
    directory = args.save_dir / run_name(args, config_from_args(args))
    assert is_complete(directory, 2)
    assert len(trainer.read_rows(directory / "eval.csv")) == 10
    assert trainer.main(argv) == 0
    # Simulate a crash after final checkpoint but before evaluations complete.
    (directory / "COMPLETE.json").unlink()
    (directory / "eval.csv").unlink()
    assert trainer.main(argv) == 0
    assert is_complete(directory, 2)
    assert len(trainer.read_rows(directory / "eval.csv")) == 10
