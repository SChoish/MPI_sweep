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
        assert config.h == float(job.tau) / 4
        assert run_name(worker_args, config) == job.tag
    args.iql_reward_normalization = "none"
    assert launcher.build_jobs(args)[0].tag != jobs[0].tag
    args.iql_reward_normalization = "iql"
    args.iql_q_scale_norm = False
    assert launcher.build_jobs(args)[0].tag != jobs[0].tag


def test_iql_baseline_only_and_rejects_explicit():
    args = launcher.parse_args(["--algorithm", "iql", "--hops", "1"])
    launcher.validate_args(args)
    args.hops = 0
    with pytest.raises(ValueError, match="mpi_steps"):
        launcher.validate_args(args)
    args.hops = 1
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


@pytest.mark.parametrize("hops,expected_rows", [(1, 5), (3, 10)])
def test_training_cli_resume_final_eval_recovery(tmp_path, monkeypatch, hops, expected_rows):
    import h5py
    import numpy as np
    from d4rl_data import dataset_path
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n = 32
    with h5py.File(dataset_path("hopper-medium-v2", data_dir), "w") as f:
        f["observations"] = np.random.default_rng(0).normal(size=(n, 3)).astype("float32")
        f["actions"] = np.zeros((n, 2), dtype="float32")
        f["rewards"] = np.arange(n, dtype="float32")
        f["terminals"] = np.zeros(n, dtype="float32")
        f["timeouts"] = np.array([0.] * 15 + [1.] + [0.] * 16, dtype="float32")
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
            "--mpi-steps", str(hops), "--hidden-dims", "8", "--mc-samples", "2",
            "--batch-size", "4", "--eval-episodes", "1"]
    assert trainer.main(argv) == 0
    args = trainer.parse_args(argv)
    directory = args.save_dir / run_name(args, config_from_args(args))
    assert is_complete(directory, 2)
    assert len(trainer.read_rows(directory / "eval.csv")) == expected_rows
    rows = trainer.read_rows(directory / "eval.csv")
    assert {float(r["time"]) for r in rows} == {1.}
    assert {int(r["K"]) for r in rows} == {1, hops}
    resolved = json.loads((directory / "config.json").read_text())
    assert resolved["resolved_protocol"]["K"] == hops
    assert resolved["reward_normalization"]["factor"] > 0
    assert trainer.main(argv) == 0
    assert json.loads((directory / "config.json").read_text()) == resolved
    # Simulate a crash after final checkpoint but before evaluations complete.
    (directory / "COMPLETE.json").unlink()
    (directory / "eval.csv").unlink()
    assert trainer.main(argv) == 0
    assert is_complete(directory, 2)
    assert len(trainer.read_rows(directory / "eval.csv")) == expected_rows


@pytest.mark.parametrize("variant,expected", [
    ("awr_gaussian_fr", [1., 3., 10.]),
    ("qbc_gaussian_w2", [.02, .05, .1, .2, .4, .7, 1.5, 2.5, 4., 5.]),
    ("qbc_deterministic_w2", [1.25]),
])
def test_default_time_grids_are_independent_of_depth(variant, expected):
    args = launcher.parse_args(["--algorithm", "iql", "--hops", "1", "--variants", variant])
    assert [float(x) for x in launcher.selected_taus(args)] == expected
    args.hops = 4
    assert [float(x) for x in launcher.selected_taus(args)] == expected
    args.iql_q_scale_norm = True
    with pytest.raises(ValueError, match="explicit --taus"):
        launcher.selected_taus(args)


def test_old_independent_actor_coefficients_are_rejected():
    for flag in ("--awr-beta", "--bc-coef", "--td3bc-alpha"):
        with pytest.raises(SystemExit):
            trainer.parse_args([flag, "3"])


def test_reward_scale_matches_official_trajectory_returns_without_changing_masks():
    import numpy as np
    # Three trajectories: terminal, skipped-timeout discontinuity, final tail.
    raw = {"observations": np.array([[0.], [1.], [2.], [3.], [10.], [11.]]),
           "next_observations": np.array([[1.], [2.], [3.], [4.], [11.], [12.]]),
           "not_dones": np.array([[1.], [0.], [1.], [1.], [1.], [1.]]),
           "rewards": np.array([[1.], [2.], [2.], [3.], [4.], [5.]])}
    masks = raw["not_dones"].copy()
    rewards, info = trainer.normalize_iql_rewards(raw)
    # returns 3, 5, 9 => multiply by 1000/6; timeout doesn't mask bootstrap.
    assert info["return_range"] == 6.
    np.testing.assert_allclose(rewards, raw["rewards"] * 1000/6, rtol=1e-6)
    np.testing.assert_array_equal(raw["not_dones"], masks)
    unscaled, info = trainer.normalize_iql_rewards(raw, "none", 2.)
    np.testing.assert_array_equal(unscaled, raw["rewards"] * 2)
    raw["rewards"][:] = 1.
    with pytest.raises(ValueError, match="return range"):
        trainer.normalize_iql_rewards(raw)
