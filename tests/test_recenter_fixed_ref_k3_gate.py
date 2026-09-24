"""The pilot gate must reject diverging shared trajectories and update counts."""

import csv
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.experiments.verify_recenter_fixed_ref_k3 import run_dir, verify_pair


def _write_pair(tmp_path, *, critic_delta=0.0, pi2_delta=0.0,
                last_actor_updates=512):
    env, tau, seed, step = "hopper-medium-replay-v2", 2.5, 0, 1024
    for branch in ("recenter", "fixed_ref"):
        path = run_dir(tmp_path, env, tau, seed, branch)
        path.mkdir(parents=True)
        checkpoint = {
            "step": step,
            "rng": np.array([3, 2], dtype=np.uint32),
            "mean": np.array([0.0]), "std": np.array([1.0]),
            "max_action": 1.0, "policy_noise": 0.2, "noise_clip": 0.5,
            "critic_params": {"w": np.array([1.0 + (critic_delta if branch == "fixed_ref" else 0)])},
            "critic_opt_state": (np.array([0.1]),), "critic_step": step,
            "target_actor_params": {"w": np.array([2.0])},
            "target_critic_params": {"w": np.array([3.0])},
            "actors_params": ({"w": np.array([1.0])},
                              {"w": np.array([2.0 + (pi2_delta if branch == "fixed_ref" else 0)])},
                              {"w": np.array([4.0 if branch == "recenter" else 5.0])}),
            "actors_steps": (512, 512, last_actor_updates),
            "actors_opt_states": ((np.array([1.0]),), (np.array([2.0]),),
                                  (np.array([3.0]),)),
            "config": {"method": "shared", "deployment_branch": branch,
                       "integrator": "implicit", "mpi_steps": 3,
                       "seed": seed, "tau": tau, "env": env,
                       "policy_freq": 2, "save_dir": str(path)},
        }
        with (path / f"params_{step}.pkl").open("wb") as handle:
            pickle.dump(checkpoint, handle)
        with (path / "eval.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "d4rl_score", "d4rl_pi3"])
            writer.writeheader()
            writer.writerow({"step": step, "d4rl_score": 80.0,
                             "d4rl_pi3": 102.0 if branch == "recenter" else 99.0})
    return env, tau, seed, step


class PilotGateTest(unittest.TestCase):
    def test_accepts_matched_shared_trajectory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = verify_pair(root, *_write_pair(root))
            self.assertEqual(report["status"], "pair_verified")
            self.assertEqual(report["critic_updates"], 1024)
            self.assertEqual(report["actor_updates_per_actor"], 512)
            self.assertEqual(report["paired_delta"], 3.0)

    def test_rejects_wrong_critic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = _write_pair(root, critic_delta=0.01)
            with self.assertRaisesRegex(ValueError, "critic_params"):
                verify_pair(root, *args)

    def test_rejects_wrong_second_actor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = _write_pair(root, pi2_delta=0.01)
            with self.assertRaisesRegex(ValueError, "shared pi_2 state differs"):
                verify_pair(root, *args)

    def test_rejects_wrong_actor_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = _write_pair(root, last_actor_updates=511)
            with self.assertRaisesRegex(ValueError, "actor optimizer update counts"):
                verify_pair(root, *args)
