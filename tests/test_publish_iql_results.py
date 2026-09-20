import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "publisher", Path(__file__).resolve().parents[1] / "scripts/publish_iql_results.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "raw"
        self.source.mkdir()

    def run_fixture(self, k=3, seed=0):
        signature = dict(schema="iql_actor_geometry_v5_consistent_gaussian",
                         algorithm=dict(variants=["qbc_gaussian_w2"], mpi_steps=k,
                                        tau=.4, gaussian_qbc_mode="stochastic"),
                         env="halfcheetah-medium-v2", seed=seed,
                         eval_hops="endpoints", eval_mode="both")
        directory = self.source / f"run-{k}-{seed}"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(dict(signature=signature)))
        checkpoint = b"test fixture only; never deserialized"
        (directory / "params_1000000.pkl").write_bytes(checkpoint)
        rows = []
        policies = [("baseline", 1)] + ([("mpi", 1), ("mpi", k)] if k > 1 else [])
        for policy, hop in policies:
            for mode in ("mean", "sample"):
                rows.append(dict(step=1000000, variant="qbc_gaussian_w2", policy=policy,
                                 hop=hop, K=1 if policy == "baseline" else k, T=.4,
                                 refinement_geometry="w2", gaussian_qbc_mode="stochastic",
                                 eval_mode=mode, d4rl_score="45.123456789" if (policy, hop, mode) ==
                                 ("baseline" if k == 1 else "mpi", k, "mean") else "999"))
        with (directory / "eval.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        marker = dict(schema=signature["schema"], step=1000000, signature=mod.signature_hash(signature),
                      eval_sha256=mod.digest((directory / "eval.csv").read_bytes()),
                      checkpoint_bytes=len(checkpoint), checkpoint_sha256=mod.digest(checkpoint),
                      evaluation_rows=len(rows))
        (directory / "COMPLETE.json").write_text(json.dumps(marker))
        return directory

    def test_final_mean_score_and_machine_provenance_reach_table(self):
        self.run_fixture()
        (self.source / "pending").mkdir()
        scores, manifest, count = mod.export(self.source, "qbc_gaussian_w2", "ext_csv")
        rows = list(csv.DictReader(io.StringIO(scores)))
        self.assertEqual(count, 1)
        self.assertEqual(rows[0]["d4rl_score"], "45.123456789")
        self.assertEqual((rows[0]["K"], rows[0]["hop"], rows[0]["step"]), ("3", "3", "1000000"))
        self.assertEqual(rows[0]["machine"], "ext_csv")
        self.assertEqual(json.loads(manifest)["pending_directories"], 1)
        table_spec = importlib.util.spec_from_file_location(
            "table", Path(__file__).resolve().parents[1] / "scripts/build_main_results.py")
        table = importlib.util.module_from_spec(table_spec)
        table_spec.loader.exec_module(table)
        folder = self.root / "sweep_results/iql_ext_csv"
        folder.mkdir(parents=True)
        (folder / "scores_verified.csv").write_text(scores)
        selected, _, warnings, *_ = table.collect(self.root)
        self.assertFalse(warnings)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[("Gaussian Q+BC–W2", "halfcheetah-medium-v2", .4, 3, 0)]["score"], 45.123456789)

    def test_standalone_k1(self):
        self.run_fixture(k=1)
        scores, _, _ = mod.export(self.source, "qbc_gaussian_w2", "ext_csv")
        row = next(csv.DictReader(io.StringIO(scores)))
        self.assertEqual((row["policy"], row["K"]), ("baseline", "1"))

    def test_missing_checkpoint_and_changed_evaluation_are_not_published(self):
        directory = self.run_fixture()
        checkpoint = directory / "params_1000000.pkl"
        original = checkpoint.read_bytes()
        checkpoint.unlink()
        with self.assertRaisesRegex(ValueError, "invalid snapshot"):
            mod.export(self.source, "qbc_gaussian_w2", "ext_csv")
        checkpoint.write_bytes(original)
        with (directory / "eval.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "evaluation hash mismatch"):
            mod.export(self.source, "qbc_gaussian_w2", "ext_csv")

    def test_duplicate_final_actor_rows_rejected_even_with_updated_hash(self):
        directory = self.run_fixture()
        path = directory / "eval.csv"
        path.write_text(path.read_text() + path.read_text().splitlines()[-1] + "\n")
        marker_path = directory / "COMPLETE.json"
        marker = json.loads(marker_path.read_text())
        marker["eval_sha256"] = mod.digest(path.read_bytes())
        marker["evaluation_rows"] += 1
        marker_path.write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError, "duplicate final evaluations"):
            mod.export(self.source, "qbc_gaussian_w2", "ext_csv")

    def test_progress_without_raw_evaluations_is_not_a_score(self):
        (self.source / "STATUS.json").write_text('{"counts":{"complete":92}}')
        with self.assertRaisesRegex(ValueError, "No verified final scores"):
            mod.export(self.source, "qbc_gaussian_w2", "ext_csv")

    def test_old_pilot_outside_queue_does_not_block_current_results(self):
        self.run_fixture()
        pilot = self.run_fixture(k=4)
        scores, manifest, count = mod.export(self.source, "qbc_gaussian_w2", "ext_csv")
        self.assertEqual(count, 1)
        self.assertEqual(json.loads(manifest)["excluded"][0]["run"], pilot.name)

    def test_publication_preserves_dirty_checkout_and_handles_remote_push_race(self):
        remote = self.root / "SChoish/MPI_sweep.git"
        remote.parent.mkdir()
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
        repo = self.root / "active"
        subprocess.run(["git", "clone", str(remote), str(repo)], check=True, capture_output=True)
        for key, value in (("user.name", "Test Publisher"), ("user.email", "test@example.invalid")):
            mod.git(repo, "config", key, value)
        mod.git(repo, "checkout", "-b", "main")
        (repo / "trainer.py").write_text("original\n")
        mod.git(repo, "add", "trainer.py")
        mod.git(repo, "commit", "-m", "Initial")
        mod.git(repo, "push", "origin", "HEAD:main")
        original_head = mod.git(repo, "rev-parse", "HEAD").stdout
        (repo / "trainer.py").write_text("uncommitted active training change\n")
        (repo / "unrelated.txt").write_text("staged local change\n")
        mod.git(repo, "add", "unrelated.txt")
        staged = mod.git(repo, "diff", "--cached").stdout
        self.run_fixture()
        scores, manifest, _ = mod.export(self.source, "qbc_gaussian_w2", "ext_csv")
        # Simulate another machine publishing just before the first push.
        competitor = self.root / "competitor"
        subprocess.run(["git", "clone", "--branch", "main", str(remote), str(competitor)],
                       check=True, capture_output=True)
        for key, value in (("user.name", "Other Machine"), ("user.email", "other@example.invalid")):
            mod.git(competitor, "config", key, value)
        original_git = mod.git
        raced = []

        def racing_git(path, *args, **kwargs):
            if args[:3] == ("push", "origin", "HEAD:main") and not raced:
                raced.append(True)
                (competitor / "other-machine.txt").write_text("new result\n")
                original_git(competitor, "add", "other-machine.txt")
                original_git(competitor, "commit", "-m", "Other machine result")
                original_git(competitor, "push", "origin", "HEAD:main")
            return original_git(path, *args, **kwargs)

        mod.git = racing_git
        try:
            mod.publish(repo, "iql_ext_csv", (scores, manifest), "ext_csv")
        finally:
            mod.git = original_git
        self.assertTrue(raced)
        self.assertEqual(mod.git(repo, "rev-parse", "HEAD").stdout, original_head)
        self.assertEqual(mod.git(repo, "diff", "--cached").stdout, staged)
        self.assertEqual((repo / "trainer.py").read_text(), "uncommitted active training change\n")
        mod.git(repo, "fetch", "origin", "main")
        self.assertEqual(mod.git(repo, "show", "origin/main:other-machine.txt").stdout, "new result\n")
        self.assertEqual(mod.git(repo, "show", "origin/main:sweep_results/iql_ext_csv/scores_verified.csv").stdout, scores)
        self.assertNotIn("unrelated.txt", mod.git(repo, "ls-tree", "--name-only", "origin/main").stdout)
        self.assertEqual(len(mod.git(repo, "worktree", "list", "--porcelain").stdout.split("worktree ")), 2)


if __name__ == "__main__":
    unittest.main()
