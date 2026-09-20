import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("main_results", Path(__file__).resolve().parents[1] / "scripts/build_main_results.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class MainResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, rows, status=None):
        folder = self.root / "sweep_results" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "STATUS.json").write_text(json.dumps(status or {}))
        with (folder / "scores.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def row(self, **extra):
        return dict(dict(env="hopper-medium-v2", tau="0.4", K=4, seed=0,
                         variant="awr_gaussian_fr", geometry="fr", eval_mode="mean",
                         policy="mpi", hop=4, step=1000000, d4rl_score=60), **extra)

    def test_filters_and_k4_merge(self):
        self.write("iql_new", [self.row(), self.row(hop=1, d4rl_score=99),
                  self.row(seed=1, step=900000), self.row(seed=2, eval_mode="sample"),
                  self.row(seed=3, variant="qbc_gaussian_w2", geometry="fr"),
                  self.row(seed=1, variant="qbc_deterministic_w2", geometry="w2")])
        selected, *_ = mod.collect(self.root)
        self.assertEqual(len(selected), 1)
        self.assertIn(("AWR–FR", "hopper-medium-v2", .4, 4, 0), selected)

    def test_baseline_h_not_full_t_and_standalone_priority(self):
        status = {"schema": "iql_actor_geometry_v5_consistent_gaussian"}
        self.write("iql_k4_t6", [self.row(policy="baseline", hop=1, d4rl_score=99)], status)
        self.write("iql_new", [self.row(K=1, hop=1, policy="baseline", d4rl_score=30)])
        self.write("iql_gauss_v5", [self.row(policy="baseline", hop=1, seed=1)],
                   {"schema": "iql_actor_geometry_v6_baseline_h"})
        selected, *_ = mod.collect(self.root)
        self.assertEqual(len(selected), 1)
        self.assertEqual(next(iter(selected.values()))["score"], 30)

    def test_conflicts_withheld_not_best_score(self):
        self.write("iql_a", [self.row()])
        self.write("iql_b", [self.row(d4rl_score=90)])
        selected, _, warnings, _, audit = mod.collect(self.root)
        self.assertFalse(selected)
        self.assertTrue(warnings)
        self.assertTrue(all(r["conflict"] for r in audit))

    def test_duplicate_not_double_counted(self):
        self.write("iql_a", [self.row(), self.row()])
        selected, *_ = mod.collect(self.root)
        self.assertEqual(len(selected), 1)

    def test_unknown_new_export_and_nan_rejected(self):
        self.write("iql_new", [self.row(step=""), self.row(seed=1, d4rl_score="nan")])
        selected, _, warnings, *_ = mod.collect(self.root)
        self.assertFalse(selected)
        self.assertEqual(len(warnings), 2)

    def test_machine_and_std(self):
        self.assertEqual(mod.machine({}, {"source": "/home/choi/results"}), "choi (계정)")
        self.assertEqual(mod.machine({"machine": "gpu-host"}, {}), "gpu-host")
        rows = [(0, dict(score=1, machine="a", source="a.csv", line=2)),
                (1, dict(score=3, machine="b", source="b.csv", line=2))]
        self.assertIn("2.00 ± 1.41", mod.cell(rows))
        self.assertNotIn("±", mod.cell(rows[:1]))

    def test_td3_complete_k_filter_preserves_iql(self):
        section = "TD3+BC MPI · Implicit"
        data = {(section, "env", .4, k, 0): {} for k in range(1, 5)}
        data[(section, "env", .1, 1, 0)] = {}
        data[("TD3+BC MPI · Explicit", "env", .4, 2, 0)] = {}
        data[("AWR–FR", "env", .1, 4, 0)] = {}
        visible = mod.visible_results(data)
        self.assertEqual(len(visible), 5)
        self.assertNotIn((section, "env", .1, 1, 0), visible)
        self.assertIn(("AWR–FR", "env", .1, 4, 0), visible)

    def test_raw_td3_matrix_and_deterministic_render(self):
        for k in range(1, 5):
            folder = self.root / f"sweep_results/K={k}/Imp"
            folder.mkdir(parents=True)
            (folder / "seed0.csv").write_text("tau,hopper-medium-v2\n0.4,42\n")
        mod.build(self.root)
        first = (self.root / "MAIN_RESULTS.md").read_bytes()
        mod.build(self.root)
        self.assertEqual(first, (self.root / "MAIN_RESULTS.md").read_bytes())
        self.assertIn("42.00", first.decode())
        self.assertIn("미확인", first.decode())


if __name__ == "__main__":
    unittest.main()
