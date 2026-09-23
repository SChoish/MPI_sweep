import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_log_index as index


class LogIndexTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def folder(self, name, status, rows=None):
        folder = self.root / "sweep_results" / name
        folder.mkdir(parents=True)
        (folder / "STATUS.json").write_text(json.dumps(status))
        if rows:
            with (folder / "scores_verified.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        return folder

    def test_status_progress_is_not_a_score_and_source_account_is_labeled(self):
        self.folder("iql_det", dict(variant="qbc_deterministic_w2", planned=12,
                                    counts=dict(complete=5), save_dir="/raid/ext_csv/results"))
        entry = index.inventory(self.root)["sources"][0]
        self.assertEqual((entry["csv_rows"], entry["main_selected"]), (0, 0))
        self.assertEqual((entry["complete_reported"], entry["planned_reported"]), (5, 12))
        self.assertEqual(entry["machine"], "ext_csv (계정)")
        self.assertIn("본문 제외", entry["methods"][0])

    def test_uses_main_parser_for_final_mean_and_excludes_gaussian_fr(self):
        self.folder("iql_verified", dict(variant="qbc_gaussian_w2", counts=dict(complete=2)),
                    [dict(env="hopper-medium-v2", tau="0.4", K=2, seed=0,
                          variant="qbc_gaussian_w2", geometry="w2",
                          gaussian_qbc_mode="stochastic", policy="mpi", hop=2,
                          eval_mode="mean", step=1000000, status="completed",
                          d4rl_score=42, machine="ext_csv", hostname="DGX-H200-02"),
                     dict(env="hopper-medium-v2", tau="0.4", K=2, seed=1,
                          variant="qbc_gaussian_w2", geometry="w2",
                          gaussian_qbc_mode="stochastic", policy="mpi", hop=1,
                          eval_mode="mean", step=1000000, status="completed",
                          d4rl_score=99, machine="ext_csv", hostname="DGX-H200-02")])
        self.folder("iql_fr", dict(variant="qbc_gaussian_w2", geometry="fr", done=1),
                    [dict(env="hopper-medium-v2", tau="0.4", K=1, seed=0,
                          variant="qbc_gaussian_w2", geometry="fr", policy="baseline",
                          hop=1, eval_mode="mean", step=1000000, d4rl_score=100)])
        data = index.inventory(self.root)
        rows = {Path(item["source"]).name: item for item in data["sources"]}
        self.assertEqual(rows["iql_verified"]["csv_rows"], 2)
        self.assertEqual(rows["iql_verified"]["main_selected"], 1)
        self.assertEqual(rows["iql_verified"]["machine"], "DGX-H200-02, ext_csv")
        self.assertEqual(rows["iql_fr"]["main_selected"], 0)
        self.assertEqual(rows["iql_fr"]["methods"], [index.OTHER])
        self.assertEqual(data["main_selected_by_method"], {"Gaussian Q+BC–W2": 1})

    def test_document_links_and_regeneration_are_stable(self):
        self.folder("iql_det", dict(variant="qbc_deterministic_w2", done=5))
        index.build(self.root)
        text = (self.root / "sweep_results/LOG_INDEX.md").read_text()
        data = (self.root / "reports/log_index.json").read_bytes()
        self.assertIn("](iql_det/STATUS.json)", text)
        self.assertIn("](../MAIN_RESULTS.md)", text)
        self.assertNotIn("](sweep_results/iql_det/", text)
        index.build(self.root)
        self.assertEqual(data, (self.root / "reports/log_index.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
