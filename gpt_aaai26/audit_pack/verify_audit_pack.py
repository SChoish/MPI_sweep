#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'claim_summary.json').read_text())
assert summary['integrity']['target_exposure_rows']==270
assert summary['integrity']['failure_diagnostic_rows']==180
assert summary['integrity']['simulator_states']==720
assert summary['headline']['p3_vs_td3_target_lower']==[88,90]
assert math.isclose(summary['headline']['p3_vs_td3_target_median_ratio'],0.6576946519,abs_tol=1e-10)
assert summary['headline']['route_primary_positive']==[8,8]
assert summary['headline']['route_final_checkpoint_positive']==[5,8]
with (ROOT/'task_cluster_uncertainty.csv').open(newline='') as f:
    rows={r['comparison']:r for r in csv.DictReader(f)}
assert math.isclose(float(rows['K4-K1']['mean_score_difference']),37.63964206349206,abs_tol=1e-12)
assert math.isclose(float(rows['K4-K3']['raw_collapse_risk_ci_high']),0.0,abs_tol=1e-12)
print('AUDIT PACK CHECKS PASSED')
