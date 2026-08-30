#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, re
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

summary={r['method']:r for r in read_csv(ROOT/'data'/'aggregate_summary.csv')}
expected={
 'TD3+BC':(25.97,35,68),
 'MPI-Prox K=2':(42.02,25,53),
 'MPI-Prox K=3':(52.78,17,41),
 'MPI-Prox K=4':(63.6104,9,29),
 'MPI-Lin K=2':(36.68,29,58),
 'MPI-Lin K=3':(43.91,24,53),
}
for m,(mean,cells,runs) in expected.items():
    r=summary[m]
    assert math.isclose(float(r['mean_T_ge_4']),mean,abs_tol=.005), (m,r)
    assert int(r['collapsed_cells_lt20_of63'])==cells
    assert int(r['collapsed_runs_lt20_of126'])==runs


# Recompute the central TD3/P3/P4 claims from frozen raw matrices.
GRID=(0.05,0.1,0.2,0.4,0.7,1.5,2.5,4.0,7.0,10.0,12.0,14.0,17.0,20.0)
def load_pair(prefix):
    arrays=[]; envs_ref=None
    for seed in (0,1):
        with (ROOT/'data'/'raw'/f'{prefix}_seed{seed}.csv').open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        envs=[x for x in rows[0] if x!='tau']
        if envs_ref is None: envs_ref=envs
        assert envs==envs_ref
        by_t={float(r['tau']):[float(r[e]) for e in envs] for r in rows if float(r['tau']) in GRID}
        assert set(by_t)==set(GRID)
        arrays.append([[*by_t[t]] for t in GRID])
    import numpy as np
    return envs_ref,np.asarray(arrays,dtype=float)
envs,raw_td3=load_pair('td3')
_,raw_p3=load_pair('prox3')
_,raw_p4=load_pair('prox4')
high=np.asarray(GRID)>=4
def raw_stats(a):
    values=a[:,high,:]
    cell=values.mean(axis=0)
    return float(values.mean()),int((cell<20).sum()),int((values<20).sum())
assert all(math.isclose(a,b,abs_tol=.005) for a,b in zip(raw_stats(raw_td3),(25.9708,35,68)))
assert all(math.isclose(a,b,abs_tol=.005) for a,b in zip(raw_stats(raw_p3),(52.7833,17,41)))
assert all(math.isclose(a,b,abs_tol=.005) for a,b in zip(raw_stats(raw_p4),(63.6104,9,29)))
cell3=raw_p3[:,high,:].mean(axis=0).reshape(-1)
cell4=raw_p4[:,high,:].mean(axis=0).reshape(-1)
assert int(((cell3<20)&(cell4>=20)).sum())==8
assert int(((cell3>=20)&(cell4<20)).sum())==0
assert math.isclose(float((cell4-cell3).mean()),10.827115873,abs_tol=1e-9)
assert math.isclose(float(np.median(cell4-cell3)),0.318625,abs_tol=.01)

unc={r['comparison']:r for r in read_csv(ROOT/'data'/'task_cluster_uncertainty.csv')}
assert math.isclose(float(unc['K4-K1']['mean_score_difference']),37.63964206349206,abs_tol=1e-10)
assert float(unc['K4-K1']['mean_score_ci_low'])>0
assert math.isclose(float(unc['K4-K3']['raw_collapse_risk_ci_high']),0.0,abs_tol=1e-12)

text='\n'.join((ROOT/p).read_text(encoding='utf-8') for p in ['paper.tex',*sorted(str(x.relative_to(ROOT)) for x in (ROOT/'sections').glob('*.tex')),'supplement.tex'])
required=[
 '63.61','35/63','9/63','37.64','20.5,54.3','10.83','88/90','36/90',
 '50{,}000','sample-anchored next-state target-action displacement',
 'one Adam update','locally Lipschitz Hessian','5/8','post hoc',
 'halfcheetah-expert-v2','sec:technical_end'
]
for token in required:
    assert token in text, f'missing manuscript token: {token}'
for forbidden in ['medium-expert-v2','required controlled rerun','critic initialization confound','directly measured target-policy displacement']:
    assert forbidden not in text, f'forbidden stale wording: {forbidden}'

# The exact implementation-specific first-hop scale distinction must be present.
assert 'z^{\\rm P}_1&=\\mu^{\\rm pre}_1' in text
assert 'r^{\\rm L}_1&=z^{\\rm L}_1=a' in text

# Submission-facing compact pack must not leak local absolute paths.
for path in list((ROOT/'audit_pack').glob('*'))+list((ROOT/'data').glob('*')):
    if path.is_file() and path.suffix in {'.md','.json','.csv','.txt','.py'}:
        s=path.read_text(encoding='utf-8',errors='ignore')
        assert '/home/' not in s and '/mnt/' not in s, f'absolute path leak: {path}'

print('BUNDLE SOURCE CHECKS PASSED')
