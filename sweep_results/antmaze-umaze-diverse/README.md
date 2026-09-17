# antmaze-umaze-diverse T-sweep (host choi)

- Algorithm: TD3+BC + MPI (Imp), K=1..4 planned; **K=1 partial** on disk.
- Env: `antmaze-umaze-diverse-v2` (D4RL success% as `d4rl_score`, 100 eval episodes @ 1M).
- Tau grid: 0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4, 7, 10, 12, 14, 17, 20
- Progress: **40 / 56** K=1 cells scored; aborted by disk full (Errno 28) during tau=12.
- Missing: tau7 seed1; tau12 seeds1–3; all of tau 14/17/20.
- K=2–4 not started.

## Files
- `K=1/Imp/seed{0..3}.csv` — compact final-score matrices
- `scores_long.csv` / `summary_by_T.csv`
- `raw/` — per-run `eval.csv` (+ config/provenance), no checkpoints
- `queue_amud_k1234_t_sweep.log` — launcher log from choi

Regenerate matrices from local `results/antmaze_umaze_diverse_t_sweep` after more runs finish.
