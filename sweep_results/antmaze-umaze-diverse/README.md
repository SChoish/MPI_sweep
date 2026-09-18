# antmaze-umaze-diverse T-sweep (host choi)

- Algorithm: TD3+BC + MPI (Imp), K=1..4
- Env: `antmaze-umaze-diverse-v2` (`d4rl_score`, 100 eval episodes @ 1M)
- Tau grid: 0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4, 7, 10, 12, 14, 17, 20
- Auto-refreshed from local `results/antmaze_umaze_diverse_t_sweep` (eval/config only).
- Current scored cells: **146**

## Files
- `K=*/Imp/seed{0..3}.csv`
- `scores_long.csv` / `summary_by_T.csv`
- `raw/` — per-run eval artifacts
- `queue_amud_k1234_t_sweep.log`
