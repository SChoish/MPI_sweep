# IQL AWR–FR trajectory evidence (2026-09-24)

This is an **observational diagnostic log**, separate from the score publisher and
`MAIN_RESULTS.md`. The endpoint scores below are recomputed from the linked
repository CSV files at `main` commit `5e174d5d`. The trajectory and checkpoint
findings were reported by the operators of the four hosts on 2026-09-24; the
original `metrics.jsonl`, checkpoints, and trajectory CSVs are **not in this
repository**. Do not describe those trajectories as independently audited.

## Published endpoint scores

Normalized D4RL mean-action score at the final actor, four training seeds per
cell. Mean ± sample standard deviation (`ddof=1`); no best-T or best-seed
selection. `baseline` is the separately labeled first actor, not an MPI hop.

| Environment | T | K | Machine / source | Final score |
| --- | ---: | ---: | --- | ---: |
| HalfCheetah-expert | 1.5 | 2 | ext_csh / [`scores.csv`](../sweep_results/iql_awr_fr/scores.csv) | −2.914 ± 0.721 |
| HalfCheetah-expert | 1.5 | 3 | ext_csh / [`scores.csv`](../sweep_results/iql_awr_fr/scores.csv) | −2.611 ± 0.202 |
| HalfCheetah-expert | 1.5 | 4 | shchoi / [`scores_long.csv`](../sweep_results/iql_k4_t6/scores_long.csv) | −2.553 ± 0.016 |
| Hopper-expert | 1.5 | 2 | svcho / [`scores_verified.csv`](../sweep_results/iql_awr_fr/scores_verified.csv) | 0.678 ± <0.001 |
| Hopper-expert | 1.5 | 3 | svcho / [`scores_verified.csv`](../sweep_results/iql_awr_fr/scores_verified.csv) | 0.678 ± <0.001 |
| Hopper-expert | 1.5 | 4 | shchoi / [`scores_long.csv`](../sweep_results/iql_k4_t6/scores_long.csv) | 0.677 ± <0.001 |
| Hopper-medium-replay | 4 | 1 | ext_csh / [`scores.csv`](../sweep_results/iql_awr_fr/scores.csv), `policy=baseline, hop=1` | 82.001 ± 21.930 |
| Hopper-medium-replay | 4 | 2 | ext_csh / [`scores.csv`](../sweep_results/iql_awr_fr/scores.csv) | 100.698 ± 1.448 |
| Hopper-medium-replay | 4 | 3 | ext_csh / [`scores.csv`](../sweep_results/iql_awr_fr/scores.csv) | 100.743 ± 0.370 |
| Hopper-medium-replay | 4 | 4 | shchoi / [`scores_long.csv`](../sweep_results/iql_k4_t6/scores_long.csv) | 100.630 ± 0.870 |

Within the shchoi K=4 runs, the **same-run** `policy=baseline, hop=1` averages
93.494 ± 0.765 (HalfCheetah-expert), 93.232 ± 7.748 (Hopper-expert), and
72.416 ± 26.910 (Hopper-medium-replay). These baselines should not be pooled
with the separately trained ext_csh K=1 runs.

## Operator-reported trajectories

Here `cap` means the *final hop's* logged minibatch-mean `std_mean` first exceeds
`0.95 exp(2)`; it is **not** a per-action-coordinate saturation fraction. The
`after 900k` fraction is the fraction of logged late minibatches satisfying
the same threshold. `last cap` identifies the last logged qualifying step,
not the point after which a run never revisits the cap. All listed runs reached
1,000,000 steps. Operators confirmed `args.q_action_transform=identity` and
`args.log_std_max=2.0` (their original extraction script incorrectly labeled
the transform `UNKNOWN` because it looked only at top-level `config.json`).

| Host; environment; T; K | Seeds / first cap steps | Last cap / late fraction | Endpoint std; sampled-action OOB | Trajectory file on host |
| --- | --- | --- | --- | --- |
| ext_csh; HC-expert; 1.5; 2 | s0–3: 5440, 4992, 5120, 4160 | 1M / 1.000 | 7.389; 0.893–0.897 | `/home/ext_csh/awr_fr_trajectory_ext_csh-box.csv` |
| ext_csh; HC-expert; 1.5; 3 | s0–3: 6912, 6464, 6464, 6144 | 1M / 1.000 | 7.389; 0.893–0.897 | same file |
| ext_csh; Hopper-mr; 4; 1 | no hop-2 or later actor | none / 0 | 0.286–0.297; 0.113–0.118 | same file |
| ext_csh; Hopper-mr; 4; 2 | s0–3: 2176, 2240, 2048, 1664 | s0–3: 32448, 28608, 20096, 35456 / 0 | 0.615–0.800; 0.257–0.285 | same file |
| ext_csh; Hopper-mr; 4; 3 | s0–3: 2560, 2816, 2944, 2176 | s0–3: 34560, 33536, 31744, 34752 / 0 | 0.527–0.723; 0.234–0.275 | same file |
| svcho; Hopper-expert; 1.5; 2 | s0–3: 3456, 4096, 4736, 5120 | 1M / 1.000 | 7.389; 0.889–0.898 | `/home/svcho/awr_fr_trajectory_iisl-server02.csv` |
| svcho; Hopper-expert; 1.5; 3 | s0–3: 4864, 5952, 5248, 6272 | 1M / 1.000 | 7.389; 0.889–0.898 | same file |
| shchoi; HC-expert; 1.5; 4 | 4 diagnosed copies: 7500–8800 | 1M / 1.000 | 7.389; ~0.89 | `/home/shchoi/awr_fr_trajectory_iisl-server04.csv` |
| shchoi; Hopper-expert; 1.5; 4 | 4 diagnosed copies: 6200–7300 | 1M / 1.000 | 7.389; ~0.89 | same file |
| shchoi; Hopper-mr; 4; 4 | 4 diagnosed copies: 3700–5000 | 26000–59000 / 0 | ~0.43–0.53; ~0.23 | same file |

`shchoi` reported 24 matching directories (two copies per cell and seed).
Only 12 have hop diagnostics; the other 12 returned `NO_HOP_DIAGNOSTIC`.
Missing diagnostics are **not** evidence of absent saturation. Run IDs linking
the 12 diagnostic copies to the published K=4 rows were not supplied, so the
score table and diagnostic summary must not be joined per seed or averaged
across copies until that mapping is established.

The separate `choi` Hopper-expert T=1.5, K=3, v6-baseline-h runs reported
first-cap steps 4800, 6016, 5824, 5760 for seeds 0–3. Seeds 0–2 stayed
at the cap through 1M (late fraction 1.000); seed 3 left the
cap around 560k (late fraction 0.000, final std ~2.05).
The separately published [v6 `scores.csv`](../sweep_results/iql_awr_fr_hopper_expert_k3/scores.csv)
now reports 16/16 completed runs with 50 evaluation episodes per run; for
T=1.5 the final seed-0–3 scores are 0.677481, 0.677486, 0.677452, 0.742194.
The local trajectory file is `/home/choi/awr_fr_trajectory_choi-ubuntu.csv`.
These v6 runs are **not** additional v5 replicates and are not pooled into the
main dashboard or the v5 score table above; the log index currently marks this
step-less export as `0` selected for main results.

## Interpretation and next measurement

Early cap entry also occurs in the successful Hopper-medium-replay runs.
Persistent cap entry and low return co-occur in the two expert environments;
the `choi` seed-3 exception shows that late cap exit alone is not sufficient
for high return. The observations do **not** establish that Fisher–Rao geometry
alone caused failure. Under `identity`, the critic consumes raw Gaussian
samples outside the environment action range; OOB frequency can also be a
consequence of a large std. Compare raw-versus-clipped critic values on the
**same saved critic, states, and samples** before inferring an OOD mechanism.

This diagnostic log is descriptive and does not alter `scores.csv`,
`scores_long.csv`, the dashboard's selection policy, or the paper.
