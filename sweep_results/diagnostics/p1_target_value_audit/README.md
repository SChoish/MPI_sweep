# P1 target-value audit (compact)

External full bundle (including `raw/**/*.npz` and `common_batches/`):
`/home/ext_csv/mpi_sweep_lab/p1-audit-20260902b/`

This directory retains the create-only scientific CSVs plus
`MANIFEST.json` / `VERIFY.json` for release inspection. Do not treat path
completeness alone as historical training-identity proof.

`RUN_STATUS.json` is the immutable pre-verifier create-only record and therefore
remains `analysis_complete_unverified`. The final artifact-arithmetic decision
is `VERIFY.json`: `pass`, `artifact_integrity_pass`, and
`scientific_inclusion_gate_pass` are all true. The compact verifier does not
reexecute the neural-network checkpoints, and the historical training revision
is not embedded in those checkpoints.

## Interpretation used in the manuscript

Under the matched TD3+BC target critic and identical smoothing noise, the
operational TD-target perturbation is lower than TD3+BC in 86/90 P3 and 88/90
P4 cells; median RMS ratios are `.717` and `.617`, and all nine task medians
are below one. Same-next-state geometry is complete for all 90 P3 and all 90
P4 cells. Relative to the paired next dataset action, the deployed actor lies
farther than the first Polyak actor in every cell, with median final-to-target
RMS ratios `1.297` and `1.365`.

Do not use absolute arithmetic means as typical-effect summaries. Extreme
finite critic magnitudes in collapsed cells dominate them. The common-critic result
is an action-side critic-space sensitivity, not a critic-error or causal-return
estimate. Comparator residuals remain a sampled post-hoc premise check, not an
implementation certificate or an estimate of the theoretical
`epsilon_k`.
