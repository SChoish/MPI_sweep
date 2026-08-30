# Source manifest

## Repository basis

- current `aistats26_manuscript/` BAR manuscript and supplement;
- `train_td3bc.py`, `d4rl_data.py`, and sweep launch code;
- complete sweep matrices under `sweep_results/K=*/`;
- compact and raw-derived diagnostic audit under `sweep_results/diagnostics/audit_pack/`;
- user-provided external audit report reproduced in the conversation.

## Frozen score inputs in this bundle

- aggregate summaries and budget means for TD3+BC, BAR-Prox K=2/3/4, and BAR-Lin K=2/3;
- raw two-seed matrices for TD3+BC, BAR-Prox K=3, and BAR-Prox K=4 used to recompute task-cluster uncertainty;
- K=1 replication seeds 2-3;
- threshold and per-dataset summaries.

## Diagnostic status

The repository audit re-aggregated 2,135 local inputs totaling approximately 3.18GB. This bundle contains only sanitized compact outputs. It does not duplicate raw checkpoints/NPZs and does not claim that training or simulator rollout was rerun.

## Source conflict resolved explicitly

The external report suggested names such as `*-medium-expert-v2`. The released data loader instead declares `halfcheetah-expert-v2`, `hopper-expert-v2`, and `walker2d-expert-v2`. The manuscript follows the implementation and uses those exact identifiers.
