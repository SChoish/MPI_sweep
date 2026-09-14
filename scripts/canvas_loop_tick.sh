#!/usr/bin/env bash
set -uo pipefail
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
"$PY" /home/ext_csh/MPI_sweep/scripts/rewrite_mpi_k123_s23_canvas.py
echo "AGENT_LOOP_TICK_mpi_exp_tau40 {\"prompt\":\"Refresh mpi-k123-s23 canvas for Exp K=2/3 high-τ {24,28,34,40}. Report exp2/exp3 1M progress, live trains, queue ok/fail, ETA.\"}"
