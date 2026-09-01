#!/usr/bin/env bash
set -uo pipefail
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
"$PY" /home/ext_csh/MPI_sweep/scripts/rewrite_mpi_k123_s23_canvas.py
echo "AGENT_LOOP_TICK_mpi_canvas {\"prompt\":\"Refresh mpi-k123-s23 canvas from MPI_sweep results (seeds 2,3). Run rewrite script if needed, briefly report 1M progress / phase / live trains.\"}"
