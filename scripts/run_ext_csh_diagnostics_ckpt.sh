#!/usr/bin/env bash
# Incremental post-hoc diagnostics for ext_csh (only newly finished 1M cells).
set -uo pipefail

ROOT=/home/ext_csh/MPI_sweep
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
PKG=$ROOT/scripts/diagnostics
HOST=ext_csh
OUT=$ROOT/sweep_results/diagnostics/hosts/$HOST
MIRROR=$ROOT/logs/_diag_results_root
DATA_DIR=$ROOT/data
CANON_TAU=0.05

# Discover new cells vs existing diagnostics CSVs; print narrowed dump args.
mapfile -t PLAN < <("$PY" - <<'PY'
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/home/ext_csh/MPI_sweep")
OUT = ROOT / "sweep_results/diagnostics/hosts/ext_csh"
ENVS = [
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "walker2d-medium-v2",
    "walker2d-expert-v2",
]
TAUS = [4.0, 7.0, 10.0, 14.0, 20.0]
SEEDS = [2, 3]
METHOD_DIRS = {
    "mpi2": ROOT / "results/mpi2_s23",
    "mpi3": ROOT / "results/mpi3_s23",
}

def tau_token(t: float) -> str:
    return f"{t:g}"

def load_done(path: Path) -> set[tuple[str, float, int, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {
        (r["env"], float(r["tau"]), int(r["seed"]), r["method"])
        for r in rows
        if r.get("env") and r.get("method")
    }

done_geo = load_done(OUT / "matched_geometry/run_diagnostics.csv")
done_exp = load_done(OUT / "target_policy_exposure/target_policy_exposure_summary.csv")

ready: list[tuple[str, float, int, str]] = []
for method, result_dir in METHOD_DIRS.items():
    if not result_dir.is_dir():
        continue
    for env in ENVS:
        for tau in TAUS:
            for seed in SEEDS:
                tag = f"{env}_tau{tau_token(tau)}_{method}_seed{seed}"
                ckpt = result_dir / tag / "params_1000000.pkl"
                if not ckpt.is_file():
                    continue
                key = (env, float(tau), int(seed), method)
                if key in done_geo and key in done_exp:
                    continue
                ready.append(key)

if not ready:
    print("NONE")
else:
    envs = sorted({e for e, _, _, _ in ready})
    taus = sorted({t for _, t, _, _ in ready})
    seeds = sorted({s for _, _, s, _ in ready})
    methods = sorted({m for _, _, _, m in ready})
    print("NEW", len(ready))
    print("ENVS", *envs)
    print("TAUS", *[f"{t:g}" for t in taus])
    print("SEEDS", *seeds)
    print("METHODS", *methods)
    for env, tau, seed, method in ready:
        print(f"CELL {method} {env} tau={tau:g} seed={seed}")
PY
)

if [[ ${#PLAN[@]} -eq 0 || "${PLAN[0]}" == "NONE" ]]; then
  echo "[diag] no new 1M cells in diagnostic grid; skip"
  exit 0
fi

echo "[diag] plan:"
printf '%s\n' "${PLAN[@]}"

ENVS=()
TAUS=()
SEEDS=()
METHODS=()
for line in "${PLAN[@]}"; do
  set -- $line
  case "$1" in
    ENVS) shift; ENVS=("$@") ;;
    TAUS) shift; TAUS=("$@") ;;
    SEEDS) shift; SEEDS=("$@") ;;
    METHODS) shift; METHODS=("$@") ;;
  esac
done

if [[ ${#METHODS[@]} -eq 0 ]]; then
  echo "[diag] empty methods after parse; skip"
  exit 0
fi

mkdir -p "$MIRROR/results_qnorm"
ln -sfn "$ROOT/results/mpi2_s23" "$MIRROR/results_mpi2"
ln -sfn "$ROOT/results/mpi3_s23" "$MIRROR/results_mpi3"
for env in "${ENVS[@]}"; do
  for s in "${SEEDS[@]}"; do
    src="$ROOT/results/mpi1_s23/${env}_tau${CANON_TAU}_mpi1_seed${s}"
    dst="$MIRROR/results_qnorm/${env}_tau${CANON_TAU}_seed${s}"
    [[ -d $src ]] && ln -sfn "$src" "$dst"
  done
done

mkdir -p "$OUT/matched_geometry" "$OUT/target_policy_exposure"
export LAB_ROOT=$ROOT
export RESULTS_ROOT=$MIRROR
export DATA_DIR
export PYTHONPATH=$ROOT:$PKG:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu

echo "[diag] host=$HOST methods=${METHODS[*]} taus=${TAUS[*]} seeds=${SEEDS[*]} envs=${#ENVS[@]}"

# Origin dump scripts overwrite CSVs and have no --merge. Dump to tmp, then
# union into the published host files by (env, tau, seed, method).
TMP="$ROOT/logs/_diag_tmp_ext_csh"
rm -rf "$TMP"
mkdir -p "$TMP/matched_geometry" "$TMP/target_policy_exposure"

"$PY" -u "$PKG/dump_mpi_frontier_geometry.py" \
  --root "$RESULTS_ROOT" --data-dir "$DATA_DIR" \
  --out-dir "$TMP/matched_geometry" \
  --seeds "${SEEDS[@]}" --methods "${METHODS[@]}" --taus "${TAUS[@]}" \
  --envs "${ENVS[@]}" \
  --canonical-critic-tau "$CANON_TAU"

"$PY" -u "$PKG/dump_target_policy_exposure.py" \
  --root "$RESULTS_ROOT" --data-dir "$DATA_DIR" \
  --out-dir "$TMP/target_policy_exposure" \
  --seeds "${SEEDS[@]}" --methods "${METHODS[@]}" --taus "${TAUS[@]}" \
  --envs "${ENVS[@]}"

"$PY" - <<PY
from pathlib import Path
import csv

host = Path("$OUT")
tmp = Path("$TMP")

def read(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))

def write(path: Path, rows: list[dict]):
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def key_run(row: dict) -> tuple:
    return (row["env"], float(row["tau"]), int(row["seed"]), row["method"])

def key_hop(row: dict) -> tuple:
    return (
        row["env"], float(row["tau"]), int(row["seed"]), row["method"],
        row.get("critic_scope", ""), row.get("hop", ""),
    )

def union(old, new, keyfn):
    out = {keyfn(r): r for r in old}
    for row in new:
        out[keyfn(row)] = row
    return list(out.values())

geo_old = host / "matched_geometry"
geo_new = tmp / "matched_geometry"
exp_old = host / "target_policy_exposure"
exp_new = tmp / "target_policy_exposure"

write(
    geo_old / "run_diagnostics.csv",
    union(read(geo_old / "run_diagnostics.csv"), read(geo_new / "run_diagnostics.csv"), key_run),
)
write(
    geo_old / "hop_geometry.csv",
    union(read(geo_old / "hop_geometry.csv"), read(geo_new / "hop_geometry.csv"), key_hop),
)
write(
    exp_old / "target_policy_exposure_summary.csv",
    union(
        read(exp_old / "target_policy_exposure_summary.csv"),
        read(exp_new / "target_policy_exposure_summary.csv"),
        key_run,
    ),
)
print("[merge] geometry/exposure unioned into", host)
PY

"$PY" -u "$PKG/summarize_mpi_frontier_geometry.py" "$OUT/matched_geometry"

echo "[diag] done → $OUT"
