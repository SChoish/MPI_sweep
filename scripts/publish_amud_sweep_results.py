#!/usr/bin/env python3
"""Refresh sweep_results/antmaze-umaze-diverse from local result dirs (no checkpoints)."""

from __future__ import annotations

import csv
import re
import shutil
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "antmaze_umaze_diverse_t_sweep"
OUT = ROOT / "sweep_results" / "antmaze-umaze-diverse"
QUEUE_LOG = Path("/home/choi/logs/mpi_amud_k1234_t_sweep.log")
ENV = "antmaze-umaze-diverse-v2"
TAU_GRID = [0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4, 7, 10, 12, 14, 17, 20]
TAG_RE = re.compile(
    rf"{re.escape(ENV)}_tau(?P<tau>[^_]+)_mpi(?P<k>\d+)_seed(?P<seed>\d+)$"
)


def latest_score(eval_csv: Path) -> float | None:
    if not eval_csv.is_file():
        return None
    rows = list(csv.DictReader(eval_csv.open(encoding="utf-8")))
    if not rows:
        return None
    raw = rows[-1].get("d4rl_score")
    if raw in (None, ""):
        return None
    return float(raw)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    by_k: dict[int, dict[int, dict[float, float | None]]] = {}
    long_rows: list[dict[str, object]] = []

    if RESULTS.is_dir():
        for d in sorted(RESULTS.iterdir()):
            if not d.is_dir():
                continue
            m = TAG_RE.match(d.name)
            if not m:
                continue
            tau = float(m.group("tau"))
            k = int(m.group("k"))
            seed = int(m.group("seed"))
            score = latest_score(d / "eval.csv")
            by_k.setdefault(k, {s: {} for s in range(4)})
            for s in range(4):
                by_k[k].setdefault(s, {})
            by_k[k][seed][tau] = score

            status = "available" if score is not None else "missing"
            long_rows.append(
                {
                    "method": f"I-MART-{k}" if k > 1 else "TD3+BC",
                    "K": k,
                    "integrator": "Imp",
                    "T": f"{tau:g}",
                    "h": f"{tau:g}",
                    "seed": seed,
                    "environment": ENV,
                    "score": "" if score is None else f"{score:.4g}",
                    "status": status,
                    "source_csv": f"K={k}/Imp/seed{seed}.csv",
                }
            )

            if score is not None:
                raw = OUT / "raw" / d.name
                raw.mkdir(parents=True, exist_ok=True)
                shutil.copy2(d / "eval.csv", raw / "eval.csv")
                for name in ("config.json", "PROVENANCE.json"):
                    src = d / name
                    if src.is_file():
                        shutil.copy2(src, raw / name)

    if not by_k:
        by_k[1] = {s: {} for s in range(4)}

    for k, by_seed in sorted(by_k.items()):
        kdir = OUT / f"K={k}" / "Imp"
        kdir.mkdir(parents=True, exist_ok=True)
        for seed in range(4):
            path = kdir / f"seed{seed}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["tau", ENV], lineterminator="\n")
                w.writeheader()
                for tau in TAU_GRID:
                    sc = by_seed.get(seed, {}).get(tau)
                    w.writerow(
                        {
                            "tau": f"{tau:g}",
                            ENV: "" if sc is None else f"{sc:.4g}",
                        }
                    )

    # Fill missing long rows for declared K×tau×seed grid (at least K=1).
    ks = sorted(by_k) or [1]
    have = {(int(r["K"]), float(r["T"]), int(r["seed"])) for r in long_rows}
    for k in ks:
        for tau in TAU_GRID:
            for seed in range(4):
                if (k, tau, seed) in have:
                    continue
                long_rows.append(
                    {
                        "method": f"I-MART-{k}" if k > 1 else "TD3+BC",
                        "K": k,
                        "integrator": "Imp",
                        "T": f"{tau:g}",
                        "h": f"{tau:g}",
                        "seed": seed,
                        "environment": ENV,
                        "score": "",
                        "status": "missing",
                        "source_csv": f"K={k}/Imp/seed{seed}.csv",
                    }
                )
    long_rows.sort(key=lambda r: (int(r["K"]), float(r["T"]), int(r["seed"])))

    with (OUT / "scores_long.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(long_rows)

    sum_rows: list[dict[str, object]] = []
    for k in ks:
        by_seed = by_k[k]
        for tau in TAU_GRID:
            vals = [
                by_seed[s][tau]
                for s in range(4)
                if by_seed.get(s, {}).get(tau) is not None
            ]
            status = (
                "complete"
                if len(vals) == 4
                else ("partial" if vals else "missing")
            )
            row: dict[str, object] = {
                "method": f"I-MART-{k}" if k > 1 else "TD3+BC",
                "K": k,
                "T": f"{tau:g}",
                "seeds": "0,1,2,3",
                "available_runs": len(vals),
                "expected_runs": 4,
                "status": status,
                "mean_score": "",
                "std_seed": "",
                "scores": "",
            }
            if vals:
                row["mean_score"] = f"{sum(vals) / len(vals):.4g}"
                row["std_seed"] = f"{(st.stdev(vals) if len(vals) > 1 else 0.0):.4g}"
                row["scores"] = ",".join(f"{v:.4g}" for v in vals)
            sum_rows.append(row)

    with (OUT / "summary_by_T.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sum_rows)

    if QUEUE_LOG.is_file():
        shutil.copy2(QUEUE_LOG, OUT / "queue_amud_k1234_t_sweep.log")

    scored = sum(1 for r in long_rows if r["status"] == "available")
    readme = f"""# antmaze-umaze-diverse T-sweep (host choi)

- Algorithm: TD3+BC + MPI (Imp), K=1..4
- Env: `{ENV}` (`d4rl_score`, 100 eval episodes @ 1M)
- Tau grid: {", ".join(f"{t:g}" for t in TAU_GRID)}
- Auto-refreshed from local `results/antmaze_umaze_diverse_t_sweep` (eval/config only).
- Current scored cells: **{scored}**

## Files
- `K=*/Imp/seed{{0..3}}.csv`
- `scores_long.csv` / `summary_by_T.csv`
- `raw/` — per-run eval artifacts
- `queue_amud_k1234_t_sweep.log`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"[publish] scored={scored} out={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
