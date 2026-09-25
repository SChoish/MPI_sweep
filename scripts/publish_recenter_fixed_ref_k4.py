#!/usr/bin/env python3
"""Publish choi/svcho recenter vs fixed_ref K=4 pair summaries into MPI_sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
BRANCHES = ("recenter", "fixed_ref")
RUN_RE = re.compile(
    r"^(?P<env>.+)_tau(?P<tau>[0-9.]+)_shared_(?P<branch>recenter|fixed_ref)4_seed(?P<seed>\d+)$"
)


def run_dir(save_dir: Path, env: str, tau: float, seed: int, branch: str) -> Path:
    return save_dir / f"{env}_tau{tau:g}_shared_{branch}4_seed{seed}"


def read_final(eval_path: Path, step: int = 1_000_000) -> tuple[float, float] | None:
    if not eval_path.is_file():
        return None
    with eval_path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["step"]) == step]
    if len(rows) != 1:
        return None
    return float(rows[0]["d4rl_score"]), float(rows[0]["d4rl_pi4"])


def discover_pairs(full_dir: Path) -> list[tuple[str, float, int]]:
    found: set[tuple[str, float, int]] = set()
    if not full_dir.is_dir():
        return []
    for path in full_dir.iterdir():
        if not path.is_dir():
            continue
        match = RUN_RE.match(path.name)
        if not match:
            continue
        found.add((match["env"], float(match["tau"]), int(match["seed"])))
    return sorted(found, key=lambda item: (item[0], item[1], item[2]))


def publish(src: Path, dst: Path, host: str, log_path: Path | None) -> dict:
    full = src / "full"
    dst.mkdir(parents=True, exist_ok=True)
    rows = [
        "host,env,T,K,seed,first_score,recenter_pi4,fixed_ref_pi4,paired_delta,status"
    ]
    done = 0
    partial = 0
    for env, tau, seed in discover_pairs(full):
        scores = {}
        for branch in BRANCHES:
            directory = run_dir(full, env, tau, seed, branch)
            final = directory / "params_1000000.pkl"
            eval_scores = read_final(directory / "eval.csv")
            if not final.is_file() or eval_scores is None:
                scores[branch] = None
                continue
            scores[branch] = eval_scores
        if all(scores[branch] is not None for branch in BRANCHES):
            first_a, recenter = scores["recenter"]
            first_b, fixed = scores["fixed_ref"]
            # Prefer recenter first-actor score; they should match when verified.
            first = first_a
            status = "complete"
            if abs(first_a - first_b) > 1e-9:
                status = "complete_first_mismatch"
            done += 1
            rows.append(
                ",".join(
                    [
                        host,
                        env,
                        f"{tau:g}",
                        "4",
                        str(seed),
                        str(first),
                        str(recenter),
                        str(fixed),
                        str(recenter - fixed),
                        status,
                    ]
                )
            )
        else:
            partial += 1
            rows.append(
                ",".join(
                    [
                        host,
                        env,
                        f"{tau:g}",
                        "4",
                        str(seed),
                        "",
                        "",
                        "",
                        "",
                        "incomplete",
                    ]
                )
            )
    if log_path and log_path.is_file():
        (dst / "queue.log").write_text(log_path.read_text(errors="replace")[-400_000:])
    status = {
        "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
        "experiment": "td3bc_recenter_vs_fixed_ref_k4_high_T",
        "host": host,
        "K": 4,
        "branches": list(BRANCHES),
        "source": str(full),
        "pairs_complete": done,
        "pairs_incomplete": partial,
        "pairs_seen": done + partial,
        "note": (
            "choi host share of matched recenter/fixed_ref pairs. "
            "Checkpoints stay local under source; only scores/STATUS are published."
        ),
    }
    (dst / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    (dst / "scores.csv").write_text("\n".join(rows) + "\n")
    print(f"published host={host} complete={done} incomplete={partial}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/choi/MPI_store/recenter_fixed_ref_k4"),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/home/choi/MPI_sweep/sweep_results/recenter_fixed_ref_k4_choi"),
    )
    parser.add_argument("--host", default="choi")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/home/choi/MPI_store/recenter_fixed_ref_k4/primary.log"),
    )
    args = parser.parse_args()
    publish(args.src, args.dst, args.host, args.log if args.log.is_file() else None)


if __name__ == "__main__":
    main()
