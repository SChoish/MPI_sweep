#!/usr/bin/env python3
"""Verify the AAAI-26 GPT manuscript bundle and its released-result claims."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED_STYLE_GIT_SHA1 = "1c587a54d5613355974d8ac25ebb7d5d741c84e0"
GRID = (0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4.0, 7.0, 10.0, 12.0, 14.0, 17.0, 20.0)
HIGH = tuple(t for t in GRID if t >= 4.0)
METHODS = {
    "TD3+BC": ("K=1", "Imp"),
    "BAR-Prox K=2": ("K=2", "Imp"),
    "BAR-Prox K=3": ("K=3", "Imp"),
    "BAR-Prox K=4": ("K=4", "Imp"),
    "BAR-Lin K=2": ("K=2", "Exp"),
    "BAR-Lin K=3": ("K=3", "Exp"),
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def close(actual: float, expected: float, tol: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise VerificationError(f"{message}: observed={actual}, expected={expected}, tol={tol}")


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"empty CSV: {path}")
    return rows


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_style_and_source() -> None:
    style = ROOT / "aaai2026.sty"
    require(style.is_file(), "aaai2026.sty is missing")
    require(git_blob_sha1(style) == EXPECTED_STYLE_GIT_SHA1,
            "aaai2026.sty differs from the pinned AAAI-26 author-kit mirror")
    style_text = style.read_text(encoding="utf-8")
    require("\\ProvidesPackage{aaai2026}[2026/04/29 AAAI 2026 Submission format]" in style_text,
            "unexpected AAAI style package declaration")

    forbidden_packages = {
        "hyperref", "geometry", "balance", "flushend", "fontenc", "fullpage",
        "setspace", "stfloats", "titlesec", "ulem", "wrapfig", "authblk",
    }
    usepackage = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}")
    for name in ("paper.tex", "supplement.tex"):
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        require("\\usepackage[submission]{aaai2026}" in text,
                f"{name} does not use anonymous AAAI-26 submission style")
        loaded: set[str] = set()
        for match in usepackage.finditer(text):
            loaded.update(part.strip() for part in match.group(1).split(","))
        bad = sorted(loaded & forbidden_packages)
        require(not bad, f"{name} loads forbidden AAAI packages: {bad}")
        require("medium-expert" not in text, f"{name} contains invalid medium-expert dataset name")
        require("-ME" not in text and "\\texttt{ME}" not in text,
                f"{name} contains stale ME abbreviation")
        require("directly measured target-policy displacement" not in text,
                f"{name} contains an overstated target-displacement phrase")
        require("required controlled rerun" not in text,
                f"{name} retains the withdrawn common-initialization requirement")

    paper = (ROOT / "paper.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "supplement.tex").read_text(encoding="utf-8")
    joined = paper + "\n" + supplement
    required = (
        "sample-anchored next-state target-action displacement",
        "every $50{,}000$ critic updates",
        "locally Lipschitz",
        "not certified",
        "action-metric-matched projected linearization",
        "only in five of eight final-checkpoint comparisons",
        "Identical critic initializations across $K$ are not required",
        "expert-v2",
        "$35/63$ to $9/63$",
        "$[20.51,54.35]$",
        "$[2.23,21.97]$",
    )
    missing = [token for token in required if token not in joined]
    require(not missing, f"required revised claims/qualifications missing: {missing}")
    require("every $5{,}000$" not in joined and "every 5,000" not in joined,
            "stale 5,000-update historical evaluation claim remains")
    require("where $\n" not in paper and "reference $\n" not in paper,
            "paper contains an accidental newline inside a LaTeX command")
    require("C_{1,t}^{\\mathrm P}" in joined and "C_{1,t}^{\\mathrm L}" in joined,
            "implementation-specific first-hop normalization equations are missing")
    require("C_{k,t}^{\\mathrm P}" in joined and "C_{k,t}^{\\mathrm L}" in joined,
            "implementation-specific later-hop normalization equations are missing")
    print("PASS style/source: AAAI format, P0 corrections, and claim wording")


def load_sweep(results_dir: Path, hops: str, realization: str) -> tuple[list[str], dict[tuple[float, int, str], float]]:
    values: dict[tuple[float, int, str], float] = {}
    envs: list[str] | None = None
    for seed in (0, 1):
        path = results_dir / hops / realization / f"seed{seed}.csv"
        rows = read_csv(path)
        current = [key for key in rows[0].keys() if key != "tau"]
        if envs is None:
            envs = current
        require(current == envs, f"environment columns differ across seeds: {hops}/{realization}")
        for row in rows:
            tau = float(row["tau"])
            if tau not in GRID:
                continue
            for env in envs:
                raw = row[env].strip()
                require(raw not in ("", "—"), f"missing complete-grid cell: {hops}/{realization}/{tau}/{seed}/{env}")
                key = (tau, seed, env)
                require(key not in values, f"duplicate sweep cell: {hops}/{realization}/{key}")
                values[key] = float(raw)
    assert envs is not None
    require(len(envs) == 9, f"expected 9 environments, found {len(envs)}")
    require(len(values) == len(GRID) * 2 * len(envs),
            f"incomplete grid: {hops}/{realization} has {len(values)} values")
    return envs, values


def bundled_summary() -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in read_csv(ROOT / "data" / "aggregate_summary.csv")}


def bundled_budget_means() -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in read_csv(ROOT / "data" / "budget_means.csv")}


def bundled_environment_means() -> dict[str, dict[str, str]]:
    return {row["environment"]: row for row in read_csv(ROOT / "data" / "environment_high_means.csv")}


def verify_complete_sweeps(results_dir: Path) -> None:
    expected_summary = bundled_summary()
    expected_budget = bundled_budget_means()
    expected_env = bundled_environment_means()
    source: dict[str, tuple[list[str], dict[tuple[float, int, str], float]]] = {}

    reference_envs: list[str] | None = None
    for method, (hops, realization) in METHODS.items():
        envs, values = load_sweep(results_dir, hops, realization)
        if reference_envs is None:
            reference_envs = envs
        require(envs == reference_envs, f"environment order differs for {method}")
        source[method] = (envs, values)

        per_budget = {
            tau: statistics.mean(values[(tau, seed, env)] for seed in (0, 1) for env in envs)
            for tau in GRID
        }
        low = statistics.mean(v for tau, v in per_budget.items() if tau <= 2.5)
        high = statistics.mean(v for tau, v in per_budget.items() if tau >= 4.0)
        tail = statistics.mean(v for tau, v in per_budget.items() if tau >= 10.0)
        cell_scores = [statistics.mean(values[(tau, seed, env)] for seed in (0, 1)) for tau in HIGH for env in envs]
        raw_scores = [values[(tau, seed, env)] for tau in HIGH for seed in (0, 1) for env in envs]
        exp = expected_summary[method]
        close(low, float(exp["mean_T_le_2.5"]), 0.015, f"{method} low mean")
        close(high, float(exp["mean_T_ge_4"]), 0.015, f"{method} high mean")
        close(tail, float(exp["mean_T_ge_10"]), 0.015, f"{method} tail mean")
        require(sum(v < 20 for v in cell_scores) == int(exp["collapsed_cells_lt20_of63"]),
                f"{method} cell collapse count mismatch")
        require(sum(v < 20 for v in raw_scores) == int(exp["collapsed_runs_lt20_of126"]),
                f"{method} raw collapse count mismatch")
        for tau, value in per_budget.items():
            close(value, float(expected_budget[method][f"{tau:g}"]), 0.015,
                  f"{method} budget mean at T={tau:g}")

    assert reference_envs is not None
    label_map = {
        "hopper-medium-v2": "Hopper-M",
        "hopper-medium-replay-v2": "Hopper-MR",
        "hopper-expert-v2": "Hopper-E",
        "halfcheetah-medium-v2": "HalfCheetah-M",
        "halfcheetah-medium-replay-v2": "HalfCheetah-MR",
        "halfcheetah-expert-v2": "HalfCheetah-E",
        "walker2d-medium-v2": "Walker2d-M",
        "walker2d-medium-replay-v2": "Walker2d-MR",
        "walker2d-expert-v2": "Walker2d-E",
    }
    for env in reference_envs:
        label = label_map[env]
        for method, (_, values) in source.items():
            observed = statistics.mean(values[(tau, seed, env)] for tau in HIGH for seed in (0, 1))
            close(observed, float(expected_env[label][method]), 0.055,
                  f"{method} high mean for {label}")

    # Recompute the two score contrasts and an independent task bootstrap.
    task_diffs: dict[str, list[float]] = {"P4-TD3": [], "P4-P3": []}
    for env in reference_envs:
        means = {}
        for method, (_, values) in source.items():
            means[method] = statistics.mean(values[(tau, seed, env)] for tau in HIGH for seed in (0, 1))
        task_diffs["P4-TD3"].append(means["BAR-Prox K=4"] - means["TD3+BC"])
        task_diffs["P4-P3"].append(means["BAR-Prox K=4"] - means["BAR-Prox K=3"])
    expected_uncertainty = read_csv(ROOT / "data" / "task_cluster_uncertainty.csv")
    score_rows = expected_uncertainty[:2]
    rng = np.random.default_rng(20260830)
    for key, row in zip(("P4-TD3", "P4-P3"), score_rows):
        data = np.asarray(task_diffs[key], dtype=float)
        close(float(data.mean()), float(row["estimate"]), 0.015, f"{key} task mean")
        picks = rng.integers(0, len(data), size=(100_000, len(data)))
        interval = np.percentile(data[picks].mean(axis=1), [2.5, 97.5])
        # Percentile endpoints move slightly with bootstrap RNG/implementation.
        close(float(interval[0]), float(row["ci_low"]), 0.12, f"{key} bootstrap lower")
        close(float(interval[1]), float(row["ci_high"]), 0.12, f"{key} bootstrap upper")
    print("PASS complete sweeps: score matrices, collapse counts, task means, and intervals")


def indexed(rows: Iterable[dict[str, str]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    out: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        require(key not in out, f"duplicate key {key}")
        out[key] = row
    return out


def verify_audit_pack(results_dir: Path) -> None:
    audit = results_dir / "diagnostics" / "audit_pack"
    expected = json.loads((ROOT / "data" / "audit_claims.json").read_text(encoding="utf-8"))

    target_rows = read_csv(audit / "target_exposure_run.csv")
    require(len(target_rows) == 270, f"target exposure rows={len(target_rows)}, expected 270")
    target_idx = indexed(target_rows, ("method", "env", "T", "seed"))
    def target_pair(left: str, right: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        lvals, rvals = [], []
        for (method, env, tau, seed), row in target_idx.items():
            if method != left:
                continue
            rkey = (right, env, tau, seed)
            require(rkey in target_idx, f"missing target pair {rkey}")
            lvals.append(float(row[metric])); rvals.append(float(target_idx[rkey][metric]))
        require(len(lvals) == 90, f"{left}/{right}/{metric}: expected 90 pairs")
        return np.asarray(lvals), np.asarray(rvals)
    for label, left, right in (("prox3_vs_td3", "mpi3", "td3"), ("prox3_vs_prox2", "mpi3", "mpi2")):
        lval, rval = target_pair(left, right, "d_target_deterministic")
        claim = expected["target_exposure"][label]
        require(int(np.sum(lval < rval)) == claim["lower"][0], f"{label} lower count mismatch")
        close(float(np.median(lval / rval)), float(claim["median_ratio"]), 1e-9, f"{label} median ratio")
    lval, rval = target_pair("mpi3", "td3", "d_final_current")
    claim = expected["target_exposure"]["final_proxy_prox3_vs_td3"]
    require(int(np.sum(lval < rval)) == claim["lower"][0], "final proxy lower count mismatch")
    close(float(np.median(lval / rval)), float(claim["median_ratio"]), 1e-9, "final proxy median ratio")

    proxy = np.asarray([float(row["current_state_proxy"]) for row in target_rows])
    det = np.asarray([float(row["d_target_deterministic"]) for row in target_rows])
    close(float(np.corrcoef(proxy, det)[0, 1]),
          float(expected["target_exposure"]["proxy_alignment"]["pearson_deterministic"]),
          1e-12, "proxy/target correlation")

    route_rows = read_csv(audit / "route_intervention_run.csv")
    require(len(route_rows) == 8, "route audit must contain 8 runs")
    require(sum(float(row["delta"]) > 0 for row in route_rows) == 8, "route primary direction is not 8/8")
    require(sum(float(row["delta_final_checkpoint_only"]) > 0 for row in route_rows) == 5,
            "route final-checkpoint sensitivity is not 5/8")
    close(statistics.mean(float(row["delta"]) for row in route_rows),
          float(expected["route"]["mean_delta"]), 1e-12, "route mean delta")

    simulator_rows = read_csv(audit / "simulator_checkpoint.csv")
    require(len(simulator_rows) == 60, "simulator audit must contain 60 checkpoints")
    stable = [float(r["median_signed_error"]) for r in simulator_rows if int(r["collapsed"]) == 0]
    collapsed = [float(r["median_signed_error"]) for r in simulator_rows if int(r["collapsed"]) == 1]
    require((len(stable), len(collapsed)) == (55, 5), "simulator stable/collapsed count mismatch")
    close(float(np.median(stable)), float(expected["simulator"]["checkpoint_median_of_medians_stable"]),
          1e-9, "simulator stable median of checkpoint medians")
    close(float(np.median(collapsed)), float(expected["simulator"]["checkpoint_median_of_medians_collapsed"]),
          1e-3, "simulator collapsed median of checkpoint medians")

    failure_rows = read_csv(audit / "failure_diagnostics_run.csv")
    require(len(failure_rows) == 180, "failure audit must contain 180 rows")
    for method in ("lin2", "prox2"):
        for group, pred in (("stable", lambda x: x >= 20), ("collapsed", lambda x: x < 20)):
            selected = [r for r in failure_rows if r["method"] == method and pred(float(r["score"]))]
            claim = expected["failure_diagnostics"][f"{method}_{group}"]
            require(len(selected) == int(claim["runs"]), f"{method}/{group} run count mismatch")
            mapping = {
                "td_p99": "td_error_p99", "q_abs": "q_abs_mean", "critic_loss": "critic_loss",
                "d_critic": "d_critic", "d_final": "d_final", "common_gain": "common_critic_gain",
            }
            for claim_field, csv_field in mapping.items():
                observed = statistics.median(float(r[csv_field]) for r in selected)
                exp = float(claim[claim_field])
                tol = max(1e-9, abs(exp) * 1e-12)
                close(observed, exp, tol, f"{method}/{group}/{claim_field}")
    print("PASS compact audit: target, route, simulator, and failure claims")


def verify_bundled_data_only() -> None:
    summary = bundled_summary()
    require(set(summary) == set(METHODS), "aggregate summary method set mismatch")
    require(summary["TD3+BC"]["collapsed_cells_lt20_of63"] == "35", "TD3 cell count mismatch")
    require(summary["BAR-Prox K=4"]["collapsed_cells_lt20_of63"] == "9", "P4 cell count mismatch")
    claims = json.loads((ROOT / "data" / "audit_claims.json").read_text(encoding="utf-8"))
    require(claims["target_exposure"]["prox3_vs_td3"]["lower"] == [88, 90], "bundled exposure claim mismatch")
    require(claims["route"]["primary_positive"] == [8, 8], "bundled route primary mismatch")
    require(claims["route"]["final_checkpoint_positive"] == [5, 8], "bundled route sensitivity mismatch")
    print("PASS bundled compact data: schemas and headline values")


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"command failed ({' '.join(command)}): {result.stderr.strip()}")
    return result.stdout


def verify_built_pdfs() -> None:
    main_pdf = ROOT / "gpt_aaai26_manuscript.pdf"
    supp_pdf = ROOT / "gpt_aaai26_supplement.pdf"
    for pdf in (main_pdf, supp_pdf):
        require(pdf.is_file() and pdf.stat().st_size > 10_000, f"missing or implausibly small PDF: {pdf.name}")
        if shutil.which("pdffonts"):
            fonts = command_output(["pdffonts", str(pdf)])
            for line in fonts.splitlines()[2:]:
                columns = line.split()
                if len(columns) >= 4:
                    require(columns[3].lower() != "yes" or "type 3" not in line.lower(),
                            f"Type 3 font detected in {pdf.name}: {line}")
            require("Type 3" not in fonts, f"Type 3 font detected in {pdf.name}")
    aux = (ROOT / "paper.aux").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"newlabel\{sec:technical_end\}\{\{[^}]*\}\{([0-9]+)\}", aux)
    require(match is not None, "technical-content boundary label missing from paper.aux")
    require(int(match.group(1)) <= 7, f"technical content exceeds AAAI page 7: boundary page {match.group(1)}")
    if shutil.which("pdfinfo"):
        main_info = command_output(["pdfinfo", str(main_pdf)])
        supp_info = command_output(["pdfinfo", str(supp_pdf)])
        main_pages = int(re.search(r"^Pages:\s+(\d+)", main_info, re.M).group(1))
        supp_pages = int(re.search(r"^Pages:\s+(\d+)", supp_info, re.M).group(1))
        require(main_pages <= 9, f"unexpectedly long main PDF: {main_pages} pages")
        require(supp_pages >= 1, "supplement PDF has no pages")
    for stem in ("paper", "supplement"):
        log = (ROOT / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
        blocking = (
            "Undefined control sequence", "Emergency stop", "Fatal error",
            "LaTeX Error", "Overfull \\hbox", "Overfull \\vbox",
            "There were undefined references",
        )
        found = [token for token in blocking if token in log]
        require(not found, f"{stem}.log contains blocking warnings/errors: {found}")
    print("PASS built PDFs: page boundary, fonts, and LaTeX logs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT.parent / "sweep_results")
    parser.add_argument("--prebuild", action="store_true")
    parser.add_argument("--postbuild", action="store_true")
    args = parser.parse_args()
    if not args.prebuild and not args.postbuild:
        args.prebuild = args.postbuild = True

    try:
        if args.prebuild:
            verify_style_and_source()
            verify_bundled_data_only()
            if args.results_dir.is_dir():
                verify_complete_sweeps(args.results_dir)
                verify_audit_pack(args.results_dir)
            else:
                print(f"NOTE source-level result recheck skipped: {args.results_dir} not present")
        if args.postbuild:
            verify_built_pdfs()
    except VerificationError as error:
        print(f"VERIFY FAILED: {error}", file=sys.stderr)
        return 1
    print("ALL REQUESTED CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
