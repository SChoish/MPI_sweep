from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "legend.fontsize": 6.8,
    }
)


def read_budget_means() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with (DATA / "budget_means.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    tau = np.asarray([float(x) for x in rows[0][1:]], dtype=float)
    values = {row[0]: np.asarray([float(x) for x in row[1:]], dtype=float) for row in rows[1:]}
    return tau, values


def read_summary() -> dict[str, dict[str, float]]:
    with (DATA / "aggregate_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return {
            row["method"]: {
                k: float(v) if "." in v else int(v)
                for k, v in row.items()
                if k != "method"
            }
            for row in rows
        }


def read_matrix(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    envs = rows[0][1:]
    tau = np.asarray([float(row[0]) for row in rows[1:]], dtype=float)
    matrix = np.asarray([[float(x) for x in row[1:]] for row in rows[1:]], dtype=float)
    return tau, envs, matrix


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def stability_figure() -> None:
    tau, curves = read_budget_means()
    summary = read_summary()
    order = [
        "TD3+BC",
        "MPI-Prox K=2",
        "MPI-Prox K=3",
        "MPI-Prox K=4",
        "MPI-Lin K=2",
        "MPI-Lin K=3",
    ]
    labels = {
        "TD3+BC": "TD3+BC ($K=1$)",
        "MPI-Prox K=2": "MPI-Prox ($K=2$)",
        "MPI-Prox K=3": "MPI-Prox ($K=3$)",
        "MPI-Prox K=4": "MPI-Prox ($K=4$)",
        "MPI-Lin K=2": "MPI-Lin ($K=2$)",
        "MPI-Lin K=3": "MPI-Lin ($K=3$)",
    }
    markers = ["o", "s", "^", "P", "D", "v"]

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.5), constrained_layout=True)
    ax = axes[0]
    for method, marker in zip(order, markers, strict=True):
        ax.plot(
            tau,
            curves[method],
            marker=marker,
            markersize=3.1,
            linewidth=1.15,
            label=labels[method],
        )
    ax.set_xscale("log")
    ax.set_xlabel("total nominal actor budget $T$")
    ax.set_ylabel("mean D4RL score")
    ax.set_title("(a) Fixed-budget stability envelope")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, ncol=2, loc="best")

    x = np.arange(len(order))
    cell = np.asarray([summary[m]["collapsed_cells_lt20_of63"] for m in order], dtype=float)
    raw = np.asarray([summary[m]["collapsed_runs_lt20_of126"] for m in order], dtype=float)
    width = 0.36
    axes[1].bar(x - width / 2, cell / 63.0, width, label="environment-budget cells")
    axes[1].bar(x + width / 2, raw / 126.0, width, label="seeded runs")
    axes[1].set_xticks(x, ["TD3", "P2", "P3", "P4", "L2", "L3"])
    axes[1].set_ylim(0, 0.62)
    axes[1].set_ylabel("collapse rate (score $<20$)")
    axes[1].set_title("(b) High-budget failures ($T\\geq4$)")
    axes[1].grid(True, axis="y", alpha=0.25, linewidth=0.5)
    axes[1].legend(frameon=False, loc="upper right")
    for i, (c, r) in enumerate(zip(cell.astype(int), raw.astype(int), strict=True)):
        axes[1].text(i - width / 2, c / 63.0 + 0.010, f"{c}/63", ha="center", va="bottom", rotation=90, fontsize=5.7)
        axes[1].text(i + width / 2, r / 126.0 + 0.010, f"{r}/126", ha="center", va="bottom", rotation=90, fontsize=5.7)

    save(fig, "stability_envelope")


def k4_figure() -> None:
    summary = read_summary()
    t3, envs3, s30 = read_matrix(RAW / "prox3_seed0.csv")
    t3b, envs3b, s31 = read_matrix(RAW / "prox3_seed1.csv")
    t4, envs4, s40 = read_matrix(RAW / "prox4_seed0.csv")
    t4b, envs4b, s41 = read_matrix(RAW / "prox4_seed1.csv")
    if not (np.array_equal(t3, t3b) and np.array_equal(t3, t4) and np.array_equal(t3, t4b)):
        raise ValueError("K=3 and K=4 tau grids differ")
    if not (envs3 == envs3b == envs4 == envs4b):
        raise ValueError("K=3 and K=4 environment orders differ")

    m3 = (s30 + s31) / 2.0
    m4 = (s40 + s41) / 2.0
    high = t3 >= 4
    x3 = m3[high].reshape(-1)
    x4 = m4[high].reshape(-1)
    rescued = int(np.sum((x3 < 20) & (x4 >= 20)))
    harmed = int(np.sum((x3 >= 20) & (x4 < 20)))
    both_failed = int(np.sum((x3 < 20) & (x4 < 20)))
    both_stable = int(np.sum((x3 >= 20) & (x4 >= 20)))
    diff = x4 - x3

    ks = np.asarray([1, 2, 3, 4])
    prox_methods = ["TD3+BC", "MPI-Prox K=2", "MPI-Prox K=3", "MPI-Prox K=4"]
    high_mean = np.asarray([summary[m]["mean_T_ge_4"] for m in prox_methods], dtype=float)
    collapse = np.asarray([summary[m]["collapsed_cells_lt20_of63"] for m in prox_methods], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), constrained_layout=True)
    left = axes[0]
    left.plot(ks, high_mean, marker="o", linewidth=1.3, label="high-budget mean")
    left.set_xlabel("number of proximal actors $K$")
    left.set_xticks(ks)
    left.set_ylabel("mean score for $T\\geq4$")
    left.set_title("(a) Descriptive scaling with $K$")
    left.grid(True, alpha=0.25, linewidth=0.5)
    right_axis = left.twinx()
    right_axis.plot(ks, collapse / 63.0, marker="s", linestyle="--", linewidth=1.2, label="collapse rate")
    right_axis.set_ylabel("cell collapse rate")
    left.text(3.08, high_mean[2] + 1.2, "high-budget mean", fontsize=6.8)
    right_axis.text(3.02, collapse[2] / 63.0 - 0.035, "collapse rate", fontsize=6.8)

    ax = axes[1]
    ax.scatter(x3, x4, s=18, alpha=0.78)
    lim_lo = min(-5.0, float(np.min([x3.min(), x4.min()])))
    lim_hi = max(118.0, float(np.max([x3.max(), x4.max()])))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--", linewidth=0.8)
    ax.axvline(20, linestyle=":", linewidth=0.8)
    ax.axhline(20, linestyle=":", linewidth=0.8)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("MPI-Prox $K=3$ cell mean")
    ax.set_ylabel("MPI-Prox $K=4$ cell mean")
    ax.set_title("(b) High-budget cells: $K=3$ vs. $K=4$")
    ax.grid(True, alpha=0.2, linewidth=0.45)
    ax.text(
        0.03,
        0.97,
        f"rescued: {rescued}    harmed: {harmed}\n"
        f"both stable: {both_stable}    both failed: {both_failed}\n"
        f"mean $\\Delta$: {diff.mean():.2f}; median $\\Delta$: {np.median(diff):.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=6.6,
    )
    save(fig, "k4_analysis")

    derived = {
        "k4_minus_k3_high_budget_mean_difference": float(diff.mean()),
        "k4_minus_k3_high_budget_median_difference": float(np.median(diff)),
        "k4_wins_by_more_than_one": int(np.sum(diff > 1)),
        "within_one_point": int(np.sum(np.abs(diff) <= 1)),
        "k3_wins_by_more_than_one": int(np.sum(diff < -1)),
        "k3_collapsed_k4_stable": rescued,
        "k3_stable_k4_collapsed": harmed,
        "both_collapsed": both_failed,
        "both_stable": both_stable,
    }
    (DATA / "derived_metrics.json").write_text(json.dumps(derived, indent=2), encoding="utf-8")


def replication_figure() -> None:
    tau, curves = read_budget_means()
    t2, envs2, s2 = read_matrix(RAW / "k1_rep_seed2.csv")
    t3, envs3, s3 = read_matrix(RAW / "k1_rep_seed3.csv")
    if not (np.array_equal(tau, t2) and np.array_equal(tau, t3) and envs2 == envs3):
        raise ValueError("replication grid differs from the core grid")
    new_mean = ((s2 + s3) / 2.0).mean(axis=1)

    fig, ax = plt.subplots(figsize=(3.6, 2.35), constrained_layout=True)
    ax.plot(tau, curves["TD3+BC"], marker="o", linewidth=1.25, label="original seeds 0-1")
    ax.plot(tau, new_mean, marker="s", linewidth=1.25, label="replication seeds 2-3")
    ax.set_xscale("log")
    ax.set_xlabel("total nominal actor budget $T$")
    ax.set_ylabel("mean D4RL score")
    ax.set_title("One-hop phase diagram replication")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    save(fig, "k1_replication")


if __name__ == "__main__":
    stability_figure()
    k4_figure()
    replication_figure()
    print(f"Wrote figures to {OUT}")
