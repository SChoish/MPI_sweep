#!/usr/bin/env python3
"""Generate AAAI-26 manuscript figures from the compact bundled CSV/JSON data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8.2,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.8,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# Figure 1: architecture, deliberately rendered outside LaTeX to avoid label collisions.
fig, ax = plt.subplots(figsize=(3.35, 1.38))
ax.set_xlim(-0.15, 7.15)
ax.set_ylim(-1.65, 1.0)
ax.axis("off")
xs = [0.45, 2.05, 3.62, 5.82]
labels = [r"data $a$", r"$\mu_1$", r"$\mu_2$", r"$\mu_K$"]
for x, label in zip(xs, labels):
    ax.text(x, 0.28, label, ha="center", va="center", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="black", linewidth=0.8))
# chain arrows
segments = [(0.83, 1.67, r"$h_1$"), (2.43, 3.24, r"$h_2$"), (4.05, 5.45, r"$h_K$")]
for xa, xb, h in segments:
    ax.annotate("", xy=(xb, 0.28), xytext=(xa, 0.28), arrowprops=dict(arrowstyle="->", linewidth=0.9))
    ax.text((xa+xb)/2, 0.52, h, ha="center", va="bottom", fontsize=8)
ax.text(4.55, 0.28, r"$\cdots$", ha="center", va="center", fontsize=11)
# branch roles
ax.annotate("", xy=(2.05, -0.52), xytext=(2.05, 0.03), arrowprops=dict(arrowstyle="->", linewidth=0.95))
ax.text(2.05, -0.78, r"TD target via $\mu_1^{-}$", ha="center", va="center", fontsize=7.7,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="black", linewidth=0.75))
ax.annotate("", xy=(5.82, -0.52), xytext=(5.82, 0.03), arrowprops=dict(arrowstyle="->", linewidth=0.95))
ax.text(5.82, -0.78, r"deploy $\mu_K$", ha="center", va="center", fontsize=7.7,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="black", linewidth=0.75))
ax.annotate("", xy=(5.45, -1.35), xytext=(2.42, -1.35), arrowprops=dict(arrowstyle="<->", linewidth=0.75))
ax.text(3.94, -1.30, "separated target and deployment roles", ha="center", va="bottom", fontsize=7.4)
save(fig, "architecture")


# Figure 2: score curves, collapse rates, and task-cluster uncertainty.
budget_rows = rows(DATA / "budget_means.csv")
summary_rows = rows(DATA / "aggregate_summary.csv")
uncertainty_rows = rows(DATA / "task_cluster_uncertainty.csv")

budgets = np.asarray([float(k) for k in budget_rows[0] if k != "method"], dtype=float)
line_styles = ["-", "--", "-.", ":", (0, (4, 1, 1, 1)), (0, (1, 1))]
markers = ["o", "s", "^", "P", "D", "v"]
shades = [0.08, 0.25, 0.4, 0.55, 0.68, 0.8]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.24), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
ax = axes[0]
for row, ls, marker, shade in zip(budget_rows, line_styles, markers, shades):
    y = np.asarray([float(row[f"{b:g}"]) for b in budgets])
    label = row["method"].replace("BAR-", "").replace(" K=", "-")
    ax.plot(budgets, y, linestyle=ls, marker=marker, markersize=2.7,
            linewidth=1.1, color=str(shade), label=label)
ax.axvline(4, linewidth=0.65, linestyle=(0, (2, 2)), color="0.3")
ax.set_xscale("log")
ax.set_xlabel(r"nominal coefficient budget $T$")
ax.set_ylabel("mean normalized score")
ax.set_title("(a) Fixed-budget phase diagram")
ax.grid(True, linewidth=0.35, alpha=0.35)
ax.legend(frameon=False, ncol=2, loc="lower left", handlelength=2.4, columnspacing=0.8)

ax = axes[1]
methods = [r["method"].replace("BAR-", "").replace(" K=", "-") for r in summary_rows]
cell = np.asarray([int(r["collapsed_cells_lt20_of63"]) / 63 for r in summary_rows])
run = np.asarray([int(r["collapsed_runs_lt20_of126"]) / 126 for r in summary_rows])
x = np.arange(len(methods))
w = 0.36
ax.bar(x-w/2, cell, width=w, facecolor="white", edgecolor="black", linewidth=0.7, hatch="///", label="cell")
ax.bar(x+w/2, run, width=w, facecolor="0.65", edgecolor="black", linewidth=0.7, label="seeded run")
ax.set_xticks(x, methods, rotation=37, ha="right")
ax.set_ylim(0, 0.62)
ax.set_ylabel("score $<20$ rate")
ax.set_title(r"(b) Tested stability envelope ($T\geq4$)")
ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
ax.legend(frameon=False, loc="upper right")

ax = axes[2]
contrasts = [uncertainty_rows[0], uncertainty_rows[1]]
y = np.array([1, 0])
est = np.asarray([float(r["estimate"]) for r in contrasts])
lo = np.asarray([float(r["ci_low"]) for r in contrasts])
hi = np.asarray([float(r["ci_high"]) for r in contrasts])
ax.errorbar(est, y, xerr=np.vstack([est-lo, hi-est]), fmt="o", color="black",
            capsize=2.5, linewidth=1.0, markersize=4)
ax.axvline(0, color="0.45", linewidth=0.7, linestyle="--")
ax.set_yticks(y, [r"P4 $-$ TD3", r"P4 $-$ P3"])
ax.set_xlabel("high-budget score difference")
ax.set_title("(c) Task-cluster intervals")
ax.grid(True, axis="x", linewidth=0.35, alpha=0.35)
ax.text(0.03, -0.36, "descriptive 95% percentile intervals; 9 tasks",
        transform=ax.transAxes, fontsize=6.6, va="top")
fig.tight_layout(w_pad=1.0)
save(fig, "stability_summary")


# Figure 3: target-action displacement audit and failure signature.
claims = json.loads((DATA / "audit_claims.json").read_text(encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.28), gridspec_kw={"width_ratios": [1.0, 1.35]})
ax = axes[0]
labels = ["P3/TD3\ntarget", "P3/P2\ntarget", "P3/TD3\nfinal proxy"]
fractions = [88/90, 84/90, 36/90]
intervals = [
    claims["target_exposure"]["prox3_vs_td3"]["fraction_lower_cluster95"],
    claims["target_exposure"]["prox3_vs_prox2"]["fraction_lower_cluster95"],
    [0.25555555555555554, 0.5444444444444444],
]
xx = np.arange(3)
yerr = np.vstack([
    np.asarray(fractions)-np.asarray([v[0] for v in intervals]),
    np.asarray([v[1] for v in intervals])-np.asarray(fractions),
])
ax.errorbar(xx, fractions, yerr=yerr, fmt="o", color="black", capsize=3, linewidth=1.0)
ax.axhline(0.5, color="0.45", linestyle="--", linewidth=0.7)
ax.set_xticks(xx, labels)
ax.set_ylim(0.15, 1.04)
ax.set_ylabel("fraction with left quantity lower")
ax.set_title("(a) Sample-anchored movement audit")
ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
ax.text(0.02, 0.02, "18-cluster bootstrap; 90 matched cells",
        transform=ax.transAxes, fontsize=6.7)

ax = axes[1]
fields = ["TD error p99", r"$|Q|$", "critic loss", r"$D_{\rm critic}$", r"$D_{\rm final}$"]
# log10 collapsed/stable ratios from the reported medians.
lin_stable = np.array([270.3539518, 311.4305363, 33.9614086, .1118297, .1471883])
lin_col = np.array([9.5967646e23, 8.4764882e11, 1.2327059e23, .5384217, .6256434])
prox_stable = np.array([160.7724583, 346.6620483, 18.6603565, .0644033, .0943493])
prox_col = np.array([5.0304927e21, 6.9064901e10, 8.8210556e20, .4458609, .5780357])
lin_ratio = np.log10(lin_col / lin_stable)
prox_ratio = np.log10(prox_col / prox_stable)
x = np.arange(len(fields))
w = .36
ax.bar(x-w/2, lin_ratio, width=w, facecolor="white", edgecolor="black", hatch="///", linewidth=.7, label="Lin-2")
ax.bar(x+w/2, prox_ratio, width=w, facecolor="0.65", edgecolor="black", linewidth=.7, label="Prox-2")
ax.set_xticks(x, fields, rotation=25, ha="right")
ax.set_ylabel(r"$\log_{10}$ collapsed/stable median ratio")
ax.set_title("(b) Failure signature at the final checkpoint")
ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(w_pad=1.2)
save(fig, "diagnostics_summary")

# Single-column versions used by the AAAI main paper to avoid a float-only page.
fig, ax = plt.subplots(figsize=(3.35, 2.15))
labels = ["P3/TD3\ntarget", "P3/P2\ntarget", "P3/TD3\nfinal proxy"]
fractions = [88/90, 84/90, 36/90]
intervals = [
    claims["target_exposure"]["prox3_vs_td3"]["fraction_lower_cluster95"],
    claims["target_exposure"]["prox3_vs_prox2"]["fraction_lower_cluster95"],
    [0.25555555555555554, 0.5444444444444444],
]
xx = np.arange(3)
yerr = np.vstack([
    np.asarray(fractions)-np.asarray([v[0] for v in intervals]),
    np.asarray([v[1] for v in intervals])-np.asarray(fractions),
])
ax.errorbar(xx, fractions, yerr=yerr, fmt="o", color="black", capsize=3, linewidth=1.0)
ax.axhline(0.5, color="0.45", linestyle="--", linewidth=0.7)
ax.set_xticks(xx, labels)
ax.set_ylim(0.15, 1.04)
ax.set_ylabel("fraction with left quantity lower")
ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
ax.text(0.02, 0.02, "18-cluster bootstrap; 90 matched cells", transform=ax.transAxes, fontsize=6.7)
fig.tight_layout()
save(fig, "movement_audit")

fig, ax = plt.subplots(figsize=(3.35, 2.15))
fields = ["TD p99", r"$|Q|$", "loss", r"$D_c$", r"$D_f$"]
x = np.arange(len(fields)); w = .36
ax.bar(x-w/2, lin_ratio, width=w, facecolor="white", edgecolor="black", hatch="///", linewidth=.7, label="Lin-2")
ax.bar(x+w/2, prox_ratio, width=w, facecolor="0.65", edgecolor="black", linewidth=.7, label="Prox-2")
ax.set_xticks(x, fields, rotation=22, ha="right")
ax.set_ylabel(r"$\log_{10}$ collapsed/stable median ratio")
ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
save(fig, "failure_signature")
