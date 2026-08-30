#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.15,
})


def budget_means():
    with (DATA / "budget_means.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    tau = np.array([float(x) for x in rows[0][1:]])
    vals = {r[0]: np.array([float(x) for x in r[1:]]) for r in rows[1:]}
    return tau, vals


def aggregate_summary():
    with (DATA / "aggregate_summary.csv").open(newline="", encoding="utf-8") as f:
        return {r["method"]: r for r in csv.DictReader(f)}


def task_uncertainty():
    with (DATA / "task_cluster_uncertainty.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def stability():
    tau, curves = budget_means()
    summary = aggregate_summary()
    uncertainty = {r["comparison"]: r for r in task_uncertainty()}
    order = [
        "TD3+BC", "MPI-Prox K=2", "MPI-Prox K=3",
        "MPI-Prox K=4", "MPI-Lin K=2", "MPI-Lin K=3",
    ]
    labels = ["TD3+BC", "BAR-P2", "BAR-P3", "BAR-P4", "BAR-L2", "BAR-L3"]
    markers = ["o", "s", "^", "P", "D", "v"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]
    greys = ["0.05", "0.20", "0.35", "0.50", "0.65", "0.80"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.38, 1.02, 1.02]})
    ax = axes[0]
    for method, label, marker, ls, grey in zip(order, labels, markers, linestyles, greys, strict=True):
        ax.plot(tau, curves[method], marker=marker, markersize=2.8,
                linestyle=ls, color=grey, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("nominal coefficient budget $T$")
    ax.set_ylabel("mean D4RL score")
    ax.set_title("(a) Fixed-budget phase diagram")
    ax.grid(True, alpha=.22, linewidth=.45)
    ax.legend(frameon=False, ncol=2, loc="best", columnspacing=.7, handlelength=2.3)

    cell = np.array([int(summary[m]["collapsed_cells_lt20_of63"]) for m in order])
    raw = np.array([int(summary[m]["collapsed_runs_lt20_of126"]) for m in order])
    x = np.arange(len(order)); w=.36
    axes[1].bar(x-w/2, cell/63, w, color="0.30", edgecolor="black", linewidth=.3,
                label="two-seed cells")
    axes[1].bar(x+w/2, raw/126, w, color="0.75", edgecolor="black", linewidth=.3,
                label="seeded runs")
    axes[1].set_xticks(x, ["TD3","P2","P3","P4","L2","L3"], rotation=20)
    axes[1].set_ylim(0,.62)
    axes[1].set_ylabel("score $<20$ rate")
    axes[1].set_title("(b) High-budget failures")
    axes[1].grid(True, axis="y", alpha=.22, linewidth=.45)
    axes[1].legend(frameon=False, loc="upper right")
    for i,c in enumerate(cell):
        axes[1].text(i-w/2, c/63+.014, str(c), ha="center", va="bottom", fontsize=6)

    comps = ["K4-K1", "K4-K3"]
    ypos = np.array([1.0, 0.0])
    score = [float(uncertainty[c]["mean_score_difference"]) for c in comps]
    slo = [float(uncertainty[c]["mean_score_ci_low"]) for c in comps]
    shi = [float(uncertainty[c]["mean_score_ci_high"]) for c in comps]
    axes[2].errorbar(score, ypos+.13, xerr=[np.array(score)-slo, np.array(shi)-score],
                     fmt="o", color="0.10", capsize=2.5, label="score difference")
    risk = [float(uncertainty[c]["raw_collapse_risk_difference"]) * 100 for c in comps]
    rlo = [float(uncertainty[c]["raw_collapse_risk_ci_low"]) * 100 for c in comps]
    rhi = [float(uncertainty[c]["raw_collapse_risk_ci_high"]) * 100 for c in comps]
    # Separate lower x axis by plotting risk as percentage on a twinned top axis.
    ax2 = axes[2].twiny()
    ax2.errorbar(risk, ypos-.13, xerr=[np.array(risk)-rlo, np.array(rhi)-risk],
                 fmt="s", color="0.55", ecolor="0.55", capsize=2.5,
                 label="collapse-risk difference")
    axes[2].axvline(0, color="0.35", linewidth=.7, linestyle=":")
    ax2.axvline(0, color="0.75", linewidth=.7, linestyle=":")
    axes[2].set_yticks(ypos, ["P4-TD3", "P4-P3"])
    axes[2].set_xlabel("mean score difference")
    ax2.set_xlabel("raw collapse-risk diff. (pp)")
    axes[2].set_title("(c) Dataset-cluster intervals")
    axes[2].grid(True, axis="x", alpha=.18, linewidth=.4)
    save(fig, "stability_envelope")


def supplemental():
    tau, curves = budget_means()
    summary = aggregate_summary()
    # K-depth summary.
    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    ks=np.array([1,2,3,4])
    methods=["TD3+BC","MPI-Prox K=2","MPI-Prox K=3","MPI-Prox K=4"]
    high=np.array([float(summary[m]["mean_T_ge_4"]) for m in methods])
    col=np.array([int(summary[m]["collapsed_cells_lt20_of63"]) for m in methods])/63
    ax.plot(ks, high, "o-", color="0.1", label="high-budget mean")
    ax.set_xlabel("proximal actor depth $K$"); ax.set_xticks(ks)
    ax.set_ylabel("mean score", color="0.1"); ax.grid(True, alpha=.22)
    axr=ax.twinx(); axr.plot(ks,col,"s--",color="0.6",label="collapse rate")
    axr.set_ylabel("cell collapse rate", color="0.4")
    ax.set_title("Proximal fixed-budget depth")
    save(fig,"k4_analysis")

    # K1 replication.
    with (DATA/"k1_replication_summary.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    fig, ax=plt.subplots(figsize=(3.35,2.25),constrained_layout=True)
    x=np.arange(2)
    means=[float(r["mean_T_ge_4"]) for r in rows]
    cells=[int(r["collapsed_cells_lt20_of63"]) for r in rows]
    ax.bar(x,means,width=.55,color=["0.35","0.7"],edgecolor="black",linewidth=.4)
    ax.set_xticks(x,["seeds 0-1","seeds 2-3"])
    ax.set_ylabel("high-budget mean")
    ax.set_title("Independent one-hop seed pair")
    for i,(m,c) in enumerate(zip(means,cells,strict=True)):
        ax.text(i,m+1.2,f"{m:.2f}\n{c}/63 collapsed",ha="center",va="bottom",fontsize=7)
    ax.set_ylim(0,max(means)+9); ax.grid(True,axis="y",alpha=.2)
    save(fig,"k1_replication")

    # Diagnostic overview, log transform for scale.
    labels=["Lin stable","Lin collapsed","Prox stable","Prox collapsed"]
    td=np.array([270.35,9.5968e23,160.77,5.0305e21])
    dfirst=np.array([.1118,.5384,.0644,.4459])
    gain=np.array([2.1109,-1.4224,2.5111,-1.0647])
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.25),constrained_layout=True)
    x=np.arange(4)
    axes[0].bar(x,np.log10(1+td),color="0.55",edgecolor="black",linewidth=.3)
    axes[0].set_xticks(x,labels,rotation=28,ha="right",fontsize=6.3)
    axes[0].set_ylabel("$\\log_{10}(1+\\mathrm{TD}_{p99})$")
    axes[0].set_title("(a) TD residual tails")
    axes[1].bar(x,dfirst,color="0.55",edgecolor="black",linewidth=.3)
    axes[1].set_xticks(x,labels,rotation=28,ha="right",fontsize=6.3)
    axes[1].set_ylabel("first-actor displacement")
    axes[1].set_title("(b) Realized movement")
    axes[2].bar(x,gain,color="0.55",edgecolor="black",linewidth=.3)
    axes[2].axhline(0,color="black",linewidth=.7)
    axes[2].set_xticks(x,labels,rotation=28,ha="right",fontsize=6.3)
    axes[2].set_ylabel("common-critic gain")
    axes[2].set_title("(c) Cross-critic disagreement")
    for a in axes: a.grid(True,axis="y",alpha=.2,linewidth=.4)
    save(fig,"diagnostic_summary")


if __name__ == "__main__":
    stability()
    supplemental()
    print(f"wrote figures to {OUT}")
