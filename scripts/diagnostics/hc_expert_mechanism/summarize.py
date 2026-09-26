#!/usr/bin/env python3
"""Summarize paired training-seed contrasts; retain state-level rows for audit."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    episodes = pd.read_csv(args.output / "episodes.csv")
    blocks = pd.read_csv(args.output / "time_blocks.csv")
    mc = pd.read_csv(args.output / "counterfactual_mc.csv")

    scores = episodes.groupby(["T", "seed", "actor", "branch"], as_index=False).agg(
        score=("score", "mean"), episode_length=("length", "mean")
    )
    paired = scores[scores.actor.isin([3, 4])].pivot(
        index=["T", "seed", "actor"], columns="branch", values="score"
    ).reset_index()
    paired["recenter_minus_fixed"] = paired["recenter"] - paired["fixed_ref"]
    paired.to_csv(args.output / "paired_stage_scores.csv", index=False)

    mc["ranking_disagrees"] = ((mc.q1_delta * mc.mc_delta) < 0)
    mc_summary = mc.groupby(["T", "seed", "actor", "source_branch"], as_index=False).agg(
        states=("mc_delta", "size"),
        critic_adv_recenter=("q1_delta", "mean"),
        mc_adv_recenter=("mc_delta", "mean"),
        ranking_disagrees=("ranking_disagrees", "mean"),
    )
    mc_summary.to_csv(args.output / "mc_by_pair.csv", index=False)

    block = blocks.groupby(["T", "seed", "actor", "branch", "block_start"], as_index=False).agg(
        mean_reward=("reward_mean", "mean"),
        mean_velocity=("velocity_mean", "mean"),
        mean_saturation=("saturation_mean", "mean"),
        episodes_reaching_block=("episode", "nunique"),
    )
    block.to_csv(args.output / "blocks_by_pair.csv", index=False)
    print("STAGE SCORES (paired unit: training seed)")
    print(paired.round(2).to_string(index=False))
    print("\nONE-ACTION THEN π1: SAME-STATE CRITIC vs MC ACTION PREFERENCE")
    print(mc_summary.round(2).to_string(index=False))
    print("\nInspect blocks_by_pair.csv for when velocity, reward and action saturation diverge.")
    print("MC states/episodes are repeated observations, not independent training runs.")


if __name__ == "__main__":
    main()
