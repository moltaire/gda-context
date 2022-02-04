#!/usr/bin/python
"""
Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour
This script processes the eyetracking data to create a dataframe containing transitions between AOIs
    1. Create transition dataframe
    2. Save dataframe (transitions)
Author: Felix Molter, felixmolter@gmail.com
"""

from os.path import join

import numpy as np
import pandas as pd

DATA_DIR = join("..", "results", "0-clean_data")
OUTPUT_DIR = DATA_DIR

# Load data
trials = pd.read_csv(join(DATA_DIR, "trials.csv"), index_col=0)
dwells = pd.read_csv(join(DATA_DIR, "dwells.csv"), index_col=0)

# 1. Create transition dataframe
# ------------------------------

# Add columns for previous dwell
dwells["prev_alt"] = dwells.groupby(["subject", "trial"])["alternative"].shift(
    1, fill_value=np.nan
)

dwells["prev_att"] = dwells.groupby(["subject", "trial"])["attribute"].shift(
    1, fill_value=np.nan
)

dwells["prev_col"] = dwells.groupby(["subject", "trial"])["col"].shift(
    1, fill_value=np.nan
)

dwells["prev_row"] = dwells.groupby(["subject", "trial"])["row"].shift(
    1, fill_value=np.nan
)

# Determine transitions
dwells["trans_within-between"] = np.where(
    pd.isnull(dwells["prev_alt"]),
    np.nan,
    np.where(
        dwells["prev_alt"] == dwells["alternative"],
        "within-alt",
        np.where(
            dwells["prev_att"] == dwells["attribute"],
            "between-alt-same",
            "between-alt-diff",
        ),
    ),
)

dwells["trans_distance"] = np.abs(dwells["col"] - dwells["prev_col"])

dwells["trans_direction"] = np.where(
    pd.isnull(dwells["prev_col"]),
    np.nan,
    np.where(
        dwells["prev_col"] == dwells["col"],
        "vertical",
        np.where(dwells["prev_row"] == dwells["row"], "horizontal", "diagonal"),
    ),
)

# Aggregate by trial and save as transitions
trans_withinbetween = (
    dwells.groupby(["subject", "trial"])["trans_within-between"]
    .value_counts(dropna=True)
    .rename("count")
    .reset_index()
    .pivot_table(
        values="count", columns="trans_within-between", index=["subject", "trial"]
    )
    .reset_index()
    .drop("nan", axis=1)
    .fillna(0)
)
trans_withinbetween.columns.name = None
trans_withinbetween["payne_index"] = (
    trans_withinbetween["within-alt"] - trans_withinbetween["between-alt-same"]
) / (trans_withinbetween["within-alt"] + trans_withinbetween["between-alt-same"])
trans_withinbetween["payne_index_relaxed"] = (
    trans_withinbetween["within-alt"]
    - (
        trans_withinbetween["between-alt-same"]
        + trans_withinbetween["between-alt-diff"]
    )
) / (
    trans_withinbetween["within-alt"]
    + (
        trans_withinbetween["between-alt-same"]
        + trans_withinbetween["between-alt-diff"]
    )
)

# Distance: Aggregate by trial and save as transitions
trans_distance = (
    dwells.groupby(["subject", "trial"])["trans_distance"]
    .value_counts(dropna=True)
    .rename("count")
    .reset_index()
    .pivot_table(values="count", columns="trans_distance", index=["subject", "trial"])
    .reset_index()
    .fillna(0)
)
trans_distance.columns.name = None

# Direction: Aggregate by trial and save as transitions
trans_direction = (
    dwells.groupby(["subject", "trial"])["trans_direction"]
    .value_counts(dropna=True)
    .rename("count")
    .reset_index()
    .pivot_table(values="count", columns="trans_direction", index=["subject", "trial"])
    .reset_index()
    .drop("nan", axis=1)
    .fillna(0)
)
trans_direction.columns.name = None

# Direction: Target-Competitor_decoy
# Add target, effect columns to dwell dataframe
dwells = dwells.merge(
    trials[["subject", "trial", "effect", "target"]], on=["subject", "trial"]
)
dwells["alternative_tcd"] = np.where(
    dwells["alternative"] == "C",
    "decoy",
    np.where(dwells["alternative"] == dwells["target"], "target", "competitor"),
)

dwells["prev_alt_tcd"] = dwells.groupby(["subject", "trial"])["alternative_tcd"].shift(
    1, fill_value=np.nan
)

dwells["trans_tcd_directed"] = (
    dwells["prev_alt_tcd"].str[0] + "-" + dwells["alternative_tcd"].str[0]
)

dwells["trans_tcd"] = np.where(
    pd.isnull(dwells["trans_tcd_directed"]),
    np.nan,
    np.where(
        dwells["trans_tcd_directed"].isin(["t-c", "c-t"]),
        "t-c",
        np.where(
            dwells["trans_tcd_directed"].isin(["t-d", "d-t"]),
            "t-d",
            np.where(
                dwells["trans_tcd_directed"].isin(["c-d", "d-c"]),
                "c-d",
                "within-alternative",
            ),
        ),
    ),
)

trans_tcd = (
    dwells.groupby(["subject", "trial"])["trans_tcd"]
    .value_counts(dropna=True)
    .rename("count")
    .reset_index()
    .pivot_table(values="count", columns="trans_tcd", index=["subject", "trial"])
    .reset_index()
    .drop("nan", axis=1)
    .fillna(0)
)

# Combine everything
transitions = (
    trans_withinbetween.merge(trans_distance, on=["subject", "trial"])
    .merge(trans_direction, on=["subject", "trial"])
    .merge(trans_tcd)
    .rename(
        {
            "between-alt-diff": "n_between-alt-diff",
            "between-alt-same": "n_between-alt-same",
            "within-alt": "n_within-alt",
            0.0: "n_dist0",
            1.0: "n_dist1",
            2.0: "n_dist2",
            "diagonal": "n_diagonal",
            "horizontal": "n_horizontal",
            "vertical": "n_vertical",
            "t-c": "n_target_competitor",
            "t-d": "n_target_decoy",
            "c-d": "n_competitor_decoy",
        },
        axis=1,
    )
)

# 2. Save dataframe (transitions)
# -------------------------------
transitions.to_csv(join(OUTPUT_DIR, "transitions.csv"))
