#!/usr/bin/python
"""
Gaze-dependent accumulation in context-dependent risky choice
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

# Combine everything
transitions = (
    trans_withinbetween.merge(trans_distance, on=["subject", "trial"])
    .merge(trans_direction, on=["subject", "trial"])
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
        },
        axis=1,
    )
)

# 2. Save dataframe (transitions)
# -------------------------------
transitions.to_csv(join(OUTPUT_DIR, "transitions.csv"))
