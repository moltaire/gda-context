#!/usr/bin/python
"""
Gaze-dependent accumulation in context-dependent risky choice
This script performs further preprocessing of dwell and fixation data
    1. Compute trial-wise relative dwells and merge them into trial dataframe
    2. Save combined dataframe (trials_with-dwells)
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

fixations = pd.read_csv(
    join(DATA_DIR, "fixations.csv"), index_col=0
)

# 1. Compute trial-wise relative dwells and merge them into trial dataframe
# -------------------------------------------------------------------------

# Aggregate dwells per option, merge to trial df
dwells_trial = {}
# Alternatives (across attributes)
for alternative in ["A", "B", "C"]:
    dwells_trial["dwell_" + alternative] = dwells.groupby(["subject", "trial"]).apply(
        lambda x: x.loc[(x["alternative"] == alternative), "duration"].sum()
    )
# Attributes (across alternatives)
for attribute in ["p", "m"]:
    dwells_trial["dwell_" + attribute] = dwells.groupby(["subject", "trial"]).apply(
        lambda x: x.loc[(x["attribute"] == attribute), "duration"].sum()
    )
# Specific AOIs (combinations of alternative and attribute)
for alternative in ["A", "B", "C"]:
    for attribute in ["p", "m"]:
        dwells_trial["dwell_" + alternative + attribute] = dwells.groupby(
            ["subject", "trial"]
        ).apply(
            lambda x: x.loc[
                (x["alternative"] == alternative) & (x["attribute"] == attribute),
                "duration",
            ].sum()
        )
dwells_trial = pd.DataFrame(dwells_trial).reset_index()

# make things relative
dwells_trial["dwell_total"] = dwells_trial[["dwell_A", "dwell_B", "dwell_C"]].sum(
    axis=1
)
for col in [
    col
    for col in dwells_trial.columns
    if col.startswith("dwell_") and col != "dwell_total"
]:
    dwells_trial[col] /= dwells_trial["dwell_total"]

trials = trials.merge(dwells_trial, on=["subject", "trial"])

# identify dwells to target, competitor and decoy
trials["dwell_target"] = np.where(
    pd.isnull(trials["target"]),
    np.nan,
    np.where(trials["target"] == "A", trials["dwell_A"], trials["dwell_B"]),
)

trials["dwell_competitor"] = np.where(
    pd.isnull(trials["target"]),
    np.nan,
    np.where(trials["target"] == "A", trials["dwell_B"], trials["dwell_A"]),
)
trials["dwell_decoy"] = np.where(pd.isnull(trials["target"]), np.nan, trials["dwell_C"])


# 2. Save combined dataframe (trials_with-dwells)
# -----------------------------------------------
trials.to_csv(
    join(OUTPUT_DIR, "trials_with-dwells.csv"), index=False
)
