#!/usr/bin/python
"""
Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour
This script generates descriptives of gaze data
    1. Compute subject level mean ± s.d. and grand average of dwells towards ABC, by effect and target alternative
    2. Compute subject level mean ± s.d. and grand average of dwells towards ABC, by effect across target alternatives
    3. Compute subject level mean ± s.d. and grand average of Payne Indices for attraction and compromise trials
Author: Felix Molter, felixmolter@gmail.com
"""

from os.path import join, exists
from os import makedirs

import numpy as np
import pandas as pd

# Set seed
np.random.seed(21)

# Set data and output directories
DATA_DIR = join("..", "results", "0-clean_data")
OUTPUT_DIR = join("..", "results", "2-gaze")
if not exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)

# Load trial data with trial-wise dwell information
trials = pd.read_csv(join(DATA_DIR, "trials_with-dwells.csv"))

# Load transition data (each line is one trial)
transitions = pd.read_csv(join(DATA_DIR, "transitions.csv"))
trials = trials.merge(transitions, on=["subject", "trial"])

# 1. Compute subject level mean ± s.d. and grand average of dwells towards ABC, by effect and target alternative
# --------------------------------------------------------------------------------------------------------------

# Subject level
rdwells = (
    trials.loc[
        trials["effect"].isin(["attraction", "compromise"]),
        [
            "subject",
            "effect",
            "target",
            "dwell_A",
            "dwell_B",
            "dwell_C",
            "dwell_p",
            "dwell_m",
            "dwell_target",
            "dwell_competitor",
            "dwell_decoy",
        ],
    ]
    .rename({"target": "target_alternative"}, axis=1)
    .groupby(["subject", "effect", "target_alternative"])
    .mean()
    .reset_index()
)

output_file = join(OUTPUT_DIR, "dwells_by-target.csv")
rdwells.round(4).to_csv(output_file, index=False)
print(
    f"Created participant level summary of relative dwells by target at '{output_file}'."
)

# Grand average
rdwells_summary = (
    rdwells.groupby(["effect", "target_alternative"])[
        [
            "dwell_A",
            "dwell_B",
            "dwell_C",
            "dwell_p",
            "dwell_m",
            "dwell_target",
            "dwell_competitor",
            "dwell_decoy",
        ]
    ].agg(["mean", "std", "min", "max"])
).T

output_file = join(OUTPUT_DIR, "dwells_by-target_summary.csv")
rdwells_summary.round(4).to_csv(output_file)
print(f"Created grand average summary of relative dwells by target at '{output_file}'.")


# 2. Compute subject level mean ± s.d. and grand average of dwells towards ABC, by effect across target alternatives
# ------------------------------------------------------------------------------------------------------------------

# Subject level
rdwells_across = (
    trials.loc[
        trials["effect"].isin(["attraction", "compromise"]),
        [
            "subject",
            "effect",
            "dwell_A",
            "dwell_B",
            "dwell_C",
            "dwell_p",
            "dwell_m",
            "dwell_target",
            "dwell_competitor",
            "dwell_decoy",
        ],
    ]
    .groupby(["subject", "effect"])
    .mean()
    .reset_index()
)
output_file = join(OUTPUT_DIR, "dwells_across-targets.csv")
rdwells_across.round(4).to_csv(output_file, index=False)
print(
    f"Created participant level summary of relative dwells across targets at '{output_file}'."
)

# Grand average
rdwells_across_summary = (
    rdwells_across.groupby("effect")[
        [
            "dwell_A",
            "dwell_B",
            "dwell_C",
            "dwell_p",
            "dwell_m",
            "dwell_target",
            "dwell_competitor",
            "dwell_decoy",
        ]
    ]
    .agg(["mean", "std", "min", "max"])
    .T
)

output_file = join(OUTPUT_DIR, "dwells_across-targets_summary.csv")
rdwells_across_summary.round(4).to_csv(output_file)
print(
    f"Created grand average summary of relative dwells across targets at '{output_file}'."
)


# 3. Compute subject level mean ± s.d. and grand average of Payne Indices for attraction and compromise trials
# ------------------------------------------------------------------------------------------------------------

pi_part_across = (
    trials.loc[trials["effect"].isin(["attraction", "compromise"])]
    .groupby(["subject", "effect"])["payne_index"]
    .mean()
    .reset_index()
)
output_file = join(OUTPUT_DIR, "payne-index_across-targets.csv")
pi_part_across.round(4).to_csv(output_file, index=False)
print(f"Created participant level summary of Payne Indices at '{output_file}'.")

# Grand average
pi_across_summary = pi_part_across.groupby("effect")["payne_index"].agg(
    ["mean", "std", "min", "max"]
)
output_file = join(OUTPUT_DIR, "payne-index_across-targets_summary.csv")
pi_across_summary.round(4).to_csv(output_file)
print(f"Created grand average summary of Payne Indices at '{output_file}'.")
