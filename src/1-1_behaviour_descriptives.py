#!/usr/bin/python
"""
Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour
This script generates descriptives of behavioural data
    1. Compute subject-wise choice shares for ABC, TCD, split by target alternative
    2. Compute subject-wise choice shares for ABC, TCD, across target alternatives
    3. Count and print number of participants with RST above 0.5
    4. Count and print number of strong (RST > 0.7) ADE responders
Author: Felix Molter, felixmolter@gmail.com
"""
from os import makedirs
from os.path import exists, join

import numpy as np
import pandas as pd

# Set seed
SEED = 11
np.random.seed(SEED)

# Set data and output directories
DATA_DIR = join("..", "results", "0-clean_data")
OUTPUT_DIR = join("..", "results", "1-behaviour")
if not exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)


# Load behavioural data
trials = pd.read_csv(join(DATA_DIR, "trials.csv"), index_col=0)

# 1. Compute subject-wise choice shares for ABC, TCD, split by target alternative
# -------------------------------------------------------------------------------

# Split by target alternative
## Compute mean ± sd choice shares for options A, B, C, by effect, target alternative
cs_abc = (
    trials.groupby(["subject", "effect", "target"])["choice"]
    .value_counts(normalize=True)
    .rename("share")
    .reset_index()
    .rename({"target": "target_alternative"}, axis=1)
    .pivot_table(
        values="share",
        columns="choice",
        index=["subject", "effect", "target_alternative"],
    )
    .fillna(0)
    .reset_index()
)
cs_abc.columns.name = None


## Compute mean ± s.d. choice shares for options T, C, D, by effect, target alternative
cs_tcd = (
    trials.groupby(["subject", "effect", "target"])["choice_tcd"]
    .value_counts(normalize=True)
    .rename("share")
    .reset_index()
    .rename({"target": "target_alternative"}, axis=1)
    .pivot_table(
        values="share",
        columns="choice_tcd",
        index=["subject", "effect", "target_alternative"],
    )
    .fillna(0)
    .reset_index()
)
cs_tcd["rst"] = cs_tcd["target"] / (cs_tcd["target"] + cs_tcd["competitor"])
cs_tcd.columns.name = None

## Combine and save dataframes
cs = cs_abc.merge(right=cs_tcd, on=["subject", "effect", "target_alternative"])
output_file = join(OUTPUT_DIR, "choiceshares_by-target.csv")
cs.to_csv(output_file, index=False)
print(f"Created dataframe of choice shares by target alternative at '{output_file}'.")

## Create summary
cs_summary = (
    cs.groupby(["effect", "target_alternative"])[
        ["A", "B", "C", "target", "competitor", "decoy", "rst"]
    ].agg(["mean", "std", "min", "max"])
).T
output_file = join(OUTPUT_DIR, "choiceshares_by-target_summary.csv")
cs_summary.to_csv(output_file)
print(f"Created summary of choice shares by target alternative at '{output_file}'.")

# 2. Compute subject-wise choice shares for ABC, TCD, across target alternatives
# ------------------------------------------------------------------------------

## ABC
cs_abc = (
    trials.groupby(["subject", "effect"])["choice"]
    .value_counts(normalize=True)
    .rename("share")
    .reset_index()
    .pivot_table(values="share", columns="choice", index=["subject", "effect"])
    .fillna(0)
    .reset_index()
)
cs_abc.columns.name = None

## TCD
cs_tcd = (
    trials.groupby(["subject", "effect"])["choice_tcd"]
    .value_counts(normalize=True)
    .rename("share")
    .reset_index()
    .pivot_table(values="share", columns="choice_tcd", index=["subject", "effect"])
    .fillna(0)
    .reset_index()
)
cs_tcd["rst"] = cs_tcd["target"] / (cs_tcd["target"] + cs_tcd["competitor"])
cs_tcd.columns.name = None

## Combine and save dataframes
cs_across = cs_abc.merge(right=cs_tcd, on=["subject", "effect"])
output_file = join(OUTPUT_DIR, "choiceshares_across-targets.csv")
cs_across.to_csv(output_file, index=False)
print(
    f"Created dataframe of choice shares across target alternatives at '{output_file}'."
)

## Create summary
cs_across_summary = (
    cs_across.groupby(["effect"])[
        ["A", "B", "C", "target", "competitor", "decoy", "rst"]
    ].agg(["mean", "std", "min", "max"])
).T
output_file = join(OUTPUT_DIR, "choiceshares_across-targets_summary.csv")
cs_across_summary.to_csv(output_file)
print(
    f"Created summary of choice shares across target alternatives at '{output_file}'."
)

# 3. Count and print number of participants with RST above 0.5
# ------------------------------------------------------------

cs_across["rst_above_0.5"] = cs_across["rst"] > 0.5
print("RST > 0.5")
print(cs_across.groupby("effect")["rst_above_0.5"].value_counts())

# 4. Count and print number of strong (RST > 0.7) ADE responders
# --------------------------------------------------------------

print("Strong (RST > 0.7) ADE responders:")
print((cs_across.loc[cs_across["effect"] == "attraction", "rst"] > 0.7).sum())
