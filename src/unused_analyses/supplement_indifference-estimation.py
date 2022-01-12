#!/usr/bin/python
"""
Eye movements in context effects
This script performs analyses of indifference estimation results.
    0. Concatenating and saving indifference estimation results for subjects included in the analysis
    1. Test, whether increases or decreases across blocks for a given option significantly differed from 0 across the group
    2. Alternative analysis: fit hierarchical linear model to the estimation data, testing whether estimates for an option increase linearly with block
Author: Felix Molter, felixmolter@gmail.com
"""

import logging
import warnings
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from bambi import Model

from analysis import best

warnings.simplefilter(action="ignore", category=FutureWarning)

# Set seed
SEED = 91
np.random.seed(SEED)

# Load behavioural data
trials = pd.read_csv(join("..", "results", "clean_data", "trials.csv"), index_col=0)

subjects = trials["subject"].unique()

# 0. Concatenating and saving indifference estimation results for subjects included in the analysis
# -------------------------------------------------------------------------------------------------

estimation_results = []
for subject in subjects:
    er_s = pd.read_csv(
        join(
            "..",
            "data",
            "indifference_estimation",
            "comp_ind-estimation_{}.csv".format(subject),
        )
    )
    estimation_results.append(er_s)
estimation_results = pd.concat(estimation_results)

estimation_results = estimation_results.rename({"subject_id": "subject"}, axis=1)

estimation_results.to_csv(
    join("..", "results", "clean_data", "indifference_estimation.csv"), index=False
)
print(
    "Wrote indifference estimation results to '../results/clean_data/indifference_estimation.csv'."
)

# calculate individual increases / decreases of estimates per block, always relative to the previous one
deltas = []
for subject in subjects:
    er_s = estimation_results.loc[estimation_results["subject"] == subject]
    deltas_s = pd.DataFrame(dict(subject=np.ones(2) * subject, block=[2, 3]))
    for alt in ["CBm", "Bm", "Am", "CAm"]:
        deltas_s["d" + alt[:-1]] = [
            er_s[alt].values[1] - er_s[alt].values[0],
            er_s[alt].values[2] - er_s[alt].values[1],
        ]
    deltas.append(deltas_s)
deltas = pd.concat(deltas).reset_index(drop=True)

# calculate mean change in estimation per subject
mean_deltas = deltas.groupby("subject")[["dCB", "dB", "dA", "dCA"]].mean()
mean_deltas.describe().to_csv(
    join("..", "results", "behaviour", "s-1_indifference-estimation_deltas_summary.csv")
)

# 1. Test, whether increases or decreases across blocks for a given option significantly differed from 0 across the group
# -----------------------------------------------------------------------------------------------------------------------

print(
    "Testing for each option whether increases/decreases across blocks differ from 0 across the group."
)
print(
    "\tRunning 1-Sample BEST of changes to previous block in indifference point against 0."
)

for alt in ["dCB", "dA", "dCA"]:
    print("\tOption:", alt[1:])
    trace = best.runBEST1G(
        y=mean_deltas[alt].values, mu=0, sigma_low=0.0001, sigma_high=100, seed=SEED
    )
    pm.summary(trace).to_csv(
        join(
            "..",
            "results",
            "behaviour",
            "s-1_indifference-estimation_BEST1G_delta-{alt}-v-0_posterior.csv".format(
                alt=alt
            ),
        )
    )
    pm.plot_posterior(
        trace, var_names=["difference", "d"], ref_val=0, credible_interval=0.95
    )
    plt.savefig(
        join(
            "..",
            "results",
            "behaviour",
            "s-1_indifference-estimation_BEST1G_delta-{alt}-v-0_posterior.png".format(
                alt=alt
            ),
        )
    )


# 2. Hierarchical linear model to the estimation data, testing whether estimates for an option increase linearly with block
# -----------------------------------------------------------------------------------------------------------------------

# reformat block variable so that first block is 0
estimation_results["block"] = estimation_results["block"] - 1

print("Running hierarchical linear model predicting indifference points from block.")
print("estimated-outcome ~ 1|subject + block|subject")

for alt in ["CB", "A", "CA"]:
    print("Option:", alt)
    model = Model(estimation_results)
    formula = alt + "m ~ 0 + block"
    results = model.fit(formula, random=["1|subject", "block|subject"])

    results.summary().to_csv(
        join(
            "..",
            "results",
            "behaviour",
            "s-1_indifference-estimation_bayesReg_m-block_{alt}_summary.csv".format(
                alt=alt
            ),
        )
    )
