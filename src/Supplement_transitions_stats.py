#!usr/bin/python
"""
Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour

This script...

1. Summarises transition and search direction data
2. Performs statistical tests on the difference of measures
between attraction and compromise trials.

felixmolter@gmail.com
"""

import logging
import warnings
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arviz import plot_trace, summary

from analysis.best import runBEST1G
from analysis.utilities import makeDirIfNeeded

# Suppress pymc3 warnings
logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)

warnings.simplefilter(action="ignore", category=FutureWarning)


# Set directories
RESULTS_DIR = join("..", "results")
FIGURE_OUTPUT_DIR = join("..", "figures")
OUTPUT_DIR = join(RESULTS_DIR, "S_supplemental-gaze-analyses", "transitions")
makeDirIfNeeded(OUTPUT_DIR)

# MCMC Settings, passed on to pm.sample
SEED = 1
sample_kwargs = {"cores": 1, "random_seed": SEED, "progressbar": False}

# Load transitions DataFrame (each row contains transition counts and statistics for a single trial)
transitions = pd.read_csv(
    join("..", "results", "0-clean_data", "transitions.csv"), index_col=0
)
# Load trial data for additional trial variables
trials = pd.read_csv(join("..", "results", "0-clean_data", "trials.csv"), index_col=0)
# Add effect column to transition dataframe
transitions = transitions.merge(
    trials[["subject", "trial", "effect"]], on=["subject", "trial"]
)

# 1. Summarise transition and search direction data
# Compute subject level means for transition counts, Payne Index
transitions_subject = (
    transitions.loc[transitions["effect"].isin(["attraction", "compromise"])]
    .groupby(["subject", "effect"])[
        [
            "n_horizontal",
            "n_vertical",
            "n_diagonal",
            "payne_index",
        ]
    ]
    .mean()
    .reset_index()
)
output_file = join(OUTPUT_DIR, "transitions_subject-summary.csv")
transitions_subject.round(4).to_csv(output_file, index=False)
print(f"\tOutput file created at '{output_file}'.")


# Compute summary statistics
# Make summary dataframe
transitions_summary = (
    transitions_subject.drop("subject", axis=1)
    .groupby("effect")
    .agg(["mean", "std"])
    .round(2)
)
output_file = join(OUTPUT_DIR, "transitions_summary.csv")
transitions_summary.round(4).to_csv(output_file)
print(f"\tOutput file created at '{output_file}'.")


# 2. Perform statistical tests on the difference of measures between attraction and compromise trials
print(
    "Running 1-Sample BEST comparisons between transition measures in attraction and compromise trials..."
)
for measure in ["n_horizontal", "n_vertical", "n_diagonal", "payne_index"]:

    # Format data and compute paired difference between attraction and compromise trials
    df = transitions_subject.pivot(index="subject", columns="effect", values=measure)
    df["difference"] = df["attraction"] - df["compromise"]

    # Run paired BEST
    trace = runBEST1G(
        y=df["difference"],
        mu=0,
        sigma_low=0.01,
        sigma_high=100,
        sample_kwargs=sample_kwargs,
    )

    # Save summary
    summary_df = summary(trace, hdi_prob=0.95)
    output_file = join(
        OUTPUT_DIR, f"transitions_{measure}_attr-vs-comp_BEST_summary.csv"
    )
    summary_df.round(4).to_csv(output_file)
    print(f"\tOutput file created at '{output_file}'.")

    # Save traceplot
    plot_trace(trace)
    plt.savefig(
        join(OUTPUT_DIR, f"transitions_{measure}_attr-vs-comp_BEST_traceplot.png"),
        dpi=50,
    )
    plt.close()

    # Print statistics
    print(
        f"\n{measure}:\n"
        + f"  Attraction: {transitions_summary.loc['attraction', measure]['mean']:.2f} ± "
        + f"{transitions_summary.loc['attraction', measure]['std']:.2f}\n"
        + f"  Compromise: {transitions_summary.loc['compromise', measure]['mean']:.2f} ± "
        + f"{transitions_summary.loc['compromise', measure]['std']:.2f}\n"
        + f"  mean difference = {summary_df.loc['mean', 'mean']} [{summary_df.loc['mean', 'hdi_2.5%']}, {summary_df.loc['mean', 'hdi_97.5%']}]\n"
        + f"  d = {summary_df.loc['d', 'mean']} [{summary_df.loc['d', 'hdi_2.5%']}, {summary_df.loc['d', 'hdi_97.5%']}]\n"
    )
