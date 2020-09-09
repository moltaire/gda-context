#!/usr/bin/python
"""
Eye movements in context effects
This script performs confirmatory statistical analyses of the gaze data
    1. One-sample (paired) BEST of Dwell(Target) - Dwell(Competitor) vs. 0, for attraction and compromise trials.
    2. Bayesian correlation between RST and Payne Index, for attraction and compromise trials
Author: Felix Molter, felixmolter@gmail.com
"""

import logging
import warnings
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from analysis import best, bayescorr

logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)


warnings.simplefilter(action="ignore", category=FutureWarning)


# Set seed
SEED = 22
np.random.seed(SEED)

# MCMC settings (passed on to pm.sample)
sample_kwargs = {"cores": 1, "random_seed": SEED, "progressbar": False}

# Set data and output directories
RESULTS_DIR = join("..", "results")
OUTPUT_DIR = join(RESULTS_DIR, "2-gaze")

# Load relative dwell data
rdwells = pd.read_csv(join(OUTPUT_DIR, "dwells_across-targets.csv"))

# Load choiceshare data
cs = pd.read_csv(join(RESULTS_DIR, "1-behaviour", "choiceshares_across-targets.csv"))

# Load Payne Index data
pi = pd.read_csv(join(RESULTS_DIR, "2-gaze", "payne-index_across-targets.csv"))

# 1. Run One-sample (paired) BEST of Dwell to Target - Dwell to Competitor
# ------------------------------------------------------------------------

print("Running Paired Sample BESTs: Dwell to Target vs Dwell to Competitor")
for effect in ["attraction", "compromise"]:
    print("\t" + effect.capitalize() + " trials")
    diffTC = (
        rdwells.loc[rdwells["effect"] == effect, "dwell_target"]
        - rdwells.loc[rdwells["effect"] == effect, "dwell_competitor"]
    )
    trace = best.runBEST1G(
        diffTC, mu=0.0, sigma_low=0.001, sigma_high=10.0, sample_kwargs=sample_kwargs
    )

    summary = pm.summary(trace, hdi_prob=0.95)
    for var in ["mean", "difference", "d"]:
        summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
    summary.to_csv(
        join(OUTPUT_DIR, f"dwell_target_vs_competitor_{effect}_BEST_summary.csv")
    )

    print(
        f"\tmean difference = {summary.loc['difference', 'mean']} ± [{summary.loc['difference', 'hdi_2.5%']}, {summary.loc['difference', 'hdi_97.5%']}]"
    )
    print(
        f"\td = {summary.loc['d', 'mean']} ± [{summary.loc['d', 'hdi_2.5%']}, {summary.loc['d', 'hdi_97.5%']}]"
    )

    print(
        "\t{}% of posterior mass above 0.".format(
            100 * np.mean(trace.get_values("d") > 0)
        )
    )
    pm.traceplot(trace)
    plt.savefig(
        join(OUTPUT_DIR, f"dwell_target_vs_competitor_{effect}_BEST_trace.png"),
        dpi=100,
    )
    pm.plot_posterior(
        trace,
        var_names=["difference", "d"],
        hdi_prob=0.95,
        round_to=2,
        ref_val=0.0,
        figsize=(10, 2.5),
    )
    plt.savefig(
        join(OUTPUT_DIR, f"dwell_target_vs_competitor_{effect}_BEST_posterior.png",),
        dpi=100,
    )


# 2. Run Bayesian Correlation of RST vs. Payne Index
# -------------------------------------------------

# Load data and align by subject column
df = pi.copy()
df = df.merge(cs[["subject", "effect", "rst"]], on=["subject", "effect"])

print("Running Bayesian Correlation of RST vs Payne Index")
for effect in ["attraction", "compromise"]:
    print("\t" + effect.capitalize() + " trials")
    df_e = df.loc[df["effect"] == effect]

    trace = bayescorr.runBayesCorr(
        y1=df_e["rst"], y2=df_e["payne_index"], sample_kwargs=sample_kwargs
    )

    summary = pm.summary(trace, hdi_prob=0.95)

    summary.loc["r", "p>0"] = np.mean(trace.get_values("r") > 0)
    summary.to_csv(
        join(OUTPUT_DIR, f"rst_payne-index_{effect}_correlation_summary.csv")
    )

    print(
        f"\tr = {summary.loc['r', 'mean']} ± [{summary.loc['r', 'hdi_2.5%']}, {summary.loc['r', 'hdi_97.5%']}]"
    )

    print(
        "\t{}% of posterior mass above 0.".format(
            100 * np.mean(trace.get_values("r") > 0)
        )
    )
    pm.traceplot(trace)
    plt.savefig(
        join(OUTPUT_DIR, f"rst_payne-index_{effect}_correlation_trace.png"), dpi=100
    )

    pm.plot_posterior(
        trace,
        var_names="r",
        hdi_prob=0.95,
        round_to=2,
        ref_val=0.0,
        figsize=(2.5, 2.5),
    )
    plt.savefig(
        join(OUTPUT_DIR, f"rst_payne-index_{effect}_correlation_posterior.png"), dpi=100
    )
