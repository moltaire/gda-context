#!/usr/bin/python
"""
Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour
This script performs statistical tests of context effects
    1. Test context effects: One-sample BEST RST vs. 0.5
    2. Test context effects: One-sample (paired) BEST: P(Target) - P(Competitor)
    3. Correlation of RSTs between effects across participants
Author: Felix Molter, felixmolter@gmail.com
"""

import logging
import warnings
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from analysis import bayescorr, best

logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)

warnings.simplefilter(action="ignore", category=FutureWarning)

# Set seed
SEED = 12
np.random.seed(SEED)

# Set data and output directories
DATA_DIR = join("..", "results", "0-clean_data")
OUTPUT_DIR = join("..", "results", "1-behaviour")

# Load choice share data
cs = pd.read_csv(join(OUTPUT_DIR, "choiceshares_across-targets.csv"))

# MCMC Settings, passed on to pm.sample
sample_kwargs = {"cores": 1, "random_seed": SEED, "progressbar": False}

# 1. Test context effects: One-sample BEST RST vs. 0.5
# ----------------------------------------------------

print("Running 1 Sample BESTs: RST vs 0.5")
for effect in ["attraction", "compromise"]:
    print("\t" + effect.capitalize() + " trials")
    rst_effect = cs.loc[cs["effect"] == effect, "rst"]
    trace = best.runBEST1G(
        rst_effect,
        mu=0.5,
        sigma_low=0.001,
        sigma_high=10.0,
        sample_kwargs=sample_kwargs,
    )
    summary = pm.summary(trace, hdi_prob=0.95)
    for var in ["mean", "difference", "d"]:
        summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
    summary.to_csv(join(OUTPUT_DIR, f"rst_vs_0.5_{effect}_BEST_summary.csv"))

    print(
        f"\tmean difference = {summary.loc['difference', 'mean']} [{summary.loc['difference', 'hdi_97.5%']}, {summary.loc['difference', 'hdi_2.5%']}]"
    )
    print(
        f"\td = {summary.loc['d', 'mean']} [{summary.loc['d', 'hdi_2.5%']}, {summary.loc['d', 'hdi_97.5%']}]"
    )

    print(
        "\t{}% of posterior mass above 0.".format(
            100 * np.mean(trace.get_values("d") > 0)
        )
    )
    pm.traceplot(trace)
    plt.savefig(join(OUTPUT_DIR, f"rst_vs_0.5_{effect}_BEST_trace.png"), dpi=100)
    pm.plot_posterior(
        trace,
        var_names=["difference", "d"],
        hdi_prob=0.95,
        round_to=2,
        ref_val=0.0,
        figsize=(10, 2.5),
    )
    plt.savefig(join(OUTPUT_DIR, f"rst_vs_0.5_{effect}_BEST_posterior.png"), dpi=100)


# 2. Test context effects: One-sample (paired) BEST: P(Target) - P(Competitor)
# ----------------------------------------------------------------------------

print("Running Paired Sample BESTs: P(Target) vs P(Competitor)")
for effect in ["attraction", "compromise"]:
    print("\t" + effect.capitalize() + " trials")
    diffTC = (
        cs.loc[cs["effect"] == effect, "target"]
        - cs.loc[cs["effect"] == effect, "competitor"]
    )
    trace = best.runBEST1G(
        diffTC, mu=0.0, sigma_low=0.001, sigma_high=10.0, sample_kwargs=sample_kwargs
    )
    summary = pm.summary(trace, hdi_prob=0.95)
    for var in ["mean", "difference", "d"]:
        summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
    summary.to_csv(
        join(
            OUTPUT_DIR,
            f"cs_target_vs_competitor_{effect}_BEST_summary.csv",
        )
    )

    print(
        f"\tmean difference = {summary.loc['difference', 'mean']} [{summary.loc['difference', 'hdi_2.5%']}, {summary.loc['difference', 'hdi_97.5%']}]"
    )
    print(
        f"\td = {summary.loc['d', 'mean']} [{summary.loc['d', 'hdi_2.5%']}, {summary.loc['d', 'hdi_97.5%']}]"
    )

    print(
        "\t{}% of posterior mass above 0.".format(
            100 * np.mean(trace.get_values("d") > 0)
        )
    )
    pm.traceplot(trace)
    plt.savefig(
        join(OUTPUT_DIR, f"cs_target_vs_competitor_{effect}_BEST_trace.png"),
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
        join(OUTPUT_DIR, f"cs_target_vs_competitor_{effect}_BEST_posterior.png"),
        dpi=100,
    )


# 3. Running correlation analysis between Attraction and Compromise RSTs
# ----------------------------------------------------------------------------
print("Running correlation analysis between Attraction and Compromise RSTs")
rsts = cs.pivot(index="subject", columns="effect", values="rst")

trace = bayescorr.runBayesCorr(
    rsts["attraction"].values, rsts["compromise"].values, sample_kwargs=sample_kwargs
)

summary = pm.summary(trace, hdi_prob=0.95)
for var in ["r"]:
    summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
summary.to_csv(join(OUTPUT_DIR, "rst_attraction_compromise_correlation_summary.csv"))

print(
    f"\tr = {summary.loc['r', 'mean']} [{summary.loc['r', 'hdi_2.5%']}, {summary.loc['r', 'hdi_97.5%']}]"
)

print(
    "\t{}% of posterior mass above 0.".format(100 * np.mean(trace.get_values("r") > 0))
)
pm.traceplot(trace)
plt.savefig(
    join(OUTPUT_DIR, "rst_attraction_compromise_correlation_trace.png"), dpi=100
)
pm.plot_posterior(
    trace,
    var_names=["r"],
    hdi_prob=0.95,
    round_to=2,
    ref_val=0.0,
    figsize=(5, 2.5),
)
plt.savefig(
    join(OUTPUT_DIR, "rst_attraction_compromise_correlation_posterior.png"), dpi=100
)
