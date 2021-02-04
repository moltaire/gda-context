#!usr/bin/python
"""
Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour

This script performs multiple tests of the hypothesis that subjects with
strong attraction effects followed a simple choice rule based on the dominance
relationship between target and decoy.

1. Test 1: High AE associated with low RT in attraction trials?
2. Test 2: High AE associated Fixation count in attraction trials?
3. Test 3: High AE associated with fixation count after dominance known?
4. Test 4: Faster RTs for target vs non-target choices in attraction trials

felixmolter@gmail.com
"""
import logging
import warnings
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arviz import plot_trace, summary

from analysis.bayescorr import runBayesCorr
from analysis.best import runBEST1G
from analysis.utilities import makeDirIfNeeded

# Suppress pymc3 warnings
logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)

warnings.simplefilter(action="ignore", category=FutureWarning)


# Set directories
RESULTS_DIR = join("..", "results")
OUTPUT_DIR = join(RESULTS_DIR, "S_supplemental-choicerule-analysis")
makeDirIfNeeded(OUTPUT_DIR)

# MCMC Settings, passed on to pm.sample
SEED = 1
sample_kwargs = {"cores": 1, "random_seed": SEED, "progressbar": False}

# %% 0. Load trial, choiceshare and fixation data
# -----------------------------------------------
trials = pd.read_csv(join(RESULTS_DIR, "0-clean_data", "trials.csv"), index_col=0)

cs = pd.read_csv(join(RESULTS_DIR, "1-behaviour", "choiceshares_across-targets.csv"))

fixations = pd.read_csv(join(RESULTS_DIR, "0-clean_data", "fixations.csv"), index_col=0)

# %% Test 1: High AE associated with low RT in attraction trials?
# ---------------------------------------------------------------
print("Test 1: Correlation between Attraction RST and Attraction RT")

# calculate mean RT in attraction trials
rt_attr = trials.loc[trials["effect"] == "attraction"].groupby("subject")["rt"].mean()

# join with attraction RST
df = cs.loc[cs["effect"] == "attraction", ["subject", "rst"]].join(
    rt_attr, on="subject"
)

# Run Bayesian Correlation analysis
trace = runBayesCorr(y1=df["rst"], y2=df["rt"] / 1000, sample_kwargs=sample_kwargs)

summary_df = summary(trace, hdi_prob=0.95)
summary_df.to_csv(
    join(OUTPUT_DIR, "1_attraction-rst_attraction-rt_bayescorr_summary.csv")
)

plot_trace(trace)
plt.savefig(
    join(OUTPUT_DIR, "1_attraction-rst_attraction-rt_bayescorr_trace.png"), dpi=100
)
plt.close()

# Print statistics
print(
    f"  mean r = {summary_df.loc['r', 'mean']} [{summary_df.loc['r', 'hdi_2.5%']}, {summary_df.loc['r', 'hdi_97.5%']}]"
)

# %% Test 2: High AE associated Fixation count in attraction trials?
# ------------------------------------------------------------------
print(
    "Test 2: Correlation between Attraction RST and fixation count in attraction trials"
)

# Compute mean number of fixations in attraction trials
fixation_counts = (
    fixations.groupby(["subject", "trial"])["number"]
    .count()
    .rename("n_fixations")
    .reset_index()
)

# Create DataFrame with subject mean fixations in attraction trials and attraction RST
df = trials.merge(fixation_counts, on=["subject", "trial"], how="left")
df = df.loc[df["effect"] == "attraction"].groupby("subject")["n_fixations"].mean()
df = cs.loc[cs["effect"] == "attraction", ["subject", "rst"]].join(df, on="subject")

# Run Bayesian Correlation analysis
trace = runBayesCorr(y1=df["rst"], y2=df["n_fixations"], sample_kwargs=sample_kwargs)

summary_df = summary(trace, hdi_prob=0.95)
summary_df.to_csv(
    join(OUTPUT_DIR, "2_attraction-rst_attraction-fixation-count_bayescorr_summary.csv")
)

plot_trace(trace)
plt.savefig(
    join(OUTPUT_DIR, "2_attraction-rst_attraction-fixation-count_bayescorr_trace.png"),
    dpi=100,
)
plt.close()

# Print statistics
print(
    f"  mean r = {summary_df.loc['r', 'mean']} [{summary_df.loc['r', 'hdi_2.5%']}, {summary_df.loc['r', 'hdi_97.5%']}]"
)

# %% Test 3: High AE associated with fixation count after dominance known?
# ------------------------------------------------------------------------
print(
    "Test 3: Correlation between Attraction RST and fixation count in attraction trials after dominance known (all AOIs / any AOI of target and decoy seen once)"
)

# Add 'effect' and 'target' columns to fixation DataFrame
fixations = fixations.merge(
    trials[["subject", "trial", "effect", "target"]],
    on=["subject", "trial"],
    how="left",
)
# Mark and count fixations after dominance is known
for attribute in ["p", "m"]:
    fixations[f"to_decoy_{attribute}"] = (fixations["alternative"] == "C") & (
        fixations["attribute"] == attribute
    )
    fixations[f"seen_decoy_{attribute}"] = (
        fixations.groupby(["subject", "trial"])[f"to_decoy_{attribute}"].cumsum() > 0
    )

    fixations[f"to_target_{attribute}"] = (
        fixations["alternative"] == fixations["target"]
    ) & (fixations["attribute"] == attribute)
    fixations[f"seen_target_{attribute}"] = (
        fixations.groupby(["subject", "trial"])[f"to_target_{attribute}"].cumsum() > 0
    )

fixations["dominance_known_allatts"] = (
    fixations["seen_target_m"]
    & fixations["seen_target_p"]
    & fixations["seen_decoy_m"]
    & fixations["seen_decoy_p"]
)

fixations["dominance_known_anyatt"] = (
    fixations["seen_target_m"] | fixations["seen_target_p"]
) & (fixations["seen_decoy_m"] | fixations["seen_decoy_p"])

# Count number of fixations when target and decoy seen
nfix_tdseen = (
    fixations.loc[fixations["effect"] == "attraction"]
    .groupby(["subject", "trial"])["dominance_known_allatts", "dominance_known_anyatt"]
    .sum()
    .reset_index()
    .drop("trial", axis=1)
    .groupby("subject")
    .mean()
    .rename(
        {
            "dominance_known_allatts": "nfix_after_dom_allatts",
            "dominance_known_anyatt": "nfix_after_dom_anyatt",
        },
        axis=1,
    )
)

# Create DataFrame with attraction RST and fixation counts after dominance known (2 ways: all AOIs of Target and Decoy seen / Any AOI of Target and Decoy seen)
df = cs.loc[cs["effect"] == "attraction", ["subject", "rst"]].join(
    nfix_tdseen, on="subject"
)

for i, (measure, label) in enumerate(
    zip(
        ["allatts", "anyatt"],
        [
            "Fixation count after all attributes of Target and Decoy seen",
            "Fixation count after any attribute of both Target and Decoy seen",
        ],
    )
):
    print(f"  {label}")

    # Run Bayesian Correlation analysis
    trace = runBayesCorr(
        y1=df["rst"], y2=df[f"nfix_after_dom_{measure}"], sample_kwargs=sample_kwargs
    )

    summary_df = summary(trace, hdi_prob=0.95)
    summary_df.to_csv(
        join(
            OUTPUT_DIR,
            f"3-{i}_attraction-rst_fixation-count-after-dom-known-{measure}_bayescorr_summary.csv",
        )
    )

    plot_trace(trace)
    plt.savefig(
        join(
            OUTPUT_DIR,
            f"3-{i}_attraction-rst_fixation-count-after-dom-known-{measure}_bayescorr_trace.png",
        ),
        dpi=100,
    )
    plt.close()

    # Print statistics
    print(
        f"  mean r = {summary_df.loc['r', 'mean']} [{summary_df.loc['r', 'hdi_2.5%']}, {summary_df.loc['r', 'hdi_97.5%']}]"
    )

# %% Test 4: Faster RTs for target vs non-target choices in attraction trials
# ---------------------------------------------------------------------------
print(
    "Test 4: Shorter RTs for target vs non-target choices in attraction trials (BEST analysis)"
)

# Make a DataFrame with subject mean RTs of attraction trials with target and other choices
trials["target_chosen"] = trials["choice_tcd"] == "target"
df = (
    trials.loc[trials["effect"] == "attraction"]
    .groupby(["subject", "target_chosen"])["rt"]
    .mean()
    .rename("rt")
    .reset_index()
    .pivot_table(index="subject", columns="target_chosen", values="rt")
)
df.columns.name = None
df.columns = ["target_chosen", "other_chosen"]
df["difference"] = (df["target_chosen"] - df["other_chosen"]) / 1000

# Run BEST analysis
trace = runBEST1G(
    y=df["difference"],
    mu=0,
    sigma_low=0.1,
    sigma_high=100.0,
    sample_kwargs=sample_kwargs,
)

summary_df = summary(trace, hdi_prob=0.95)
summary_df.to_csv(
    join(
        OUTPUT_DIR,
        "4_attraction-rt-targetchoice_attraction-rt-otherchoice_best_summary.csv",
    )
)

plot_trace(trace)
plt.savefig(
    join(
        OUTPUT_DIR,
        "4_attraction-rt-targetchoice_attraction-rt-otherchoice_best_bayescorr_trace.png",
    ),
    dpi=100,
)
plt.close()

# Print statistics
print(
    f"  mean difference = {summary_df.loc['difference', 'mean']} [{summary_df.loc['difference', 'hdi_2.5%']}, {summary_df.loc['difference', 'hdi_97.5%']}]\n"
    + f"  mean d = {summary_df.loc['d', 'mean']} [{summary_df.loc['d', 'hdi_2.5%']}, {summary_df.loc['d', 'hdi_97.5%']}]\n"
)
