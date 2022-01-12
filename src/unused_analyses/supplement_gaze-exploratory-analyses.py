#!/usr/bin/python
"""
Eye movements in context effects
This script performs additional exploratory statistical analyses of the gaze data
    1. Pairwise (paired) BEST comparisons of relative dwell times towards options A, B, C and attributes
    2. Pairwise BEST analyses of transition types (vertical, horizontal, diagonal) between effects
    3. One-sample (paired) BEST of Payne Index in Attraction vs Compromise trials
    4. Bayesian regression analysis of RST onto dwell to decoy with interaction between effects
    5. Bayesian regression analysis of P(choose decoy) onto dwell to decoy with interaction between effects
    6. Compute subject level mean ± s.d. and grand average of transitions for attraction and compromise trials
Author: Felix Molter, felixmolter@gmail.com
"""

import logging
import warnings
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from analysis import best
from analysis.utilities import makeDirIfNeeded

warnings.filterwarnings("ignore")


logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)


# Set seed
SEED = 41
np.random.seed(SEED)

# Sampling settings
sample_kwargs = dict(cores=1)

# Set data and output directories
RESULTS_DIR = join("..", "results")
OUTPUT_DIR = join(RESULTS_DIR, "S_supplemental-gaze-analyses")

# 1. Pairwise (paired) BEST comparisons of relative dwell times towards options A, B, C and attributes
# ----------------------------------------------------------------------------------------------------
label = "rel-dwell_pairwise-BEST_alternatives-attributes"
ANALYSIS_DIR = join(OUTPUT_DIR, label)
makeDirIfNeeded(ANALYSIS_DIR)

# Load relative dwells
dwells = pd.read_csv(join(RESULTS_DIR, "2-gaze", "dwells_across-targets.csv"))

print(
    "\tRunning pairwise comparisons of relative dwells towards alternatives and attributes..."
)

# Run these analyses separately for attraction and compromise trials
for effect in ["attraction", "compromise"]:
    print("\t{}".format(effect.capitalize()))

    dwells_e = dwells.loc[dwells["effect"] == effect]

    for pair in [("A", "B"), ("A", "C"), ("B", "C"), ("p", "m")]:

        print("\t\t{} vs {}".format(*pair))

        # Run BEST
        trace = best.runBEST1G(
            dwells_e["dwell_{}".format(pair[0])] - dwells_e["dwell_{}".format(pair[1])],
            mu=0.0,
            sigma_low=0.001,
            sigma_high=10.0,
            seed=SEED,
            sample_kwargs=sample_kwargs,
        )

        print(
            "{}% of posterior mass above 0.".format(
                100 * np.mean(trace.get_values("d") > 0)
            )
        )

        # Save traceplot
        pm.traceplot(trace)
        plt.savefig(
            join(
                ANALYSIS_DIR,
                f"{label}_{effect}_{pair[0].lower()}-v-{pair[1].lower()}_trace.png",
            ),
            dpi=100,
        )

        # Save posterior plot
        pm.plot_posterior(
            trace,
            var_names=["difference", "d"],
            credible_interval=0.95,
            round_to=2,
            ref_val=0.0,
            figsize=(10, 2.5),
        )
        plt.savefig(
            join(
                ANALYSIS_DIR,
                f"{label}_{effect}_{pair[0].lower()}-v-{pair[1].lower()}_posterior.png",
            ),
            dpi=100,
        )

        # Save summary
        summary = pm.summary(trace)
        for var in ["mean", "difference", "d"]:
            summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
        summary.to_csv(
            join(
                ANALYSIS_DIR,
                f"{label}_{effect}_{pair[0].lower()}-v-{pair[1].lower()}_summary.csv".format(
                    effect, pair[0].lower(), pair[1].lower()
                ),
            )
        )


# 2. One-sample (paired) BEST of Payne Index in Attraction vs Compromise trials
# -----------------------------------------------------------------------------
label = "payne-index_BEST_attraction-vs-compromise"
ANALYSIS_DIR = join(OUTPUT_DIR, label)
makeDirIfNeeded(ANALYSIS_DIR)

# Load Payne Index data
pi = pd.read_csv(join(RESULTS_DIR, "2-gaze", "payne-index_across-targets.csv"))
pi_effect = pi.pivot(index="subject", columns="effect", values="payne_index")

print("\tRunning Paired Sample BESTs: Payne Index Attraction vs. Compromise")

# Run BEST
trace = best.runBEST1G(
    pi_effect["attraction"] - pi_effect["compromise"],
    mu=0.0,
    sigma_low=0.001,
    sigma_high=10.0,
    seed=SEED,
    sample_kwargs=sample_kwargs,
)
print("{}% of posterior mass above 0.".format(100 * np.mean(trace.get_values("d") > 0)))

# Save traceplot
pm.traceplot(trace)
plt.savefig(join(ANALYSIS_DIR, f"{label}_trace.png"), dpi=100)

# Save posterior plot
pm.plot_posterior(
    trace,
    var_names=["difference", "d"],
    credible_interval=0.95,
    round_to=2,
    ref_val=0.0,
    figsize=(10, 2.5),
)
plt.savefig(join(ANALYSIS_DIR, f"{label}_posterior.png"), dpi=100)

# Save summary
summary = pm.summary(trace)
for var in ["mean", "difference", "d"]:
    summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
summary.to_csv(join(ANALYSIS_DIR, f"{label}_summary.csv"))

# 3. Bayesian regression analysis of RST onto dwell to decoy with interaction between effects
# -------------------------------------------------------------------------------------------
label = "rel-dwell_pairwise-BEST_alternatives-attributes"
ANALYSIS_DIR = join(OUTPUT_DIR, label)
makeDirIfNeeded(ANALYSIS_DIR)

print(
    "Running Bayesian Regression of RST onto dwell-to-decoy with interaction between effects..."
)

# Load trial-wise dwell data
trials = pd.read_csv(join(RESULTS_DIR, "0-clean_data", "trials_with-dwells.csv"))

# Compute mean dwell to decoy per subject and effect
# /!\ excluding trials where the decoy was chosen
dwells = (
    trials.loc[trials["effect"].isin(["attraction", "compromise"])]
    .loc[trials["choice_tcd"] != "decoy"][
        ["subject", "effect", "dwell_target", "dwell_decoy", "dwell_competitor"]
    ]
    .groupby(["subject", "effect"])
    .mean()
    .reset_index()
)

# Load choiceshare data
cs = pd.read_csv(join(RESULTS_DIR, "1-behaviour", "choiceshares_across-targets.csv"))

# Construct joint dataframe
df = dwells[["subject", "effect", "dwell_decoy"]].merge(
    cs[["subject", "effect", "rst"]], on=["subject", "effect"]
)

# z-score variables
df["rst_z"] = (df["rst"] - df["rst"].mean()) / df["rst"].std(ddof=0)
df["dwell_decoy_z"] = (df["dwell_decoy"] - df["dwell_decoy"].mean()) / df[
    "dwell_decoy"
].std(ddof=0)

# create dummy variable for attraction trials
df["is_attraction"] = (df["effect"] == "attraction").astype(int)

# Save this dataframe
df.to_csv(
    join(OUTPUT_DIR, "bayesReg_rst-by-decoy-dwell-x-effect_data.csv"), index=False
)

with pm.Model():
    pm.GLM.from_formula(
        formula="rst_z ~ dwell_decoy_z + is_attraction + dwell_decoy_z:is_attraction",
        data=df,
    )

    trace = pm.sample(**sample_kwargs)
    pm.trace_to_dataframe(trace).to_csv(
        join(OUTPUT_DIR, "bayesReg_rst-by-decoy-dwell-x-effect_trace.csv"), index=False
    )
    pm.traceplot(trace)
    plt.savefig(
        join(OUTPUT_DIR, "bayesReg_rst-by-decoy-dwell-x-effect_trace.png"), dpi=100
    )

    # Compute combined effect: dwell_decoy_z in attraction trials:
    dwell_decoy_attr = trace["dwell_decoy_z"] + trace["dwell_decoy_z:is_attraction"]
    trace.add_values(dict(dwell_decoy_attr=dwell_decoy_attr))

    variables = [
        "Intercept",
        "dwell_decoy_z",
        "is_attraction",
        "dwell_decoy_z:is_attraction",
        "dwell_decoy_attr",
    ]

    pm.plot_posterior(
        trace,
        var_names=variables,
        credible_interval=0.95,
        round_to=2,
        ref_val=0.0,
        figsize=(len(variables) * 2.5, 2.5),
    )
    plt.savefig(
        join(OUTPUT_DIR, "bayesReg_rst-by-decoy-dwell-x-effect_posterior.png"), dpi=100
    )

    summary = pm.summary(trace)
    for var in variables:
        summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
    summary.to_csv(join(OUTPUT_DIR, "bayesReg_rst-by-decoy-dwell-x-effect_summary.csv"))


# 3. Bayesian regression analysis of P(choose decoy) onto dwell to decoy with interaction between effects
# -------------------------------------------------------------------------------------------

print(
    "Running Bayesian regression of P(choose decoy) onto dwell-to-decoy with interaction between effects"
)

# Load dwell data
dwells = pd.read_csv(join(RESULTS_DIR, "2-gaze", "dwells_across-targets.csv"))

# Load choiceshare data
cs = pd.read_csv(join(RESULTS_DIR, "1-behaviour", "choiceshares_across-targets.csv"))

# Construct joint dataframe
df = (
    dwells[["subject", "effect", "dwell_decoy"]]
    .merge(cs[["subject", "effect", "decoy"]], on=["subject", "effect"])
    .rename({"decoy": "cs_decoy"}, axis=1)
)

# z-score variables
df["cs_decoy_z"] = (df["cs_decoy"] - df["cs_decoy"].mean()) / df["cs_decoy"].std(ddof=0)
df["dwell_decoy_z"] = (df["dwell_decoy"] - df["dwell_decoy"].mean()) / df[
    "dwell_decoy"
].std(ddof=0)

# create dummy variable for attraction trials
df["is_attraction"] = (df["effect"] == "attraction").astype(int)

# Save this dataframe
df.to_csv(
    join(OUTPUT_DIR, "bayesReg_cs-decoy-by-decoy-dwell-x-effect_data.csv"), index=False
)

with pm.Model():
    pm.GLM.from_formula(
        formula="cs_decoy_z ~ dwell_decoy_z + is_attraction + dwell_decoy_z:is_attraction",
        data=df,
    )

    trace = pm.sample(**sample_kwargs)

    pm.traceplot(trace)
    plt.savefig(
        join(OUTPUT_DIR, "bayesReg_cs-decoy-by-decoy-dwell-x-effect_trace.png"), dpi=100
    )

    # Compute combined effect: dwell_decoy_z in attraction trials:
    dwell_decoy_attr = trace["dwell_decoy_z"] + trace["dwell_decoy_z:is_attraction"]
    trace.add_values(dict(dwell_decoy_attr=dwell_decoy_attr))

    variables = [
        "Intercept",
        "dwell_decoy_z",
        "is_attraction",
        "dwell_decoy_z:is_attraction",
        "dwell_decoy_attr",
    ]

    pm.plot_posterior(
        trace,
        var_names=variables,
        credible_interval=0.95,
        round_to=2,
        ref_val=0.0,
        figsize=(len(variables) * 2.5, 2.5),
    )
    plt.savefig(
        join(OUTPUT_DIR, "bayesReg_cs-decoy-by-decoy-dwell-x-effect_posterior.png"),
        dpi=100,
    )

    summary = pm.summary(trace)
    for var in variables:
        summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
    summary.to_csv(
        join(OUTPUT_DIR, "bayesReg_cs-decoy-by-decoy-dwell-x-effect_summary.csv")
    )

# Transitions
# 4. Compute subject level mean ± s.d. and grand average of transitions for attraction and compromise trials
# ----------------------------------------------------------------------------------------------------------

print("Computing subject level and grand average transition data.")

# load trial-wise transition data
transitions = pd.read_csv(join(RESULTS_DIR, "0-clean_data", "transitions.csv"))
# add effect column from trial dataframe
transitions = transitions.merge(
    trials[["subject", "trial", "effect"]], on=["subject", "trial"]
)

# Subject level
trans_sub = (
    transitions.loc[
        transitions["effect"].isin(["attraction", "compromise"]),
        [
            "subject",
            "effect",
            "n_vertical",
            "n_horizontal",
            "n_diagonal",
            "n_between-alt-diff",
            "n_between-alt-same",
            "n_within-alt",
            "n_dist0",
            "n_dist1",
            "n_dist2",
        ],
    ]
    .groupby(["subject", "effect"])
    .mean()
    .reset_index()
)

trans_sub.to_csv(join(OUTPUT_DIR, "transitions_across-targets.csv"), index=False)

# Grand average
trans_sub_summary = (
    trans_sub.groupby("effect")[
        [
            "n_vertical",
            "n_horizontal",
            "n_diagonal",
            "n_between-alt-diff",
            "n_between-alt-same",
            "n_within-alt",
            "n_dist0",
            "n_dist1",
            "n_dist2",
        ]
    ].agg(["mean", "std", "min", "max"])
).T

trans_sub_summary.to_csv(join(OUTPUT_DIR, "transitions_across-targets_summary.csv"))

# 5. Pairwise BEST analyses of transition types (vertical, horizontal, diagonal) between effects
for ttype in ["n_vertical", "n_horizontal", "n_diagonal"]:

    print(
        "\tRunning Paired Sample BESTs: {ttype} Attraction vs. Compromise".format(
            ttype=ttype
        )
    )
    df = trans_sub.pivot(values=ttype, columns="effect", index="subject")
    diff = df["attraction"] - df["compromise"]

    trace = best.runBEST1G(
        diff,
        mu=0.0,
        sigma_low=0.001,
        sigma_high=10.0,
        seed=SEED,
        sample_kwargs=sample_kwargs,
    )
    print(
        "{}% of posterior mass above 0.".format(
            100 * np.mean(trace.get_values("d") > 0)
        )
    )
    pm.traceplot(trace)
    plt.savefig(
        join(
            OUTPUT_DIR,
            "BEST1G_trans_{ttype}_attr-v-comp_trace.png".format(ttype=ttype[2:]),
        ),
        dpi=100,
    )
    pm.plot_posterior(
        trace,
        var_names=["difference", "d"],
        credible_interval=0.95,
        round_to=2,
        ref_val=0.0,
        figsize=(10, 2.5),
    )
    plt.savefig(
        join(
            OUTPUT_DIR,
            "BEST1G_trans_{ttype}_attr-v-comp_posterior.png".format(ttype=ttype[2:]),
        ),
        dpi=100,
    )
    summary = pm.summary(trace)
    for var in ["mean", "difference", "d"]:
        summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
    summary.to_csv(
        join(
            OUTPUT_DIR,
            "BEST1G_trans_{ttype}_attr-v-comp_summary.csv".format(ttype=ttype[2:]),
        )
    )
