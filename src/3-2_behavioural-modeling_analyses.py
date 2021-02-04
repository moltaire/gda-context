#!/usr/bin/python
"""
Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour

This script performs model comparison of previously fitted models
    1. Compute mean ± s.d. BICs per model
    2. Count individually best fitting models
    3. Bayesian correlation analysis: observed vs. predicted context effects for GLA
    4. Bayesian linear regression analysis: observed vs. predicted context effects for GLA
    5. Bias analysis: BEST analysis of difference between predicted vs. observed RST
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

warnings.filterwarnings("ignore")

logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)

# Random seed
SEED = 32
np.random.seed(SEED)

# MCMC settings (passed on to pm.sample)
sample_kwargs = {"cores": 1, "random_seed": SEED, "progressbar": False}

# Directories
RESULTS_DIR = join("..", "results")
OUTPUT_DIR = join("..", "results", "3-behavioural-modeling")

# Load estimates and BICs
estimates = pd.read_csv(
    join(RESULTS_DIR, "3-behavioural-modeling", "estimates", "estimates_de1.csv")
)

# %% -1. Summarize GLA estimates
# ------------------------------
gla_estimates = estimates.loc[
    estimates["model"] == "glickman1layer",
    ["subject", "bic", "nll", "alpha", "beta", "gamma", "lam", "theta"],
].reset_index(drop=True)

gla_estimates_summary = (
    gla_estimates[["alpha", "beta", "gamma", "lam", "theta"]].describe().round(2).T
)
output_file = join(OUTPUT_DIR, "gla_estimates_summary.csv")
gla_estimates_summary.to_csv(output_file)
print(f"Saved summary of GLA estimates to '{output_file}'.")

# %% 0. Save BIC dataframe for BMS
# --------------------------------
output_file = join(OUTPUT_DIR, "model-comparison_bics.csv")
(
    estimates[["subject", "model", "bic"]]
    .pivot(index="subject", columns="model", values="bic")
    .reset_index()
    .round(4)
    .to_csv(
        output_file,
        index=False,
    )
)
print("\tGenerated output file:", output_file)


# %% 1. Compute mean ± s.d. BICs per model
# -------------------------------------
print("1. Summarising mean ± s.d. BICs per model...")
bic_summary = (
    estimates.groupby("model")["bic"].describe().sort_values("mean").reset_index()
)
output_file = join(OUTPUT_DIR, "model-comparison_bic_summary.csv")
bic_summary.round(4).to_csv(output_file, index=False)
print("\tGenerated output file:", output_file)


# %% 2) Count individually best fitting models
# -----------------------------------------

bics = estimates.pivot_table(values="bic", columns="model", index="subject")
best_model = bics.idxmin(axis=1)
best_model.name = "model"
output_file = join(OUTPUT_DIR, "model-comparison_individual-best-models.csv")
best_model.reset_index().round(4).to_csv(output_file, index=False)
print(
    "\tGenerated output file:",
    output_file,
)

print("2. Individually best fitting (lowest BIC) model counts:")
individual_best_model_counts = best_model.value_counts().rename("N")
print(individual_best_model_counts)
output_file = join(OUTPUT_DIR, "model-comparison_individual-best-models_count.csv")
individual_best_model_counts.to_csv(output_file)
print(
    "\tGenerated output file:",
    output_file,
)

# %% 3) Bayesian correlation analysis: observed vs. predicted context effects for GLA
# -------------------------------------------------------------------------------------------


def calc_rst(df, effect="attraction"):
    """Calculate RST.

    Parameters
    ----------
    df : pandas.DataFrame
        trial DataFrame, containing columns `subject`, `effect`, `choice_tcd`
    effect : str, one of ['attraction', 'compromise'], optional
        which effect to look at, by default 'attraction'

    Returns
    -------
    pandas.Series
        subject-wise RSTs
    """
    cs = (
        df.loc[df["effect"] == effect]
        .groupby("subject")["choice_tcd"]
        .value_counts()
        .rename("frequency")
        .reset_index()
        .pivot_table(index="subject", values="frequency", columns="choice_tcd")
        .fillna(0)
    )
    cs["rst"] = cs["target"] / (cs["target"] + cs["competitor"])
    return cs["rst"]


def calc_ptpc(df, effect="attraction"):
    """Calculate P(Target)-P(Competitor)

    Parameters
    ----------
    df : pandas.DataFrame
        trial DataFrame, containing columns `subject`, `effect`, `choice_tcd`
    effect : str, one of ['attraction', 'compromise'], optional
        which effect to look at, by default 'attraction'

    Returns
    -------
    pandas.Series
        subject-wise P(Target) - P(Competitor)
    """
    cs = (
        df.loc[df["effect"] == effect]
        .groupby("subject")["choice_tcd"]
        .value_counts(normalize=True)
        .rename("frequency")
        .reset_index()
        .pivot_table(index="subject", values="frequency", columns="choice_tcd")
        .fillna(0)
    )
    cs["ptpc"] = cs["target"] - cs["competitor"]
    return cs["ptpc"]


# Load observed behaviour
observed = pd.read_csv(join(RESULTS_DIR, "0-clean_data", "trials.csv"))

# Load prediction dataframe (includes predictions of all models)
predicted = pd.read_csv(
    join(RESULTS_DIR, "3-behavioural-modeling", "predictions", "predictions_de1.csv"),
    index_col=[0, 1],
).reset_index(drop=True)
## Add `choice_tcd` variable to predicted data
predicted["choice_tcd"] = np.where(
    pd.isnull(predicted["target"]),
    np.nan,
    np.where(
        predicted["predicted_choice"] == 2,
        "decoy",
        np.where(
            ((predicted["predicted_choice"] == 0) & (predicted["target"] == "A"))
            | ((predicted["predicted_choice"] == 1) & (predicted["target"] == "B")),
            "target",
            "competitor",
        ),
    ),
)

print("3. Running Bayesian Correlation analysis of observed vs. GLA-predicted RST")

for e, effect in enumerate(["attraction", "compromise"]):
    rst_obs = calc_rst(observed, effect=effect)
    pred_gla = predicted.loc[predicted["model"] == "glickman1layer"]
    rst_pred = calc_rst(pred_gla, effect=effect)
    assert np.alltrue(rst_obs.index == rst_pred.index), "Mismatching indices!"

    trace = bayescorr.runBayesCorr(rst_obs, rst_pred, sample_kwargs=sample_kwargs)

    # Summary
    summary = pm.summary(trace, hdi_prob=0.95)
    summary.loc["r", "p>0"] = np.mean(trace.get_values("r") > 0)
    summary.to_csv(
        join(
            OUTPUT_DIR,
            f"rst_observed_gla-predicted_{effect}_correlation_summary.csv",
        )
    )

    # Traceplot
    pm.traceplot(trace)
    plt.savefig(
        join(
            OUTPUT_DIR,
            f"rst_observed_gla-predicted_{effect}_correlation_trace.png",
        ),
        dpi=100,
    )

    # Posterior plot
    pm.plot_posterior(
        trace,
        var_names="r",
        hdi_prob=0.95,
        round_to=2,
        ref_val=0.0,
        figsize=(2.5, 2.5),
    )
    plt.savefig(
        join(
            OUTPUT_DIR, f"rst_observed_gla-predicted_{effect}_correlation_posterior.png"
        ),
        dpi=100,
    )

    # Print results
    print(
        "GLA predicted\t{effect}\tr = {r:.2f} [{lower:.2f}, {upper:.2f}], P(r>0) = {plarger0:.2f}".format(
            effect=effect,
            r=summary.loc["r", "mean"],
            lower=summary.loc["r", "hdi_2.5%"],
            upper=summary.loc["r", "hdi_97.5%"],
            plarger0=summary.loc["r", "p>0"],
        )
    )


# %% 4) Bayesian linear regression analysis: observed vs. predicted context effects for GLA
# -------------------------------------------------------------------------------------------------
def runBayesReg(x, y, sample_kwargs={}):
    """Run Bayesian Gaussian regression with PyMC3 default priors.

    Args:
        x (array-like): Predictor variable
        y (array-like): Outcome variable

    Returns:
        pymc3.trace: Posterior trace
    """
    data = pd.DataFrame(dict(x=x, y=y))
    with pm.Model():
        pm.glm.GLM.from_formula("y ~ x", data=data)
        trace = pm.sample(**sample_kwargs)
    return trace


print("4. Running Bayesian Regression analysis of observed vs. GLA-predicted RST")
for e, effect in enumerate(["attraction", "compromise"]):
    rst_obs = calc_rst(observed, effect=effect)
    pred_gla = predicted.loc[predicted["model"] == "glickman1layer"]
    rst_pred = calc_rst(pred_gla, effect=effect)
    assert np.alltrue(rst_obs.index == rst_pred.index), "Mismatching indices!"

    trace = runBayesReg(x=rst_obs, y=rst_pred, sample_kwargs=sample_kwargs)

    # Summary
    summary = pm.summary(trace, hdi_prob=0.95)
    variables = ["Intercept", "x", "sd"]
    for var in variables:
        summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
    summary.to_csv(join(OUTPUT_DIR, f"rst_observed_gla-predicted_{effect}_summary.csv"))

    # Traceplot
    pm.traceplot(trace)
    plt.savefig(
        join(OUTPUT_DIR, f"rst_observed_gla-predicted_{effect}_regression_trace.png"),
        dpi=100,
    )

    # Posterior plot
    pm.plot_posterior(
        trace,
        var_names=variables,
        hdi_prob=0.95,
        round_to=2,
        ref_val=0.0,
        figsize=(len(variables) * 2.5, 2.5),
    )
    plt.savefig(
        join(OUTPUT_DIR, f"rst_observed_gla-predicted_{effect}_posterior.png"),
        dpi=100,
    )

    # Print results
    print(
        "GLA predicted:\t{effect}\tIntercept = {intercept:.2f} [{intercept_lower:.2f}, {intercept_upper:.2f}], beta = {beta:.2f} [{beta_lower:.2f}, {beta_upper:.2f}]".format(
            effect=effect,
            intercept=summary.loc["Intercept", "mean"],
            intercept_lower=summary.loc["Intercept", "hdi_2.5%"],
            intercept_upper=summary.loc["Intercept", "hdi_97.5%"],
            beta=summary.loc["x", "mean"],
            beta_lower=summary.loc["x", "hdi_2.5%"],
            beta_upper=summary.loc["x", "hdi_97.5%"],
        )
    )


# %% 5) Bias analysis: BEST of difference predicted-observed vs 0
print(
    "5. Running bias analysis: BEST of difference (RST_GLA-predicted - RST_observed) vs 0"
)
for e, effect in enumerate(["attraction", "compromise"]):
    rst_obs = calc_rst(observed, effect=effect)
    pred_m = predicted.loc[predicted["model"] == "glickman1layer"]
    rst_pred = calc_rst(pred_m, effect=effect)
    assert np.alltrue(rst_obs.index == rst_pred.index), "Mismatching indices!"

    trace = best.runBEST1G(
        y=(rst_pred - rst_obs),
        mu=0,
        sigma_low=0.0001,
        sigma_high=10.0,
        sample_kwargs=sample_kwargs,
    )

    # Summary
    summary = pm.summary(trace, hdi_prob=0.95)
    variables = ["difference", "d"]
    for var in variables:
        summary.loc[var, "p>0"] = np.mean(trace.get_values(var) > 0)
    summary.to_csv(
        join(OUTPUT_DIR, f"rst_observed_rst-predicted_{effect}_BEST_summary.csv")
    )

    # Traceplot
    pm.traceplot(trace)
    plt.savefig(
        join(OUTPUT_DIR, f"rst_observed_rst-predicted_{effect}_BEST_trace.png"),
        dpi=100,
    )

    # Posterior plot
    pm.plot_posterior(
        trace,
        var_names=variables,
        hdi_prob=0.95,
        round_to=2,
        ref_val=0.0,
        figsize=(len(variables) * 2.5, 2.5),
    )
    plt.savefig(
        join(OUTPUT_DIR, f"rst_observed_rst-predicted_{effect}_BEST_posterior.png"),
        dpi=100,
    )

    # Print results
    print(
        "GLA predicted\t{effect}\tmean difference = {mean:.2f} [{mean_lower:.2f}, {mean_upper:.2f}], d = {d:.2f} [{d_lower:.2f}, {d_upper:.2f}]".format(
            effect=effect,
            mean=summary.loc["mean", "mean"],
            mean_lower=summary.loc["mean", "hdi_2.5%"],
            mean_upper=summary.loc["mean", "hdi_97.5%"],
            d=summary.loc["d", "mean"],
            d_lower=summary.loc["d", "hdi_2.5%"],
            d_upper=summary.loc["d", "hdi_97.5%"],
        )
    )
