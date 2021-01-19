#!/usr/bin/python
"""
Gaze-dependent accumulation in context-dependent risky choice
This script makes figures of the behavioural-modeling analyses.
    1) Composite figure of
        a) BIC violin plot
        b) Barplot of individually best fitting model counts
        c, d) Observed vs. predicted RST for the winning model
        e-f) gaze-advantage choice probability plots for all models
    2) Supplementary Figure of RST vs relative model fit (GLA vs MDFT)
"""

import logging
import warnings
from os.path import join
from string import ascii_lowercase

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from plotting.plot_share import (
    lm,
    violin,
    plot_dwell_adv_set,
    plot_observed_predicted_rst,
)
from plotting.plot_utils import *

matplotlib = set_mpl_defaults(matplotlib)

warnings.filterwarnings("ignore")

logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)

# Set data and output directories
RESULTS_DIR = join("..", "results")
OUTPUT_DIR = join("..", "figures")

# Random seed
SEED = 3
np.random.seed(SEED)

# Colours to use
palette = [
    "slategray",
    "darksalmon",
    "mediumaquamarine",
    "indianred",
    "paleturquoise",
    "lightpink",
    "tan",
    "orchid",
]

# Model labels to use in the figures
models = np.array(
    ["glickman1layer", "mdft", "eu", "gaze-baseline-stat", "gaze-baseline-dyn"]
)

model_labels = {
    "glickman1layer": "GLA",
    "mdft": "MDFT",
    "eu": "EU",
    "gaze-baseline-stat": "GB$_{stat}$",
    "gaze-baseline-dyn": "GB$_{dyn}$",
}

# MCMC settings passed on to pm.sample
sample_kwargs = {"cores": 1, "random_seed": SEED, "progressbar": False}

# Size of a single plot panel, in cm
panel_width = 4.5
panel_height = 4.5

# %% Load and preprocess data
# Load trial data
trials = pd.read_csv(join(RESULTS_DIR, "0-clean_data", "trials_with-dwells.csv"))

# Read model predictions, convert choice column to alphanumeric, compute `choice_tcd` variable
predictions = pd.read_csv(
    join(RESULTS_DIR, "3-behavioural-modeling", "predictions", "predictions_de1.csv")
)
predictions["predicted_choice_abc"] = np.array(["A", "B", "C"])[
    predictions["predicted_choice"]
]
predictions["predicted_choice_tcd"] = np.where(
    predictions["predicted_choice_abc"] == "C",
    "decoy",
    np.where(
        predictions["predicted_choice_abc"] == predictions["target"],
        "target",
        "competitor",
    ),
)
predictions.loc[pd.isnull(predictions["target"]), "predicted_choice_tcd"] = np.nan

# add dwell time advantage for each alternative
bins = np.arange(-0.35, 0.36, 0.1).round(2)

for i in ["A", "B", "C", "target", "competitor", "decoy"]:
    others = [j for j in ["A", "B", "C"] if j != i]
    for df in [trials, predictions]:
        adv_i = df[f"dwell_{i}"] - df[[f"dwell_{j}" for j in others]].mean(axis=1)
        df[f"total_dwell_adv_{i}"] = pd.cut(adv_i, bins=bins)

# Load parameter estimate with BIC values
estimates = pd.read_csv(
    join(RESULTS_DIR, "3-behavioural-modeling", "estimates", "estimates_de1.csv")
)[["subject", "model", "bic"]]

# Load BMS results from MATLAB
bms = pd.read_csv(
    join(RESULTS_DIR, "3-behavioural-modeling", "model-comparison_bms_results.csv")
)

# Load individual best fitting model list
individual_best_models = pd.read_csv(
    join(
        RESULTS_DIR,
        "3-behavioural-modeling",
        "model-comparison_individual-best-models.csv",
    )
)

# Read subject-level choice share data
cs = pd.read_csv(join(RESULTS_DIR, "1-behaviour", "choiceshares_across-targets.csv"))


# %% Plotting functions
# -------------------------
def plot_bic_violins(estimates, models, model_labels, ax=None):
    """
    Violin plots of BIC distributions for each model.
    """
    if ax is None:
        ax = plt.gca()
    # Determine the model order, so that the best model is first and so on
    model_order = (
        estimates.groupby("model")["bic"]
        .mean()
        .reset_index()
        .sort_values("bic")["model"]
        .values
    )

    # pivot: each line should be a subject, each column contains BIC of a single model
    bics_wide = estimates.pivot(values="bic", columns="model", index="subject")[
        [model for model in model_order if model in models]
    ]
    bics_wide.columns.name = None

    # Make the figure
    ax = axs[0, 0]
    ax = violin(
        data=bics_wide,
        ax=ax,
        palette=[
            palette[np.where(models == model)[0][0]]
            for model in model_order  # ensure consistent colors for models
        ]
        * len(bics_wide.columns),
    )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_labels[c] for c in bics_wide.columns], rotation=45)
    ax.set_ylabel("BIC")
    ax.set_xlabel(None)
    ax.set_ylim(0, 600)

    # add dashed line for random model
    ax.axhline(
        -2 * 225 * np.log(1 / 3),
        linestyle="--",
        color="black",
        zorder=-1,
        linewidth=0.75,
    )
    return ax


def plot_n_best_fitting_pxp(individual_best_models, bms, models, model_labels, ax=None):
    """
    Makes a barplot of best fitting models in `individual_best_models` with inset showing protected exceedance probabillities from `bms`.
    """
    if ax is None:
        ax = plt.gca()

    models_bms = bms["model"].values
    pxp = bms["pxp"].values

    individual_best_models["model"] = pd.Categorical(
        individual_best_models["model"].str.replace("_", "-"), categories=models
    )
    N = individual_best_models.astype({"model": "category"})["model"].value_counts()

    # Make the figure
    ax = axs[0, 1]
    ax.bar(
        np.arange(len(N)),
        N,
        color=[palette[np.where(models == model)[0][0]] for model in N.index],
    )
    ax.set_xticks(np.arange(len(N)))
    ax.set_xticklabels([model_labels[i] for i in N.index], rotation=45)
    ax.set_ylabel("N Lowest BIC")
    ax.set_yticks(np.arange(0, 41, 10))
    ax.set_ylim(0, 40)

    # Annotate counts
    for i, n in enumerate(N):
        ax.annotate(str(n), (i, n + 1), ha="center", fontsize=5)

    # Make inset for exceedance probabilities
    axins = ax.inset_axes(bounds=(0.6, 0.7, 0.4, 0.3))
    axins.bar(np.arange(len(models)), pxp[np.argsort(pxp)[::-1]], color="#666666")
    axins.set_xticks(np.arange(len(models)))
    axins.set_xticklabels(
        [
            model_labels[model.replace("_", "-")]
            for model in models_bms[np.argsort(pxp)[::-1]]
        ],
        fontsize=4,
        rotation=90,
    )
    axins.set_ylabel("$pxp$", fontsize=4, labelpad=-5)
    axins.set_yticks([0, 1])
    axins.set_ylim(0, 1)
    return ax


# %% 1) Make composite figure
# Set up figure
fig, axs = plt.subplots(
    4,
    4,
    gridspec_kw={"height_ratios": [3, 0.7, 2, 2]},
    figsize=cm2inch(4 * panel_width, 5 / 3 * panel_height),
)

# a) BIC Violin
axs[0, 0] = plot_bic_violins(estimates, models, model_labels, ax=axs[0, 0])

# b) Best fitting barplots with pXP inset
axs[0, 1] = plot_n_best_fitting_pxp(
    individual_best_models, bms, models, model_labels, ax=axs[0, 1]
)

# c-d) observed vs GLA predicted RST
for ax, effect in zip(axs[0, 2:4], ["attraction", "compromise"]):

    ax = plot_observed_predicted_rst(
        cs,
        predictions.loc[predictions["model"] == "glickman1layer"],
        effect=effect,
        scatter_color=palette[0],
        line_color=palette[0],
        ax=ax,
        sample_kwargs=sample_kwargs,
    )
    ax.set_ylabel("GLA pred. RST")

# Include a row of empty axes for better spacing
for ax in axs[1, :]:
    ax.axis("off")

# e-j: Gaze advantage plots for targets, competitors, decoys in attraction, compromise trials
axs[2:, :-1] = plot_dwell_adv_set(
    trials, predictions, models, model_labels, palette=palette, axs=axs[2:, :-1]
)
axs[2, -1].axis("off")
axs[3, -1].axis("off")

plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0.5, wspace=0.6)

# Label panels
all_axes = (
    list(axs[0, :].ravel()) + list(axs[2, :-1].ravel()) + list(axs[3, :-1].ravel())
)
for ax, label in zip(all_axes, list(ascii_lowercase)):
    # Place axis labels in figure space, so that they are aligned
    # https://stackoverflow.com/a/52309638
    if label == "a":
        xshift = 0.045
    elif label == "b":
        xshift = 0.039
    else:
        xshift = 0.05
    X = ax.get_position().x0
    Y = ax.get_position().y1
    fig.text(X - xshift, Y, label, size=10, weight="bold", ha="right", va="center")

# Save the figure
plt.savefig(join(OUTPUT_DIR, "3-model-comparison.pdf"), bbox_inches="tight")

# %% 2) Supplementary Figure: deltaBIC vs RST
# -------------------------------------------
# Preprocess data (already loaded)
# BIC
estimates_long = (
    estimates.loc[estimates["model"].isin(["glickman1layer", "mdft"])]
    .pivot(index="subject", columns="model", values="bic")
    .reset_index()
)
estimates_long.columns.name = None

# Observed RST
rst_long = cs.pivot(index="subject", columns="effect", values="rst")

# Combine
df = estimates_long.merge(rst_long, on="subject")
df["dBIC_scaled"] = (df["glickman1layer"] - df["mdft"]) / 100

# Make the figure
fig, axs = plt.subplots(1, 2, figsize=cm2inch(9, 4.5), dpi=300)

for effect, ax in zip(["attraction", "compromise"], axs):

    ax.set_title(effect.capitalize())

    ax.set_xlabel(r"$(BIC_{GLA} - BIC_{MDFT})$ / 100")
    ax.set_ylabel("RST")
    ax.axvline(0, color="gray", linewidth=0.5, zorder=-1)
    ax, trace, summary = lm(
        df["dBIC_scaled"],
        df[effect],
        ax=ax,
        scatter_color="slategray",
        line_color="slategray",
        sample_kwargs=sample_kwargs,
    )

    summary.loc["x", "P>0"] = np.mean(trace.get_values("x") > 0)

    stat_str = (
        f"Intercept = {summary.loc['Intercept', 'mean']:.2f} [{summary.loc['Intercept', 'hdi_2.5%']:.2f}, {summary.loc['Intercept', 'hdi_97.5%']:.2f}]"
        + "\n"
        + f"Slope = {summary.loc['x', 'mean']:.2f} [{summary.loc['x', 'hdi_2.5%']:.2f}, {summary.loc['x', 'hdi_97.5%']:.2f}]"
        + "\n"
        + f"P(Slope > 0) = {100 * summary.loc['x', 'P>0']:.2f}%"
    )
    ax.annotate(
        stat_str,
        [1, 0.05],
        ha="right",
        va="bottom",
        fontsize=4,
        xycoords="axes fraction",
    )
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    ax.set_ylim(0, 1)

fig.tight_layout()
plt.savefig(join(OUTPUT_DIR, "S_GLA-vs-MDFT_dBIC_RST.pdf"), bbox_inches="tight")
