# /usr/bin/python
"""
This script makes figures of the behavioural-modeling analyses.
    1) Composite figure of
        a) BIC violin plot
        b) Barplot of individually best fitting model counts
        c, d) Observed vs. predicted RST for the winning model
    2) Supplementary Figure of RST vs relative model fit (GLA vs MDFT)
"""

import logging
import warnings
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.io import loadmat

from analysis.bayescorr import runBayesCorr
from plotting.plot_share import lm, violin
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


# Model labels to use in the figures
model_names = {
    "glickman1layer": "GLA",
    "mdft": "MDFT",
    "eu": "EU",
    "gaze-baseline-stat": "GB$_{stat}$",
    "gaze-baseline-dyn": "GB$_{dyn}$"
}

short_names = {
    "glickman1layer": "GLA",
    "mdft": "MDFT",
    "eu": "EU",
    "gaze-baseline-stat": "GB$_{stat}$",
    "gaze-baseline-dyn": "GB$_{dyn}$",
}


# MCMC settings passed on to pm.sample
sample_kwargs = {"cores": 1, "random_seed": SEED, "progressbar": False}

panel_width = 4.5
panel_height = 4.5

# %% 1) Make the composite figure
# -------------------------
fig, axs = plt.subplots(
    1,
    4,
    gridspec_kw={"width_ratios": [5, 3.5, 5, 5]},
    figsize=cm2inch(4 * panel_width, panel_height),
)

# a) BIC Violin
# Load and preprocess data
estimates = pd.read_csv(
    join(RESULTS_DIR, "3-behavioural-modeling", "estimates", "estimates_de1.csv")
)[["subject", "model", "bic"]]

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
    [model for model in model_order if model in model_names.keys()]
]
bics_wide.columns.name = None

# Make the figure
axs[0] = violin(
    data=bics_wide, ax=axs[0], palette=["slategray"] * len(bics_wide.columns)
)
axs[0].set_xticks(range(len(model_names)))
axs[0].set_xticklabels([model_names[c] for c in bics_wide.columns])
axs[0].set_ylabel("BIC")
axs[0].set_xlabel(None)
axs[0].set_ylim(0, 600)
# add dashed line for random model
axs[0].axhline(
    -2 * 225 * np.log(1 / 3), linestyle="--", color="black", zorder=-1, linewidth=0.75,
)

# b) Count of individual best fitting models

# Load BMS results from MATLAB
bms = loadmat(join(RESULTS_DIR, "3-behavioural-modeling", "model-comparison_bms_results.mat"))
models = np.concatenate(bms['result']["model_names"][0])[1:]
xp = bms["result"]["xp"][0][0].flatten()

# Load data
individual_best_models = pd.read_csv(
    join(
        RESULTS_DIR,
        "3-behavioural-modeling",
        "model-comparison_individual-best-models.csv",
    )
)
N = individual_best_models["model"].value_counts()

# Make the figure
axs[1].bar(np.arange(len(N)), N, color="slategray")
axs[1].set_xlim(-0.75, 2.75)
axs[1].set_xticks(np.arange(len(N)))
axs[1].set_xticklabels([model_names[i] for i in N.index])
axs[1].set_ylabel("N Lowest BIC")
axs[1].set_ylim(0, 40)

# Make inset for exceedance probabilities
axins = axs[1].inset_axes(bounds=(0.6, 0.6, 0.4, 0.4))
axins.bar(np.arange(len(models)), xp[np.argsort(xp)[::-1]], color='#666666')
axins.set_xticks(np.arange(len(models)))
axins.set_xticklabels([short_names[model.replace('_', '-')] for model in models[np.argsort(xp)[::-1]]], fontsize=4, rotation=90)
axins.set_ylabel("Exceedance\nprobability", fontsize=4, labelpad=-5)
axins.set_yticks([0, 1])
axins.set_ylim(0, 1)

# c), d) GLA predicted vs. observed RST in attraction and compromise trials
# Read and process the data
# Observed RST
cs = pd.read_csv(join(RESULTS_DIR, "1-behaviour", "choiceshares_across-targets.csv"))

# Read model predictions and compute GLA predicted RST
predictions = pd.read_csv(
    join(RESULTS_DIR, "3-behavioural-modeling", "predictions", "predictions_de1.csv")
)
predictions["pred_choice_abc"] = np.array(["A", "B", "C"])[
    predictions["predicted_choice"]
]
predictions["pred_choice_tcd"] = np.where(
    predictions["pred_choice_abc"] == "C",
    "decoy",
    np.where(
        predictions["pred_choice_abc"] == predictions["target"], "target", "competitor"
    ),
)
predictions.loc[pd.isnull(predictions["target"]), "pred_choice_tcd"] = np.nan
gla_pred = predictions.loc[predictions["model"] == "glickman1layer"]

rst_gla_pred = (
    gla_pred.loc[(gla_pred["effect"].isin(["attraction", "compromise"]))]
    .groupby(["effect", "subject", "model"])["pred_choice_tcd"]
    .value_counts()
    .rename("count")
    .reset_index()
    .pivot_table(index=["effect", "subject"], columns="pred_choice_tcd", values="count")
    .reset_index()
    .fillna(0)
)
rst_gla_pred["rst"] = rst_gla_pred["target"] / (
    rst_gla_pred["target"] + rst_gla_pred["competitor"]
)

# Merge observed and GLA-predicted RST dataframes
rst = (
    cs[["subject", "effect", "rst"]]
    .rename({"rst": "rst_obs"}, axis=1)
    .merge(
        rst_gla_pred[["subject", "effect", "rst"]].rename({"rst": "rst_pred"}, axis=1),
        on=["subject", "effect"],
    )
)

# # Make the figure(s)
for effect, ax in zip(["attraction", "compromise"], [axs[2], axs[3]]):
    rst_e = rst.loc[rst["effect"] == effect]

    # Linear Model
    ax, trace, summary = lm(
        x=rst_e["rst_obs"],
        y=rst_e["rst_pred"],
        ax=ax,
        scatter_color="slategray",
        line_color="lightgray",
        xrange=[0, 1],
        sample_kwargs=sample_kwargs,
    )

    # Correlation
    corrTrace = runBayesCorr(
        y1=rst_e["rst_obs"], y2=rst_e["rst_pred"], sample_kwargs=sample_kwargs
    )
    corrSummary = pm.summary(corrTrace, hdi_prob=0.95)

    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], lw=0.5, color="gray", zorder=-1)
    ax.set_xlabel("observed RST")
    ax.set_ylabel("GLA predicted RST")
    ax.set_title(f"{effect.capitalize()}")

    stat_str = (
        f"$r$ = {corrSummary.loc['r', 'mean']:.2f} [{corrSummary.loc['r', 'hdi_2.5%']}, {corrSummary.loc['r', 'hdi_97.5%']}]"
        + "\n"
        f"Intercept = {summary.loc['Intercept', 'mean']}, Slope = {summary.loc['x', 'mean']}"
    )
    ax.annotate(stat_str, [1, 0.05], ha="right", va="bottom", fontsize=4)

fig.tight_layout(h_pad=4, w_pad=2)

# Label panels
for ax, label in zip(axs.ravel(), list("abcd")):
    # Place axis labels in figure space, so that they are aligned
    # https://stackoverflow.com/a/52309638
    X = ax.get_position().x0
    Y = ax.get_position().y1
    fig.text(X - 0.05, Y + 0.05, label, size=10, weight="bold", ha="right", va="center")

plt.savefig(join(OUTPUT_DIR, "3-model-comparison.pdf"))


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
plt.savefig(join(OUTPUT_DIR, "3-dBIC-gla-mdft_RST.pdf"), bbox_inches="tight")
