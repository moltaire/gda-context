#!/usr/bin/python
"""
Gaze-dependent accumulation in context-dependent risky choice
This script makes figure illustrating performance of the hybrid model identified in the switchboard analysis
    1) Figure 5
        a, b) Histogram of individual RST values, for participants better described by GLA or the hybrid, separate for attraction and compromise trials
        c, d) Observed vs. Hybrid-predicted RST in attraction and compromise trials
        3) Observed and predicted association of dwell time advantage and choice probability for target alternatives. Predictions from GLA and hybrid models.
    2) Supplementary Figure X:
        Supgroups: Hybrid vs. GLA predicted associations of dwell time advantage and choice probability, separately for participants with strong (RST > 0.7) attraction effect and the rest.
"""

from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from analysis.bayescorr import runBayesCorr
from plotting.plot_share import lm, plot_dwell_adv, plot_observed_predicted_rst
from plotting.plot_utils import cm2inch, set_mpl_defaults

# %% Preparations
matplotlib = set_mpl_defaults(matplotlib)
matplotlib.rcParams["font.size"] = 6

RESULTS_DIR = join("..", "results")
OUTPUT_DIR = join("..", "figures")

sample_kwargs = {"cores": 1, "random_seed": 4, "progressbar": False}

# Bins for dwell time advantage plots
dwell_time_bins = np.arange(-0.35, 0.36, 0.1).round(2)

# Bins to use for RST histograms
RST_bins = np.arange(0, 1.01, 0.1)

# Width and height of a single plotting panel (in cm)
panel_width = 4.5
panel_height = 4.5

# Color palette
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

# Which models to include, and how to name them in the plots
hybrid_variant = "sb_int-multiplicative_comp-vsmean_gbatt-false_gbalt-true_lk-free_inh-distance-dependent"
gla_variant = (
    "sb_int-multiplicative_comp-absolute_gbatt-false_gbalt-true_lk-free_inh-none"
)
models = [
    "glickman1layer",
    "mdft",
    "gaze-baseline-dyn",
    "gaze-baseline-stat",
    "eu",
    hybrid_variant,
]
model_labels = {
    "glickman1layer": "GLA",
    "mdft": "MDFT",
    "eu": "EU",
    "gaze-baseline-stat": "GB" + r"$_{stat}$",
    "gaze-baseline-dyn": "GB" + r"$_{dyn}$",
    hybrid_variant: "Hybrid",
    gla_variant: "GLA",
}

# %% Load and preprocess needed data
# Load trial data
trials = pd.read_csv(join(RESULTS_DIR, "0-clean_data", "trials_with-dwells.csv"))

# Load prediction dataframe (includes predictions of all models)
predictions = pd.read_csv(
    join(RESULTS_DIR, "3-behavioural-modeling", "predictions", "predictions_de1.csv")
)

# add predictions of hybrid switchboard model
predictions_sb = pd.read_csv(
    join(RESULTS_DIR, "4-switchboard", "predictions", "sb_predictions_de1.csv")
)
predictions_hybrid = predictions_sb.loc[predictions_sb["model"] == models[-1]]
predictions_hybrid = trials.merge(
    predictions_hybrid[["subject", "trial", "predicted_choice", "rep", "model"]],
    on=["subject", "trial"],
)
predictions = pd.concat([predictions, predictions_hybrid])

# Recode `predicted_choice` variable
predictions["predicted_choice"] = np.array(["A", "B", "C"])[
    predictions["predicted_choice"]
]
# Add `predicted_choice_tcd` variable to predicted data
predictions["predicted_choice_tcd"] = np.where(
    pd.isnull(predictions["target"]),
    np.nan,
    np.where(
        predictions["predicted_choice"] == "C",
        "decoy",
        np.where(
            ((predictions["predicted_choice"] == "A") & (predictions["target"] == "A"))
            | (
                (predictions["predicted_choice"] == "B")
                & (predictions["target"] == "B")
            ),
            "target",
            "competitor",
        ),
    ),
)

# add dwell time advantage for each alternative
for i in ["A", "B", "C", "target", "competitor", "decoy"]:
    others = [j for j in ["A", "B", "C"] if j != i]
    for df in [trials, predictions]:
        adv_i = df[f"dwell_{i}"] - df[[f"dwell_{j}" for j in others]].mean(axis=1)
        df[f"total_dwell_adv_{i}"] = pd.cut(adv_i, bins=dwell_time_bins)

# Load individual best fitting variants dataframe
best_variants = pd.read_csv(
    join(RESULTS_DIR, "4-switchboard", "individual_best-variants_bic.csv"), index_col=0
).reset_index(drop=True)
best_variants["variant"] = (
    best_variants[
        ["integration", "comparison", "gb_alt", "gb_att", "leak", "inhibition"]
    ]
    .astype(str)
    .agg("_".join, axis=1)
)

# Load DataFrame with switchboard estimates and BICs
sb_estimates = pd.read_csv(
    join(RESULTS_DIR, "4-switchboard", "estimates", "sb_estimates_de1.csv"),
    index_col=[0, 1],
).reset_index(drop=True)

# for each subject, test if the GLA variant or the hybrid variant performed better
better_variant = sb_estimates.loc[
    sb_estimates["model"].isin([gla_variant, hybrid_variant])
][["subject", "model", "bic"]]
better_variant["variant"] = better_variant["model"].apply(lambda x: model_labels[x])
better_variant = better_variant.pivot(index="subject", values="bic", columns="variant")
better_variant["better_variant"] = better_variant[["GLA", "Hybrid"]].idxmin(axis=1)
better_variant.reset_index(inplace=True)
better_variant.columns.name = None

# Load individual best fitting models
best_variants_individual = pd.read_csv(
    join(RESULTS_DIR, "4-switchboard", "individual_best-variants_bic.csv")
)

# Load observed RST
rst_obs = pd.read_csv(
    join(RESULTS_DIR, "1-behaviour", "choiceshares_across-targets.csv")
)

# Compute predicted RST for GLA and Hybrid and put everything together into one DataFrame
pred_rst = {}
for model in [models[0], models[-1]]:
    m_pred = predictions.loc[predictions["model"] == model].copy()

    # compute predicted RSTs
    m_pred_rst = (
        m_pred.loc[(m_pred["effect"].isin(["attraction", "compromise"]))]
        .groupby(["effect", "subject", "model"])["predicted_choice_tcd"]
        .value_counts()
        .rename("count")
        .reset_index()
        .pivot_table(
            index=["effect", "subject"], columns="predicted_choice_tcd", values="count"
        )
        .reset_index()
        .fillna(0)
    )
    m_pred_rst["rst"] = m_pred_rst["target"] / (
        m_pred_rst["target"] + m_pred_rst["competitor"]
    )
    m_pred_rst = m_pred_rst.merge(
        best_variants_individual[
            [
                "subject",
                "comparison",
                "integration",
                "gb_att",
                "gb_alt",
                "leak",
                "inhibition",
            ]
        ],
        on="subject",
    )
    pred_rst[model] = m_pred_rst

# Merge dataframes
df = (
    rst_obs.rename({"rst": "rst_obs"}, axis=1)
    .merge(
        pred_rst[models[0]][["subject", "effect", "rst"]].rename(
            {"rst": "rst_pred_gla"}, axis=1
        ),
        on=["subject", "effect"],
    )
    .merge(
        pred_rst[models[-1]][["subject", "effect", "rst"]].rename(
            {"rst": "rst_pred_hybrid"}, axis=1
        ),
        on=["subject", "effect"],
    )
)

df = df.merge(best_variants[["subject", "variant"]], on="subject")
df = df.merge(better_variant[["subject", "better_variant"]], on="subject")


# %% Make Figure 5
fig, axs = plt.subplots(
    3,
    2,
    gridspec_kw={"height_ratios": [2 / 3, 1, 2 / 3]},
    sharey="row",
    figsize=cm2inch(2 * panel_width, panel_height * 7 / 3),
)

for e, effect in enumerate(["attraction", "compromise"]):

    df_e = df.loc[df["effect"] == effect]
    axs[0, e].set_title(effect.capitalize())

    # Histogram of variant by RST
    for variant, color in zip(["GLA", "Hybrid"], [palette[0], palette[5]]):
        if e == 1:
            label = variant
        else:
            label = None
        axs[0, e].hist(
            df_e.loc[df["better_variant"] == variant, "rst_obs"],
            color=color,
            edgecolor="white",
            linewidth=0.5,
            bins=RST_bins,
            label=label,
            alpha=0.75,
        )

        ## Labels, ticks, limits
        axs[0, e].set_xticks(np.arange(0, 1.01, 0.25))
        axs[0, e].set_xlim(0, 1)
        axs[0, e].set_xlabel("observed RST")
        if e == 0:
            axs[0, e].set_ylabel("N\nparticipants")
            axs[0, e].set_ylim(0, 20)
            axs[0, e].set_yticks([0, 10, 20])
        else:
            axs[0, e].yaxis.set_tick_params(labelbottom=True)

    # Observed-predicted linear model correlation plot
    ax = plot_observed_predicted_rst(
        rst_obs,
        predictions.loc[predictions["model"] == hybrid_variant],
        effect=effect,
        scatter_color=palette[5],
        line_color=palette[5],
        ax=axs[1, e],
        sample_kwargs=sample_kwargs,
    )
    ax.set_title(None)

    if e == 0:
        axs[1, e].set_ylabel("Hybrid variant\npredicted RST")
        axs[1, e].set_yticks(np.arange(0, 1.01, 0.25))
        axs[1, e].set_ylim(0, 1)
    else:
        axs[1, e].yaxis.set_tick_params(labelbottom=True)
        axs[1, e].set_ylabel(None)

    # Gaze advantage prediction
    ax = axs[2, e]
    ## Observed data (even trials) as bars
    ax, summary = plot_dwell_adv(
        trials.loc[(trials["effect"] == effect) & (trials["trial"] % 2 == 0)],
        kind="bar",
        color="white",
        edgecolor="black",
        linewidth=0.5,
        alternative="target",
        choicecol="choice_tcd",
        ax=ax,
    )

    ## Model predictions as lines
    for model, color in zip([models[0], models[-1]], [palette[0], palette[5]]):
        ax, summary = plot_dwell_adv(
            predictions.loc[
                (predictions["model"] == model)
                & (predictions["effect"] == effect)
                & (predictions["trial"] % 2 == 1)
            ],
            kind="line",
            color=color,
            choicecol="predicted_choice_tcd",
            alternative="target",
            label=model_labels[model],
            ax=ax,
        )

    if e == 0:
        axs[2, e].set_ylabel("P(Choose\nTarget)")
    else:
        axs[2, e].set_ylabel(None)
        axs[2, e].yaxis.set_tick_params(labelbottom=True)
    axs[2, e].set_xlabel("Target rel. dwell adv.")
    axs[2, e].set_yticks([0, 0.5, 1])

# Adjust margins and padding
plt.tight_layout(h_pad=2, w_pad=3)

# Add legends (call this after tight_layout so that the legend does not squash the rest)
## Best fitting models for histogram
axs[0, 1].legend(
    loc="center left",
    ncol=1,
    frameon=False,
    columnspacing=1,
    bbox_to_anchor=(1, 0.5),
    handlelength=1,
    title="Better variant",
)

## Model predictions for dwell advantage
axs[2, 1].legend(
    loc="center left",
    ncol=1,
    frameon=False,
    columnspacing=1,
    bbox_to_anchor=(1, 0.5),
    handlelength=1,
)

# Align y-labels in each column
fig.align_ylabels()

# Label panels
from string import ascii_lowercase

for ax, label in zip(axs.ravel(), list(ascii_lowercase)):
    # Place axis labels in figure space, so that they are aligned
    # https://stackoverflow.com/a/52309638
    xshift = 0.095
    X = ax.get_position().x0
    Y = ax.get_position().y1
    fig.text(X - xshift, Y, label, size=10, weight="bold", ha="right", va="center")

# Save the figure
plt.savefig(join(OUTPUT_DIR, "5-hybrid-variant.pdf"), bbox_inches="tight")


# %% Supplemental Figure: Hybrid & GLA predictions of P(choice) vs. dwell time advantage, for subgroups (AE RST > 0.7, vs. rest)