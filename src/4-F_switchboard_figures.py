#!/usr/bin/python
"""
Gaze-dependent accumulation in context-dependent risky choice
This script makes figures for the switchboard analysis.
    1) Heatmap of all variants' BIC values
    2) Barplot of switch-level mean BICs
    3) Barplot of switch-level individual best-fitting counts
    4) Context-effect predictions of the two models that described most participants best
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
from plotting.plot_share import factorial_heatmap, lm
from plotting.plot_utils import break_after_nth_tick, cm2inch, set_mpl_defaults

matplotlib = set_mpl_defaults(matplotlib)
matplotlib.rcParams["font.size"] = 6

RESULTS_DIR = join("..", "results")
OUTPUT_DIR = join("..", "figures")

sample_kwargs = {"cores": 1, "random_seed": 4, "progressbar": False}

switches = dict(
    # order by switchboard figure from top to bottom
    inhibition=["none", "free", "gaze-dependent", "distance-dependent"],
    leak=["none", "free", "gaze-dependent"],
    comparison=["absolute", "vsmean", "n.d."],
    gb_alt=[True, False],
    integration=["additive", "multiplicative"],
    gb_att=[True, False],
)

switch_labels = {
    "comparison": "Comparison",
    "integration": "Integration",
    "gb_alt": "GD" + r"$_{Alt}$",
    "gb_att": "GD" + r"$_{Att}$",
    "inhibition": "Inhibition",
    "leak": "Leak",
}

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


# %% 1) Heatmap of all variants' BIC values
# -----------------------------------------

height = 4.5
width = 12

fontsize = 5
matplotlib.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    }
)

# Load mean BIC dataframe
variants_mean_bic = pd.read_csv(
    join(RESULTS_DIR, "4-switchboard", "variants_mean_bic.csv")
)
# Recode switches to ordered Categoricals so I can control their order
variants_mean_bic["integration"] = pd.Categorical(
    variants_mean_bic["integration"], categories=[r"$+$", r"$\times$"], ordered=True
)
variants_mean_bic["comparison"] = pd.Categorical(
    variants_mean_bic["comparison"],
    categories=["independent", "comparative"],
    ordered=True,
)
variants_mean_bic["gb_att"] = pd.Categorical(
    variants_mean_bic["gb_att"], categories=[False, True], ordered=True
)
variants_mean_bic["gb_alt"] = pd.Categorical(
    variants_mean_bic["gb_alt"], categories=[False, True], ordered=True
)
variants_mean_bic["leak"] = pd.Categorical(
    variants_mean_bic["leak"], categories=["None", "Constant", "Gaze"], ordered=True
)
variants_mean_bic["inhibition"] = pd.Categorical(
    variants_mean_bic["inhibition"],
    categories=["None", "Constant", "Distance", "Gaze"],
    ordered=True,
)

# Make the figure
fig, ax = plt.subplots(figsize=cm2inch(width, height), dpi=300)
ax, values = factorial_heatmap(
    variants_mean_bic,
    row_factors=["comparison", "gb_att", "integration"],
    col_factors=["leak", "gb_alt", "inhibition"],
    value_var="BIC",
    cmap="RdYlGn_r",
    norm=TwoSlopeNorm(vcenter=275),  # mean_bics['BIC'].mean()),
    factor_labels={"integration": "Integration", "inhibition": "Inhibition"},
    level_labels={
        "comparison": {"independent": "Independent", "comparative": "Comparative"},
        "gb_alt": {False: "No GD" + r"$_{Alt}$", True: "With GD" + r"$_{Alt}$"},
        "leak": {"None": "None", "Constant": "Leak\nConstant", "Gaze": "Gaze"},
        "inhibition": {
            "None": "None",
            "Constant": "Const.",
            "Distance": "Dist.",
            "Gaze": "Gaze",
        },
        "gb_att": {True: "With\nGD" + r"$_{Att}$", False: "No\nGD" + r"$_{Att}$"},
        "integration": {"multiplicative": r"$\times$", "additive": r"$+$"},
    },
    xlabel_rotation=90,
    pad_label_bar=0.2,
    pad_per_factor_top=1.75,
    pad_per_factor_right=2.5,
    pad_colorbar=0.05,
    cb_args={"fraction": 0.0125},
)

# Mark minima (two, because the top two models are actually equivalent, differences due to noise in optimization)
xcoords, ycoords = np.where(values < 233)
for x, y in list(zip(xcoords, ycoords)):
    rect = Rectangle(
        [y - 0.5, x - 0.5],
        width=1,
        height=1,
        linewidth=1,
        edgecolor="white",
        facecolor="none",
    )
    ax.add_patch(rect)

fig.tight_layout()

plt.savefig(
    join(OUTPUT_DIR, "4-1_switchboard_bic_heatmap.pdf"), dpi=300, bbox_inches="tight"
)

# %% 2) Switch-level mean BIC barplot
# -----------------------------------

width = 11.5
height = 4

fontsize = 6
matplotlib.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    }
)

# Load data
switch_levels_bic = pd.read_csv(
    join(RESULTS_DIR, "4-switchboard", "switch-levels_bic.csv")
)


fig, axs = plt.subplots(
    2,
    3,
    figsize=cm2inch(width, height),
    sharex=True,
    dpi=300,
    gridspec_kw={"height_ratios": [3, 2]},
)

for i, (switch, color) in enumerate(
    zip(
        ["inhibition", "leak", "comparison", "gb_alt", "integration", "gb_att"],
        [
            "lightpink",
            "paleturquoise",
            "slategray",
            "mediumaquamarine",
            "darksalmon",
            "indianred",
        ],
    )
):
    ax = axs.ravel()[i]
    ax.set_title(switch_labels[switch], fontsize=6, y=0.95)

    means = switch_levels_bic.loc[switch_levels_bic["switch"] == switch, "mean"].values
    sems = switch_levels_bic.loc[switch_levels_bic["switch"] == switch, "sem"].values
    sort = np.argsort(means)[::-1]
    ylabels = switch_levels_bic.loc[
        switch_levels_bic["switch"] == switch, "label"
    ].values
    y = np.arange(len(means))
    ax.barh(y, means[sort], color=color)
    ax.hlines(
        y,
        means[sort] - sems[sort],
        means[sort] + sems[sort],
        color="black",
        linewidth=0.75,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels[sort], fontsize=5)
    ax.set_ylim(-1, len(y) - 0.5)

    ax.set_xlim(200, 300)
    ax.set_xticks(np.arange(200, 401, 100))
    ax = break_after_nth_tick(ax, n=0, axis="x")


for ax in axs[-1, :]:
    ax.set_xlabel("BIC")
    ax = break_after_nth_tick(ax, n=0, axis="x")
    ax.set_xticklabels(np.insert(np.arange(200, 401, 100)[1:], 0, 0))

plt.tight_layout(w_pad=0.25, h_pad=1)

plt.savefig(join(OUTPUT_DIR, "4-2_switch-level_bic.pdf"), dpi=300, bbox_inches="tight")

# %% 3) Switch-level individual counts
# ------------------------------------
width = 11.5
height = 4

fontsize = 6
matplotlib.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    }
)

# Load the data
best_switches_individual_counts = pd.read_csv(
    join(RESULTS_DIR, "4-switchboard", "switch-levels_individual-counts.csv"),
    index_col=0
)

fig, axs = plt.subplots(
    2,
    3,
    figsize=cm2inch(width, height),
    sharex=True,
    dpi=300,
    gridspec_kw={"height_ratios": [3, 2]},
)

for i, (switch, color) in enumerate(
    zip(
        ["inhibition", "leak", "comparison", "gb_alt", "integration", "gb_att"],
        [
            "lightpink",
            "paleturquoise",
            "slategray",
            "mediumaquamarine",
            "darksalmon",
            "indianred",
        ],
    )
):
    ax = axs.ravel()[i]
    ax.set_title(switch_labels[switch], fontsize=6, y=0.95)

    counts = best_switches_individual_counts.loc[best_switches_individual_counts["switch"] == switch, "count"].values
    sort = np.argsort(counts)
    ylabels = best_switches_individual_counts.loc[
        best_switches_individual_counts["switch"] == switch, "label"
    ].values
    y = np.arange(len(counts))
    ax.barh(y, counts[sort], color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels[sort], fontsize=5)
    ax.set_ylim(-1, len(y) - 0.5)

    ax.set_xlim(0, 40)
    ax.set_xticks(np.arange(0, 41, 10))


for ax in axs[-1, :]:
    ax.set_xlabel("N subjects")

plt.tight_layout(w_pad=0.25, h_pad=1)


plt.savefig(
    join(OUTPUT_DIR, "S_switch-level_individual_counts.pdf"),
    dpi=300,
    bbox_inches="tight",
)


# %% 4) Context-effect predictions of the two models that described most participants best

model1 = "sb_int-multiplicative_comp-absolute_gbatt-false_gbalt-true_lk-free_inh-none"
model2 = "sb_int-multiplicative_comp-vsmean_gbatt-false_gbalt-true_lk-free_inh-distance-dependent"
model_labels = {model1: "GLA~variant", model2: "Hybrid~variant"}

# Observed RST
rst_obs = pd.read_csv(
    join(RESULTS_DIR, "1-behaviour", "choiceshares_across-targets.csv")
)

# Load SB model predictions
sb_predictions = pd.read_csv(
    join(RESULTS_DIR, "4-switchboard", "predictions", "sb_predictions_de1.csv")
)

# Individual best fitting models
best_variants_individual = pd.read_csv(
    join(RESULTS_DIR, "4-switchboard", "individual_best-variants_bic.csv")
)

# Code predicted choice as target, competitor or decoy
sb_predictions["pred_choice_abc"] = np.array(["A", "B", "C"])[
    sb_predictions["predicted_choice"]
]
sb_predictions["pred_choice_tcd"] = np.where(
    sb_predictions["pred_choice_abc"] == "C",
    "decoy",
    np.where(
        sb_predictions["pred_choice_abc"] == sb_predictions["target"],
        "target",
        "competitor",
    ),
)
sb_predictions.loc[pd.isnull(sb_predictions["target"]), "pred_choice_tcd"] = np.nan

# Compute predicted RST for the two variants
pred_rst = {}
for model in [model1, model2]:
    m_pred = sb_predictions.loc[sb_predictions["model"] == model].copy()

    # compute predicted RSTs
    m_pred_rst = (
        m_pred.loc[(m_pred["effect"].isin(["attraction", "compromise"]))]
        .groupby(["effect", "subject", "model"])["pred_choice_tcd"]
        .value_counts()
        .rename("count")
        .reset_index()
        .pivot_table(
            index=["effect", "subject"], columns="pred_choice_tcd", values="count"
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
        pred_rst[model1][["subject", "effect", "rst"]].rename(
            {"rst": "rst_pred_m1"}, axis=1
        ),
        on=["subject", "effect"],
    )
    .merge(
        pred_rst[model2][["subject", "effect", "rst"]].rename(
            {"rst": "rst_pred_m2"}, axis=1
        ),
        on=["subject", "effect"],
    )
)

# Make the figure
fig, axs = plt.subplots(2, 2, dpi=300, figsize=cm2inch(10, 9.5))

i = 0
for model, color in zip(["m1", "m2"], ["slategray", "darksalmon"]):
    for effect in ["attraction", "compromise"]:

        x = df.loc[df["effect"] == effect, "rst_obs"]
        y = df.loc[df["effect"] == effect, f"rst_pred_{model}"]

        # Linear Model
        ax, trace, summary = lm(
            x=x,
            y=y,
            ax=axs.ravel()[i],
            scatter_color=color,
            line_color="lightgray",
            xrange=[0, 1],
            sample_kwargs=sample_kwargs,
        )

        # Correlation
        corrTrace = runBayesCorr(y1=x.values, y2=y.values, sample_kwargs=sample_kwargs)
        corrSummary = pm.summary(corrTrace, hdi_prob=0.95)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], lw=0.5, color="gray", zorder=-1)
        ax.set_xlabel("observed RST")
        ax.set_ylabel("predicted RST")

        stat_str = (
            f"r = {corrSummary.loc['r', 'mean']:.2f} [{corrSummary.loc['r', 'hdi_2.5%']:.2f}, {corrSummary.loc['r', 'hdi_97.5%']:.2f}]"
            + "\n"
            + f"Intercept = {summary.loc['Intercept', 'mean']:.2f} [{summary.loc['Intercept', 'hdi_2.5%']:.2f}, {summary.loc['Intercept', 'hdi_97.5%']:.2f}]"
            + "\n"
            + f"Slope = {summary.loc['x', 'mean']:.2f} [{summary.loc['x', 'hdi_2.5%']:.2f}, {summary.loc['x', 'hdi_97.5%']:.2f}]"
        )
        ax.annotate(
            stat_str, [1, 0.05], ha="right", va="bottom", fontsize=4, ma="right"
        )

        i += 1

axs[0, 0].set_title("Attraction", fontsize=7)
axs[0, 1].set_title("Compromise", fontsize=7)
axs[0, 0].set_ylabel(r"$\bf{" + model_labels[model1] + "}$" + "\n\npredicted RST")
axs[1, 0].set_ylabel(r"$\bf{" + model_labels[model2] + "}$" + "\n\npredicted RST")

fig.tight_layout(h_pad=4, w_pad=4)

# Label the axes
for ax, label in zip(axs.ravel(), list("abcd")):

    # Left-align axes
    ax.set_anchor("W")

    # Place axis labels in figure space, so that they are aligned
    # https://stackoverflow.com/a/52309638
    X = ax.get_position().x0
    Y = ax.get_position().y1
    fig.text(X - 0.1, Y, label, size=10, weight="bold")

plt.savefig(
    join(OUTPUT_DIR, "5-best-2-model-variants_rst-predictions.pdf"), dpi=300,
)

