#!/usr/bin/python
"""
Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour
This script makes figures for the switchboard analysis.
    1) Heatmap of all variants' BIC values (Fig 4c)
    2) Barplot of switch-level mean BICs (Fig 4b)
    3) Barplot of switch-level individual best-fitting counts (Supplementary Figure)
"""
import logging
import warnings
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from plotting.plot_share import factorial_heatmap
from plotting.plot_utils import break_after_nth_tick, cm2inch, set_mpl_defaults

matplotlib = set_mpl_defaults(matplotlib)
matplotlib.rcParams["font.size"] = 6

warnings.filterwarnings("ignore")

logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)


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
    cmap="viridis_r",
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
        linewidth=1.5,
        edgecolor="white",
        facecolor="none",
    )
    ax.add_patch(rect)

# Mark hybrid model
x = 5
y = 14
rect = Rectangle(
    [y - 0.5, x - 0.5],
    width=1,
    height=1,
    linewidth=1,
    ls=(0, (1.5, 1)),
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
    index_col=0,
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

    counts = best_switches_individual_counts.loc[
        best_switches_individual_counts["switch"] == switch, "count"
    ].values
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
