#!usr/bin/python
"""
This script makes figures for the switchboard analysis.
    1) Heatmap of all variants' BIC values
    2) Barplot of switch-level mean BICs
    3) Barplot of switch-level individual best-fitting counts
"""
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import DivergingNorm
from matplotlib.patches import Rectangle

from plotting.plot_share import factorial_heatmap
from plotting.plot_utils import break_after_nth_tick, cm2inch, set_mpl_defaults

matplotlib = set_mpl_defaults(matplotlib)
matplotlib.rcParams["font.size"] = 6

RESULTS_DIR = join("..", "results", "4-switchboard")
OUTPUT_DIR = join("..", "figures")

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
# --------------------------------------

# Load mean BIC dataframe
variants_mean_bic = pd.read_csv(join(RESULTS_DIR, "variants_mean_bic.csv"))
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
fig, ax = plt.subplots(figsize=cm2inch(12, 5), dpi=300)
ax, values = factorial_heatmap(
    variants_mean_bic,
    row_factors=["leak", "gb_att", "integration"],
    col_factors=["comparison", "gb_alt", "inhibition"],
    value_var="BIC",
    cmap="RdYlGn_r",
    norm=DivergingNorm(vcenter=275),  # mean_bics['BIC'].mean()),
    factor_labels={"integration": "Integration", "inhibition": "Inhibition"},
    level_labels={
        "comparison": {"independent": "Independent", "comparative": "Comparative"},
        "gb_alt": {False: "No GD" + r"$_{Alt}$", True: "With GD" + r"$_{Alt}$"},
        "leak": {"None": "None", "Constant": "Leak\nConstant", "Gaze": "Gaze"},
        "inhibition": {
            "None": "None",
            "Constant": "Constant",
            "Distance": "Distance",
            "Gaze": "Gaze",
        },
        "gb_att": {True: "With\nGD" + r"$_{Att}$", False: "No\nGD" + r"$_{Att}$"},
        "integration": {"multiplicative": r"$\times$", "additive": r"$+$"},
    },
    xlabel_rotation=90,
    pad_label_bar=0.1,
    pad_per_factor=2.5,
    pad_colorbar=0.075,
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

plt.savefig(
    join(OUTPUT_DIR, "4-1_switchboard_bic_heatmap.png"), dpi=300, bbox_inches="tight"
)

# %% 2) Switch-level mean BIC barplot
# -----------------------------------
# Load data
switch_levels_bic = pd.read_csv(join(RESULTS_DIR, "switch-levels_bic.csv"))

# Set up colors
colors = np.array(
    [
        palette[c]
        for c in pd.Categorical(
            switch_levels_bic["switch"],
            categories=[
                "comparison",
                "integration",
                "gb_alt",
                "gb_att",
                "leak",
                "inhibition",
            ],
            ordered=True,
        ).codes
    ]
)

# Make the figure
fig, ax = plt.subplots(figsize=cm2inch(2, 6), dpi=300)

bic_means = switch_levels_bic["mean"]
bic_sems = switch_levels_bic["sem"]

bars = ax.barh(np.arange(len(switch_levels_bic)), bic_means, color=colors)
ax.hlines(
    y=np.arange(len(switch_levels_bic)),
    xmin=bic_means - bic_sems,
    xmax=bic_means + bic_sems,
    linewidth=0.75,
)
ax.set_yticks(np.arange(len(switch_levels_bic)))
ax.set_yticklabels(switch_levels_bic["label"], fontsize=5)
ax.set_ylim(len(switch_levels_bic), -0.5)
ax.set_xlabel("Mean BIC")
ax.set_xlim(200, 400)
ax.set_xticks(np.arange(200, 401, 100))
ax = break_after_nth_tick(ax, n=0, axis="x")
ax.set_xticklabels(np.insert(np.arange(200, 401, 100)[1:], 0, 0))

switch_indices = (
    switch_levels_bic["switch"].drop_duplicates().index
)  # identify indices where switches occur first
ax.legend(
    handles=[bars[i] for i in switch_indices],
    labels=[
        switch_labels[switch]
        for switch in switch_levels_bic["switch"][switch_indices].values
    ],
    fontsize=5,
    bbox_to_anchor=(0.9, 1.025),
)

plt.savefig(join(OUTPUT_DIR, "4-2_switch-level_bic.png"), dpi=300, bbox_inches="tight")

# %% 3) Switch-level individual counts
# Load the data
best_switches_individual_counts = pd.read_csv(
    join(RESULTS_DIR, "switch-levels_individual-counts.csv")
)

# Make the figure
colors = np.array(
    [
        palette[c]
        for c in pd.Categorical(
            best_switches_individual_counts["switch"],
            categories=[
                "comparison",
                "integration",
                "gb_alt",
                "gb_att",
                "leak",
                "inhibition",
            ],
            ordered=True,
        ).codes
    ]
)

# Make the figure
fig, ax = plt.subplots(figsize=cm2inch(3, 4.5), dpi=300)

counts = best_switches_individual_counts["count"]

bars = ax.barh(np.arange(len(best_switches_individual_counts)), counts, color=colors)

ax.set_yticks(np.arange(len(best_switches_individual_counts)))
ax.set_yticklabels(best_switches_individual_counts["label"], fontsize=5)
ax.set_ylim(len(best_switches_individual_counts), -0.5)
ax.set_xlabel("N best fitting")
ax.set_xticks(np.arange(0, 41, 10))
ax.set_xlim(0, 40)

switch_indices = (
    best_switches_individual_counts["switch"].drop_duplicates().index
)  # identify indices where switches occur first
ax.legend(
    handles=[bars[i] for i in switch_indices],
    labels=[
        switch_labels[switch]
        for switch in best_switches_individual_counts["switch"][switch_indices].values
    ],
    fontsize=5,
    bbox_to_anchor=(1.05, 1.025),
)

plt.savefig(
    join(OUTPUT_DIR, "4-2_switch-level_individual_counts.png"),
    dpi=300,
    bbox_inches="tight",
)
