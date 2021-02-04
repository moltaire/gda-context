#!/usr/bin/python
"""
Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour

    Plot weights from dwell-regression.

felixmolter@gmail.com
"""

from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.plot_utils import *

matplotlib = set_mpl_defaults(matplotlib)

RESULTS_DIR = join("..", "results")
OUTPUT_DIR = join("..", "figures")

predictors = [
    "col_c",
    "row_c",
    "rank_c",
    "is_target",
    "is_decoy",
    "is_probability",
    "is_chosen",
]
predictor_labels = {
    "col_c": "Column",
    "row_c": "Row",
    "rank_c": "Rank",
    "is_target": "Alt. (T)",
    "is_decoy": "Alt. (D)",
    "is_probability": "Att. (p)",
    "is_chosen": "Choice",
}

includecolor = "slategray"
excludecolor = "mediumaquamarine"
hdi_color = "slategray"

fig, axs = plt.subplots(2, 2, figsize=cm2inch(9, 6), sharex="col", sharey="col")

for e, effect in enumerate(["attraction", "compromise"]):

    # Plot regression weights
    x = np.arange(len(predictors))

    for a, analysis in enumerate(["fulltrial", "timebinned"]):
        # Read regression weights:
        df = pd.read_csv(
            join(
                RESULTS_DIR,
                "S_supplemental-gaze-analyses",
                f"dwell-regression_{analysis}",
                "estimates",
                f"dwell-regression_{analysis}_{effect}.csv",
            ),
            index_col=0,
        )

        if analysis == "fulltrial":
            vars = [f"{predictor}" for predictor in predictors]
        elif analysis == "timebinned":
            vars = [f"{predictor}:time_bin" for predictor in predictors]

        means = df.loc[vars, "mean"].values
        hdi_lower = df.loc[vars, "hdi_2.5%"].values
        hdi_upper = df.loc[vars, "hdi_97.5%"].values
        hdi_excludes_zero = (hdi_lower > 0) | (hdi_upper < 0)

        color = np.array([includecolor, excludecolor])[hdi_excludes_zero.astype(int)]

        # HDI
        axs[e, a].vlines(x, hdi_lower, hdi_upper, color=color, zorder=1)
        # Mean
        axs[e, a].scatter(
            x,
            means,
            marker="o",
            edgecolor="black",
            linewidth=0.5,
            color=color,
            s=9,
            clip_on=False,
            zorder=2,
        )

        # Labels, titles
        if e == 0:
            if analysis == "fulltrial":
                analysis_label = "Full trial"
            elif analysis == "timebinned":
                analysis_label = "Change per time bin"
            axs[e, a].set_title(analysis_label)

        if a == 0:
            axs[e, a].set_ylabel(f"{effect.capitalize()}\n\nWeight (ms)")
            axs[e, a].set_ylim(-400, 400)
            axs[e, a].set_yticks(np.arange(-400, 401, 200))
        elif a == 1:
            axs[e, a].set_ylim(-20, 40)
            axs[e, a].set_yticks(np.arange(-20, 41, 20))

    # X-ticks and labels
    if e == 1:
        for ax in axs[e, :].ravel():
            ax.set_xticks(x)
            ax.set_xticklabels(
                [predictor_labels[predictor] for predictor in predictors], rotation=90
            )
            ax.set_xlim(-0.5, len(predictors) - 0.5)


# Add horizontal lines at 0
for ax in axs.ravel():
    ax.axhline(0, linewidth=0.75, color="black", zorder=-1)

# Label panels
for ax, label in zip(axs.ravel(), list("abcd")):

    # set a different x position for the first plots in each row, because they have y-ticklabels
    if label in ["a", "c"]:
        xshift = -0.34
    else:
        xshift = -0.225
    ax.text(
        xshift,
        1.1,
        label,
        size=8,
        fontweight="bold",
        va="top",
        ha="right",
        transform=ax.transAxes,
    )

# Align y-labels
fig.align_ylabels(axs[:, 0])

# Padding
plt.tight_layout(h_pad=2.5, w_pad=1)

# Save figure
plt.savefig(
    join(OUTPUT_DIR, "S_dwell-regression_weights.pdf"), dpi=300, bbox_inches="tight"
)
