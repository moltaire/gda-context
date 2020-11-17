# usr/bin/python
"""
Figure 2: Context Effects in Behaviour and Gaze
"""
import math
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary

from plotting.plot_share import violin, addEffectBar
from plotting.plot_utils import *

matplotlib = set_mpl_defaults(matplotlib)

PALETTE = ["turquoise", "violet", "gold"]

# Set data and output directories
DATA_DIR = join("..", "results", "0-clean_data")
RESULTS_DIR = join("..", "results")
OUTPUT_DIR = join("..", "figures")

# Load choice share data
cs = pd.read_csv(join(RESULTS_DIR, "1-behaviour", "choiceshares_across-targets.csv"))

# Load gaze data
rdwells = pd.read_csv(join(RESULTS_DIR, "2-gaze", "dwells_across-targets.csv"))

# Components: Violin plot, ternary plot, some helper functions
# ------------------------------------------------------------


def color_point(x, y, z, scale, alpha=0.1):
    """Create a color value for a simplex point.

    This function is lifted from the examples from the python-ternary GitHub page.
    
    Args:
        x (float): x value
        y (float): y value
        z (float): z value
        scale (float): Scale parameter for ternary plot.
        alpha (float, optional): Alpha level of returned color. Defaults to 0.1.
    
    Returns:
        tuple: rgba tuple
    """
    w = 255
    x_color = x * w / scale
    y_color = y * w / scale
    z_color = z * w / scale
    r = math.fabs(w - z_color) / w
    g = math.fabs(w - y_color) / w
    b = math.fabs(w - x_color) / w
    return (r, g, b, alpha)


def generate_heatmap_data(scale=5):
    """Generates a heatmap data dictionary for ternary plots.
    This function is lifted from the examples on the python-ternary GitHub page.
    
    Args:
        scale (int, optional): Scale parameter for ternary plots. Defaults to 5.
    
    Returns:
        dict: Heatmap data
    """
    from ternary.helpers import simplex_iterator

    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = color_point(i, j, k, scale)
    return d


def ternaryTCD(choiceshares, scale=10, group_mean=True, ax=None):
    """Make a ternary plot of choiceshares for target, competitor and decoy options.
    
    Args:
        choiceshares (pandas.DataFrame): DataFrame containing participant-wise choiceshares for target, competitor and decoy.
        scale (int, optional): Scale parameter for ternary plots. Defaults to 10.
        group_mean (bool, optional): Toggle plotting the group mean. Defaults to True.
        ax (matplotlib.axis, optional): Axis to plot on. Defaults to None.
    
    Returns:
        matplotlib.axis: Axis with ternary plot.
    """
    scatter_data = choiceshares[["decoy", "competitor", "target"]].values

    if ax is None:
        fig, ax = plt.subplots(figsize=cm2inch(4.5, 4.5))

    fig, tax = ternary.figure(ax=ax, scale=scale)

    # Fill the background with the colormap
    background_data = generate_heatmap_data(scale)
    tax.heatmap(background_data, style="triangular", use_rgba=True, colorbar=False)

    # Scatter individuals
    ## translucent white face
    tax.scatter(
        scale * scatter_data,
        facecolors="white",
        edgecolors="none",
        linewidths=0.5,
        s=12,
        alpha=0.6,
        zorder=9,
    )
    # more solid black edges
    tax.scatter(
        scale * scatter_data,
        facecolors="none",
        edgecolors="black",
        linewidths=0.5,
        s=12,
        alpha=0.9,
        zorder=9,
    )

    # Group mean
    if group_mean:
        tax.scatter(
            scatter_data.mean(axis=0)[None, :] * scale,
            marker="X",
            facecolors="crimson",
            edgecolors="black",
            linewidth=0.5,
            alpha=0.95,
            s=24,
            zorder=10,
        )

    # Isoprob lines
    center = np.ones(3) * scale / 3
    left = np.array([0, 1, 1]) * scale / 2
    bottom = np.array([1, 0, 0]) * scale / 2
    right = np.array([1, 1, 0]) * scale / 2
    for side in [left, bottom, right]:
        tax.line(center, side, linewidth=0.5, color="black", zorder=2, alpha=0.5)

    # draw a boundary
    tax.boundary(linewidth=1)

    return tax


# Make Figure 2
# -------------

if __name__ == "__main__":

    # Size of each panel
    panel_width = 4.5
    panel_height = 4.5
    TERNSCALE = 100

    fig, axs = plt.subplots(
        2,
        3,
        gridspec_kw={"width_ratios": [3, 3, 3]},
        figsize=cm2inch(3 * panel_width, 2 * panel_height),
    )

    for e, effect in enumerate(["attraction", "compromise"]):

        # Ternary plot: P(Choice)
        tax = ternaryTCD(
            choiceshares=cs.loc[cs["effect"] == effect], scale=TERNSCALE, ax=axs[e, 0]
        )
        # Remove spines and ticks, but include a row-label for the effect
        ax = tax.get_axes()
        ax.set_xlim(-0.01 * TERNSCALE, 1.01 * TERNSCALE)
        ax.set_ylim(
            -0.01 * TERNSCALE * np.sqrt(3) / 2, 1.1 * TERNSCALE * np.sqrt(3) / 2
        )  # height of equilateral triangle = a * sqrt(3) / 2
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_ylabel(
            "{effect}".format(effect=effect.capitalize()),
            labelpad=6,
            size=7,
            # fontweight="bold",
        )
        ax.set_title(None)
        ax.text(-0.025 * TERNSCALE, -0.025 * TERNSCALE, "Target", ha="left", va="top")
        ax.text(0.5 * TERNSCALE, 0.95 * TERNSCALE, "Competitor", ha="center", va="top")
        ax.text(1.025 * TERNSCALE, -0.025 * TERNSCALE, "Decoy", ha="right", va="top")

        # Violin: P(Choice)
        axs[e, 1] = violin(
            data=cs.loc[cs["effect"] == effect, ["target", "competitor", "decoy"]],
            palette=PALETTE,
            ax=axs[e, 1],
        )
        axs[(e, 1)].set_yticks(np.arange(0, 1.01, 0.2))
        axs[(e, 1)].set_ylim(0, 1)
        axs[e, 1].set_ylabel("P(Choice)")
        axs[e, 1].set_xlabel(None)
        axs[e, 1].set_xticks([0, 1, 2])
        axs[e, 1].set_xticklabels(["Target", "Competitor", "Decoy"])

        ## Annotation
        best_summary = pd.read_csv(
            join(
                RESULTS_DIR,
                "1-behaviour",
                "cs_target_vs_competitor_{effect}_BEST_summary.csv".format(
                    effect=effect
                ),
            ),
            index_col=0,
        )
        axs[(e, 1)] = addEffectBar(
            text="$d$ = {mean_d:.2f}\n[{hdi_lower:.2f}, {hdi_upper:.2f}]".format(
                mean_d=best_summary.loc["d", "mean"],
                hdi_lower=best_summary.loc["d", "hdi_2.5%"],
                hdi_upper=best_summary.loc["d", "hdi_97.5%"],
            ),
            x0=0,
            x1=1,
            y=0.9,
            ax=axs[(e, 1)],
            color="black",
        )

        # Violin: Gaze
        axs[e, 2] = violin(
            data=rdwells.loc[
                rdwells["effect"] == effect,
                ["dwell_target", "dwell_competitor", "dwell_decoy"],
            ],
            palette=PALETTE,
            ax=axs[e, 2],
        )
        axs[e, 2].set_ylim(0.1, 0.5)
        axs[e, 2].set_yticks(np.arange(0.1, 0.51, 0.1))
        axs[e, 2].set_ylabel("Relative dwell")
        axs[e, 2].set_xlabel(None)
        axs[e, 2].set_xticks([0, 1, 2])
        axs[e, 2].set_xticklabels(["Target", "Competitor", "Decoy"])

        ## Annotation
        best_summary = pd.read_csv(
            join(
                RESULTS_DIR,
                "2-gaze",
                "dwell_target_vs_competitor_{effect}_BEST_summary.csv".format(
                    effect=effect
                ),
            ),
            index_col=0,
        )
        axs[(e, 2)] = addEffectBar(
            text="$d$ = {mean_d:.2f}\n[{hdi_lower:.2f}, {hdi_upper:.2f}]".format(
                mean_d=best_summary.loc["d", "mean"],
                hdi_lower=best_summary.loc["d", "hdi_2.5%"],
                hdi_upper=best_summary.loc["d", "hdi_97.5%"],
            ),
            x0=0,
            x1=1,
            y=0.45,
            ax=axs[(e, 2)],
            color="black",
            lineTextGap=0.01,
        )

        # Break y-axis in dwell plots
        ## get original y-ticklabels
        yticklabels = axs[e, 2].get_yticks().round(2)
        ## replace first entry with zero
        yticklabels[0] = 0
        ## re-set yticks
        axs[e, 2].set_yticklabels(yticklabels)

        boxheight = 0.05
        boxwidth = 0.05
        whitebox = matplotlib.patches.Rectangle(
            (0 - boxwidth / 2, 0.125 - boxheight / 2),
            width=boxwidth,
            height=boxheight,
            color="white",
            transform=axs[e, 2].transAxes,
            clip_on=False,
            zorder=9,
        )
        axs[e, 2].add_patch(whitebox)

        ## Breaker lines
        linewidth = 0.05
        lineheight = 0.05
        for ypos in [0.0875, 0.0875 + boxheight]:
            axs[e, 2].annotate(
                "",
                xy=(-0.01 - linewidth / 2, ypos - lineheight / 2),
                xytext=(0 - 0.01 + linewidth, ypos + lineheight),
                xycoords="axes fraction",
                arrowprops=dict(color="black", arrowstyle="-", capstyle="butt"),
                zorder=99,
                clip_on=False,
            )

    # Label panels
    for ax, label in zip(axs.ravel(), list("abcdef")):
        # set a different x position for the ternary plots, because they have no spines
        if label in ["a", "d"]:
            xshift = 0.1
        else:
            xshift = -0.25
        ax.text(
            xshift,
            1.025,
            label,
            size=10,
            fontweight="bold",
            va="top",
            ha="right",
            transform=ax.transAxes,
        )

    fig.tight_layout(h_pad=4, w_pad=2)
    plt.savefig(join(OUTPUT_DIR, "2-context-effects.pdf"))
