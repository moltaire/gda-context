from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.plot_utils import cm2inch, set_mpl_defaults

matplotlib = set_mpl_defaults(matplotlib)


def plotPredictorBinned(
    aoidata,
    predictor_info,
    dependent="dur_total",
    linecolors=None,
    plotLegend=True,
    fontsizeLegend=5,
    ax=None,
):
    """This function plots the data of for one predictor over time.

    Args:
        aoidata (pandas.DataFrame): The timebinned AOI DataFrame.
        predictor_info (dict): Dictionary of predictor information, with str key for "label" and dict key for "map", mapping labels to values.
        dependent (str, optional): Dependent variable, one of 'dwell', 'p-fixate' or 'count'. Defaults to "dwell".
        linecolors ([type], optional): [description]. Defaults to None.
        plotLegend (bool, optional): [description]. Defaults to True.
        fontsizeLegend (int, optional): [description]. Defaults to 5.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        matplotlib.axis: Axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=cm2inch(3, 3), dpi=300)

    if predictor_info.get("map", None) is None:
        predictor_info["map"] = {
            level: level for level in aoidata[predictor_info["varname"]].unique()
        }

    if linecolors is None:
        linecolors = ["C{}".format(i) for i in range(len(predictor_info["map"]))]

    # Iterate over levels of the predictor variable
    for l, level in enumerate(predictor_info["map"].keys()):

        # Compute mean and SEM for each time bin
        summary = (
            aoidata.loc[aoidata[predictor_info["varname"]] == level]
            .groupby(["subject", "time_bin"])[dependent]
            .mean()
            .reset_index()
            .groupby(["time_bin"])[dependent]
            .agg(["mean", "sem"])
            .reset_index()
        )

        # Plot line and error shading
        ax.plot(
            summary["time_bin"],
            summary["mean"].values,
            color=linecolors[l],
            label=predictor_info["map"][level],
        )
        ax.fill_between(
            summary.index,
            summary["mean"] - summary["sem"],
            summary["mean"] + summary["sem"],
            color=linecolors[l],
            alpha=0.2,
        )

    # Optional: Add a legend
    if plotLegend:
        legend = ax.legend(
            loc="upper center",
            facecolor="none",
            fontsize=fontsizeLegend,
            framealpha=1,
            edgecolor="black",
            handlelength=1,
            handletextpad=0.35,
            borderpad=0.5,
            labelspacing=0.25,
            bbox_to_anchor=(0.5, 1.25),
        )
        legend.get_frame().set_linewidth(0.75)

    # Set ticks and labels
    ax.set_xticks(summary["time_bin"])
    ax.set_xticklabels(summary["time_bin"] + 1)
    ax.set_xlabel("Time bin")

    # Set title
    if predictor_info.get("label", None) is None:
        predictor_info["label"] = predictor_info["varname"].capitalize()
    ax.set_title(predictor_info["label"], y=1.15)

    return ax


def plotAllPredictorsBinned(
    aois_timebinned,
    predictors,
    dependent="dwell",
    dependent_label="Dwell (s)",
    ylim=(0, 0.4),
    width=18,
    height=None,
):
    """
    This function plots a given eye movement data variable for each predictor across time.
    """

    # Define plotting colors
    colors = ["slategray", "darksalmon", "mediumaquamarine"]

    # Use different colors for alternative, to match Figure 2
    tcdcolors = ["turquoise", "violet", "gold"]

    if height is None:
        height = 2.25 * width / len(predictors)

    fig, axs = plt.subplots(
        2,
        len(predictors),
        figsize=cm2inch(width, height),
        sharex=True,
        sharey=True,
        dpi=300,
    )

    for e, effect in enumerate(["attraction", "compromise"]):
        for p, (predictor, predictor_info) in enumerate(predictors.items()):
            if predictor == "alternative":
                linecolors = tcdcolors
            else:
                linecolors = colors
            if e == 0:
                plotLegend = True
            else:
                plotLegend = False
            axs[e, p] = plotPredictorBinned(
                aois_timebinned.loc[aois_timebinned["effect"] == effect],
                predictor_info=predictor_info,
                dependent=dependent,
                linecolors=linecolors,
                plotLegend=plotLegend,
                ax=axs[e, p],
            )
            axs[e, p].set_ylim(*ylim)
            if e == 0:
                axs[e, p].set_xlabel(None)
            if e == 1:
                axs[e, p].set_title(None)
            if p == 0:
                axs[e, p].set_ylabel(f"{effect.capitalize()}\n\n{dependent_label}")

    # Label panels
    for ax, label in zip(axs.ravel(), list("abcdefghijklmnopqrst")):

        # set a different x position for the first plots in each row, because they have y-ticklabels
        if label in ["a", "g"]:
            xshift = -0.35
        else:
            xshift = -0.15
        ax.text(
            xshift,
            1.075,
            label,
            size=8,
            fontweight="bold",
            va="top",
            ha="right",
            transform=ax.transAxes,
        )

    fig.tight_layout(h_pad=0.75, w_pad=0.5)

    return fig


if __name__ == "__main__":

    OUTPUT_DIR = join("..", "figures")

    # Define predictors
    predictors = {
        "row": {"varname": "row", "label": "Row", "map": {0: "top", 1: "bottom"}},
        "col": {
            "varname": "col",
            "label": "Column",
            "map": {0: "left", 1: "center", 2: "right"},
        },
        "is_chosen": {
            "varname": "is_chosen",
            "label": "Choice",
            "map": {0: "not chosen", 1: "chosen"},
        },
        "alternative": {
            "varname": "alternative",
            "label": "Alternative",
            "map": {"target": "Target", "competitor": "Competitor", "decoy": "Decoy"},
        },
        "attribute": {
            "varname": "is_probability",
            "label": "Attribute",
            "map": {0: "Outcome", 1: "Probability"},
        },
        "rank": {
            "varname": "rank",
            "label": "Rank",
            "map": {1: "worst", 2: "middle", 3: "best"},
        },
    }

    # Load timebinned AOI data (made by gaze-regression script)
    aois_timebinned = pd.read_csv(
        join(
            "..",
            "results",
            "S_supplemental-gaze-analyses",
            "dwell-regression_timebinned",
            "data",
            "aois_timebinned.csv",
        )
    )

    # transform dwell duration to seconds
    aois_timebinned["dwell"] = aois_timebinned["dur_total"] / 1000

    # rename this column
    aois_timebinned["p-fixate"] = aois_timebinned["is_fixated"]

    # recode 'alternative' column
    aois_timebinned["alternative"] = np.where(
        aois_timebinned["is_target"].astype(bool),
        "target",
        np.where(aois_timebinned["is_decoy"], "decoy", "competitor"),
    )

    # Make a predictor plot with each predictor, for each dependent variable
    for dependent, dependent_label, ylim in zip(
        ["dwell", "p-fixate", "count"],
        ["Dwell (s)", "P(fixate)", "Fixation count"],
        [(0, 0.4), (0, 1), (0, 1)],
    ):

        fig = plotAllPredictorsBinned(
            aois_timebinned,
            predictors,
            dependent=dependent,
            dependent_label=dependent_label,
            ylim=ylim,
        )

        plt.savefig(
            join(OUTPUT_DIR, f"S_gaze-data_timebinned_{dependent}.pdf"),
            bbox_inches="tight",
        )

