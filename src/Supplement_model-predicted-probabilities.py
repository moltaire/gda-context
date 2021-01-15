from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.plot_utils import cm2inch, set_mpl_defaults
from plotting.plot_share import violin
from analysis.utilities import makeDirIfNeeded

matplotlib = set_mpl_defaults(matplotlib)


def plot_predicted_choiceprobs(predictions, models, model_labels, axs=None):
    """Plot model-predicted choice probabilities for target, competitor, decoy and chosen options for each trial type.

    Args:
        predictions (pandas.Dataframe): DataFrame containing model predictions. Must include columns `model`, `subject`, `trial`, `effect`, `target`, `choice`, `predicted_choice`, `predicted_choiceprob_A`, `predicted_choiceprob_B`, `predicted_choiceprob_C`
        models (list): List of models to include. Must be the same values as in the `model` column in the prediction DataFrame.
        model_labels (dict): Dictionary mapping model names to labels used in the figure.
        axs (numpy.ndarray, optional): Array (len(models) x 2) of matplotlib.axes to plot on. Defaults to None, which creates new axes.

    Returns:
        [numpy.ndarray]: Array containing matplotlib.axes with plots on them.
    """

    # Prepare DataFrame with model-predicted choice probabilities
    df = (
        predictions.loc[
            (
                predictions["rep"] == 0
            ),  # we use predicted choice probability variables, which are identical over trial repetitions
            [
                "model",
                "subject",
                "trial",
                "effect",
                "target",
                "choice",
                "predicted_choice",
                "predicted_choiceprob_A",
                "predicted_choiceprob_B",
                "predicted_choiceprob_C",
            ],
        ]
        .reset_index(drop=True)
        .rename(
            {
                "predicted_choiceprob_A": "pcp_0",
                "predicted_choiceprob_B": "pcp_1",
                "predicted_choiceprob_C": "pcp_2",
            },
            axis=1,
        )
    )

    # Add: Model predicted probability of choosing target option
    df["pcp_target"] = np.where(
        pd.isnull(df["target"]),
        np.nan,
        np.where(df["target"] == "A", df["pcp_0"], df["pcp_1"]),
    )

    # Add: Model predicted probability of choosing competitor option
    df["pcp_competitor"] = np.where(
        pd.isnull(df["target"]),
        np.nan,
        np.where(df["target"] == "A", df["pcp_1"], df["pcp_0"]),
    )

    # Add: Model predicted probability of choosing decoy option (that's a little easier. Decoy is always third option)
    df["pcp_decoy"] = df["pcp_2"]

    # Add: Model predicted probability of choice of empirically chosen option
    df["pcp_chosen"] = np.where(
        df["choice"] == 0,
        df["pcp_0"],
        np.where(df["choice"] == 1, df["pcp_1"], df["pcp_2"]),
    )
    df["log_pcp_chosen"] = np.log(df["pcp_chosen"])

    # Compute model predicted choice probabilities for target, competitor, decoy and chosen alternative
    pcps = (
        df.loc[df["effect"].isin(["attraction", "compromise"])]
        .groupby(["effect", "model", "subject"])[
            ["pcp_target", "pcp_competitor", "pcp_decoy", "pcp_chosen"]
        ]
        .mean()
    )

    # Prepare figure
    if axs is None:
        fig, axs = plt.subplots(
            2,
            len(models),
            figsize=cm2inch(18, 22 * 2 / len(models)),
            sharex="col",
            sharey="row",
        )

    # Cycle over models
    for m, model in enumerate(models):
        # And trial types
        for e, effect in enumerate(["attraction", "compromise"]):

            # Make a violin predicted choice probabilities
            ax = violin(
                pcps.loc[effect, model],
                palette=["turquoise", "violet", "gold"] + ["darkgray"],
                ax=axs[e, m],
                box_width=0.2,
            )

            # Add dashed line for random prediction
            ax.axhline(
                (1 / 3),
                color="lightgray",
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )

            # Adjust labels and titles
            ax.set_xticks(np.arange(4))
            ax.set_xticklabels(["Target", "Competitor", "Decoy", "Chosen"], rotation=90)
            ax.set_xlabel(None)

            if m == 0:
                ax.set_ylabel(f"{effect.capitalize()}\n\npred. P(choice)")
            else:
                ax.set_ylabel(None)
            if e == 0:
                ax.set_title(f"{model_labels[model]}")

            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.01, 0.25))

    plt.tight_layout(w_pad=3, h_pad=3)
    return axs


if __name__ == "__main__":

    # Directories
    RESULTS_DIR = join("..", "results")
    OUTPUT_DIR = join("..", "figures")
    makeDirIfNeeded(OUTPUT_DIR)

    # Which models to include, and how to name them in the plots
    models = [
        "glickman1layer",
        "mdft",
        "gaze-baseline-dyn",
        "gaze-baseline-stat",
        "eu",
        "sb_int-multiplicative_comp-vsmean_gbatt-false_gbalt-true_lk-free_inh-distance-dependent",
    ]
    model_labels = {
        "glickman1layer": "GLA",
        "mdft": "MDFT",
        "eu": "EU",
        "gaze-baseline-stat": "GB" + r"$_{stat}$",
        "gaze-baseline-dyn": "GB" + r"$_{dyn}$",
        "sb_int-multiplicative_comp-vsmean_gbatt-false_gbalt-true_lk-free_inh-distance-dependent": "Hybrid",
    }

    #% Load data
    # Load prediction dataframe (includes predictions of all models, except hybrid, which originates from switchboard analysis)
    predictions = pd.read_csv(
        join(
            RESULTS_DIR, "3-behavioural-modeling", "predictions", "predictions_de1.csv"
        )
    )

    # add predictions of hybrid switchboard model
    predictions_sb = pd.read_csv(
        join(RESULTS_DIR, "4-switchboard", "predictions", "sb_predictions_de1.csv")
    )
    predictions_hybrid = predictions_sb.loc[predictions_sb["model"] == models[-1]]
    predictions = pd.concat([predictions, predictions_hybrid])

    # Make the figure
    axs = plot_predicted_choiceprobs(predictions, models, model_labels)

    # Label panels
    for ax, label in zip(axs.ravel(), list("abcdefghijklmnopqrst")):

        # set a different x position for the first plots in each row, because they have y-ticklabels
        if label in ["a", "g"]:
            xshift = -0.6
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

    # Save figure
    plt.savefig(
        join(OUTPUT_DIR, f"S_model-predicted-choiceprobs.pdf"), bbox_inches="tight"
    )
