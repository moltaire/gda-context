#!/usr/bin/python
"""
Gaze-dependent accumulation in context-dependent risky choice
    Summarise GLA estimates and make pairwise scatterplots
"""

from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cycler

from plotting.plot_share import lm, scatter
from plotting.plot_utils import *

matplotlib = set_mpl_defaults(matplotlib)

RESULTS_DIR = join("..", "results")
OUTPUT_DIR = join("..", "figures")


matplotlib.rcParams.update(
    {
        "axes.prop_cycle": cycler(
            "color",
            [
                "slategray",
                "darksalmon",
                "mediumaquamarine",
                "indianred",
                "orchid",
                "paleturquoise",
                "tan",
                "lightpink",
            ],
        )
    }
)

# Load GLA parameter estimates
estimates = (
    pd.read_csv(
        join(RESULTS_DIR, "3-behavioural-modeling", "estimates", "estimates_de1.csv")
    )
    .drop("Unnamed: 0", axis=1)
    .reset_index(drop=True)
)
gla_estimates = estimates.loc[
    estimates["model"] == "glickman1layer",
    ["subject", "bic", "nll", "alpha", "beta", "gamma", "lam", "theta"],
].reset_index(drop=True)

gla_estimates_summary = (
    gla_estimates[["alpha", "beta", "gamma", "lam", "theta"]].describe().round(2).T
)
gla_estimates_summary.to_csv(
    join(RESULTS_DIR, "3-behavioural-modeling", "gla_estimates_summary.csv")
)


def pairPlot(df, kind="lm", bins=None, limits=None, labels=None, titles=None):
    """
    Plot associations between all pairs of variables in df.
    
    kind : str
        'lm': Calculates linear models per pair
        'scatter': Only scatter
    limits : dict
        Dictionary of variable limits.
    labels : dict
        Dictionary of variable x- and y-labels
    titles : dict
        Dictionary of variable column titles
    
    Returns:
        fig, axs
    """

    variables = df.columns
    N = len(variables)

    fig, axs = plt.subplots(N, N, figsize=cm2inch(18, 18), dpi=300)

    for vx, varx in enumerate(variables):
        for vy, vary in enumerate(variables):
            ax = axs[vy, vx]
            if varx == vary:
                # Histogram
                if (limits is not None) and (bins is not None):
                    varBins = np.linspace(*limits[varx], bins + 1)
                ax.hist(df[varx], linewidth=0.75, edgecolor="white", bins=varBins)
                ax.set_ylabel("Frequency")

                if limits is not None:
                    ax.set_xlim(limits[varx])
                if labels is not None:
                    ax.set_xlabel(labels[varx])
                else:
                    ax.set_xlabel(varx)
                if titles is not None:
                    ax.set_title(titles[varx])
                else:
                    pass

            elif vy < vx:
                ax.axis("off")
            else:
                if kind == "lm":
                    ax, trace, summary = lm(df[varx], df[vary], ax=ax)
                elif kind == "scatter":
                    ax = scatter(df[varx], df[vary], alpha=1, ax=ax)
                else:
                    raise ValueError('Unknown argument for "kind": {}'.format(kind))
                ax.set_xlabel(varx)
                ax.set_ylabel(vary)

                if limits is not None:
                    ax.set_xlim(limits[varx])
                    ax.set_ylim(limits[vary])
                if labels is not None:
                    ax.set_xlabel(labels[varx])
                    ax.set_ylabel(labels[vary])
                else:
                    ax.set_xlabel(varx)
                    ax.set_ylabel(vary)

    fig.tight_layout()
    return fig, axs


limits = dict(alpha=[0, 2], beta=[0, 50], gamma=[0, 1], lam=[0, 1], theta=[0, 1])
labels = dict(
    alpha=r"$\alpha$",
    beta=r"$\beta$",
    gamma=r"$\gamma$",
    lam=r"$\lambda$",
    theta=r"$\theta$",
)
titles = dict(
    alpha="Utility",
    beta="Inverse temperature\n(0 = random choice)",
    gamma="Probability weighting\n(1 = linear weighting)",
    lam="Leak\n(0 = perfect memory,\n1 = full leak)",
    theta="Gaze-discount\n(1 = no discount,\n0 = full discount)",
)

fig, axs = pairPlot(
    gla_estimates[["alpha", "beta", "gamma", "lam", "theta"]],
    kind="scatter",
    bins=10,
    limits=limits,
    labels=labels,
    titles=titles
)

fig.tight_layout(h_pad=0.1, w_pad=0.1)
plt.savefig(join(OUTPUT_DIR, "S_gla-estimates.pdf"), bbox_inches="tight")
