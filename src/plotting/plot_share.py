# /usr/bin/python
"""
Plotting functions shared across plots.
"""
from itertools import groupby

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from matplotlib.colors import colorConverter
from seaborn import violinplot

from .plot_utils import *


def scatter(
    x,
    y,
    color=None,
    facealpha=0.8,
    edgealpha=1,
    size=4,
    edgewidth=0.5,
    ax=None,
    **kwargs,
):
    """Make a custom scatterplot, with solid outlines and translucent faces.

    Args:
        x (array like): x values
        y (array like): y values
        color (optional): color to use for scatter faces. Defaults to default color.
        ax (matplotlib.axis, optional): Axis to plot on. Defaults to None.
        kwargs: Keyword arguments passed on to matplotlib.pyplot.plot

    Returns:
        matplotlib.axis: Axis with the violinplot.
    """
    if ax is None:
        ax = plt.gca()

    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]

    # Solid outlines and translucent faces
    scatterArtists = ax.scatter(
        x,
        y,
        color=colorConverter.to_rgba_array(color, alpha=facealpha),
        linewidth=edgewidth,
        s=size ** 2,
        edgecolor=colorConverter.to_rgba("k", alpha=edgealpha),
        clip_on=False,
        zorder=5,
        **kwargs,
    )

    return ax


def violin(
    data,
    violin_width=0.8,
    box_width=0.1,
    xlabel=None,
    xticklabels=None,
    ylabel=None,
    palette=None,
    ax=None,
):
    """Make a custom violinplot, with nice inner boxplot.

    Args:
        data (pandas.DataFrame): Data to plot. Each column will be made into one violin.
        violin_width (float, optional): Width of the violins. Defaults tmo 0.8.
        box_width (float, optional): Width of the boxplot. Defaults to 0.1.
        xlabel (str, optional): x-axis label. Defaults to None.
        xticklabels (list, optional): x-tick labels. Defaults to None.
        ylabel (str, optional): y-axis label. Defaults to None.
        palette (list, optional): list of colors to use for violins. Defaults to default colors.
        ax (matplotlib.axis, optional): Axis to plot on. Defaults to None.

    Returns:
        matplotlib.axis: Axis with the violinplot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=cm2inch(4.5, 4.5))

    # transform data into long format for seaborn violinplot
    data_long = pd.melt(data)

    # Violinplot
    violinplot(
        x="variable",
        y="value",
        data=data_long,
        palette=palette,
        linewidth=0,
        inner=None,
        scale="width",
        width=violin_width,
        saturation=1,
        ax=ax,
    )

    # Boxplot
    # Matplotlib boxplot uses a different data format (list of arrays)
    # Matplotlib boxplot also cannot deal with NaN values, so these must be dropped.
    boxplot_data = [
        data[var].values[~pd.isnull(data[var].values)] for var in data.columns
    ]

    bplot = ax.boxplot(
        boxplot_data,
        positions=range(len(boxplot_data)),
        widths=box_width,
        showcaps=False,
        boxprops=dict(linewidth=0.5),
        medianprops=dict(linewidth=0.5, color="black"),
        whiskerprops=dict(linewidth=0.5),
        flierprops=dict(
            marker="o",
            markersize=2,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.25,
            alpha=0.9,
        ),
        patch_artist=True,
    )
    for patch in bplot["boxes"]:
        patch.set_facecolor("white")

    # Labels, etc.
    ax.set_xlim(-0.5, len(data.columns) + -0.5)

    return ax


def addEffectBar(text, x0, x1, y, ax, linewidth=0.5, lineTextGap=0.02, fontsize=5):
    """Add a horizontal line and some text. Good for p-values and similar stuff.
    
    Args:
        text (str): Text.
        x0 (float): Line start value.
        x1 (float): Line end value.
        y (float): Height of the line.
        ax (matplotlib.axis): Axis to annotate
        linewidth (float, optional): Linewidth. Defaults to 0.5.
        lineTextGap (float, optional): Distance between the line and the text. Defaults to 0.02.
        fontsize (int, optional): Fontsize. Defaults to 5.
    
    Returns:
        matplotlib.axis: Annotated axis.
    """
    ax.hlines(y, x0, x1, linewidth=linewidth, clip_on=False)
    ax.text(
        x=(x0 + x1) / 2,
        y=y + lineTextGap,
        s=text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )
    return ax


def lm(
    x,
    y,
    trace=None,
    credible_interval=0.95,
    ax=None,
    bandalpha=0.6,
    scatter_kws={},
    scatter_color=None,
    line_color=None,
    xrange=None,
    sample_kwargs={"cores": 1},
    **kwargs,
):
    """Make a custom linear model plot with confidence bands.

    Args:
        x (array like): x values
        y (array like): y values
        trace (pymc3.MultiTrace, optional): GLM trace from PyMC3.
        ax (matplotlib.axis, optional): Axis to plot on. Defaults to current axis.
        bandalpha (float, optional): Opacity level of confidence band.
        scatter_kws (dict, optional): Dictionary of keyword arguments passed onto `scatter`.
        **kwargs: Keyword arguments passed onto plot of regression line.

    Returns:
        tuple
            matplotlib.axis: Axis with the linear model plot.
            pymc3.trace object: The linear model trace
            pandas.DataFrame: The linear model pymc3.summary
    """
    if ax is None:
        ax = plt.gca()

    # Determine color (this is necessary so that the scatter and the line have the same color)
    if scatter_color is None:
        scatter_color = next(ax._get_lines.prop_cycler)["color"]
    if line_color is None:
        line_color = next(ax._get_lines.prop_cycler)["color"]

    # Scatter
    ax = scatter(x, y, color=scatter_color, ax=ax, **scatter_kws)

    # Run GLM in PyMC3
    if trace is None:
        df = pd.DataFrame(dict(x=x, y=y))
        with pm.Model() as glm:
            pm.GLM.from_formula("y ~ x", data=df)
            trace = pm.sample(**sample_kwargs)

    summary = pm.summary(trace, hdi_prob=credible_interval)

    # Plot MAP regression line
    if xrange is None:
        xs = np.linspace(np.min(x), np.max(x), 100)
    else:
        xs = np.linspace(*xrange, 100)
    intercept = summary.loc["Intercept", "mean"]
    beta = summary.loc["x", "mean"]
    ax.plot(xs, intercept + beta * xs, color=line_color, zorder=4, **kwargs)

    # Plot posterior predictive credible region band
    intercept_samples = trace.get_values("Intercept")
    beta_samples = trace.get_values("x")
    ypred = intercept_samples + beta_samples * xs[:, None]
    ypred_lower = np.quantile(ypred, (1 - credible_interval) / 2, axis=1)
    ypred_upper = np.quantile(ypred, 1 - (1 - credible_interval) / 2, axis=1)
    ax.fill_between(
        xs,
        ypred_lower,
        ypred_upper,
        color=line_color,
        zorder=1,
        alpha=bandalpha,
        linewidth=0,
    )

    return ax, trace, summary


def factorial_heatmap(
    df,
    row_factors,
    col_factors,
    value_var,
    factor_labels={},
    level_labels={},
    cmap="viridis_r",
    norm=None,
    ax=None,
    ylabel_rotation=0,
    xlabel_rotation=0,
    pad_label_bar=0.2,
    pad_per_factor=1.5,
    pad_colorbar=0.05,
    cb_args={},
):
    """Make a factorial heatmap.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing categorical factors and numerical value variable
    row_factors : list
        List of factors determining heatmap rows
    col_factors : list
        List of factors determining heatmap columns
    value_var : str
        Name of the value variable
    factor_labels : dict, optional
        Dictionary containing mappings from variable names in the DataFrame to displayed variable names, by default {}
    level_labels : dict, optional
        Dictionary containing dictionaries for each factor, containing mappings from level names to displayed level names, by default {}
    cmap : str, optional
        cmap argument passed on to matplotlib.pyplot.imshow, by default "viridis_r". But try "inferno", "magma", ...
    ax : matplotlib.axis, optional
        Axis to plot on, by default None
    Returns
    -------
    matplotlib.axis
        axis with plot
    """
    all_factors = row_factors + col_factors
    default_factor_labels = {factor: factor for factor in all_factors}
    factor_labels = {**default_factor_labels, **factor_labels}
    default_level_labels = {
        factor_labels[factor]: {
            level: f"{factor_labels[factor]}={level}" for level in df[factor].unique()
        }
        for factor in all_factors
    }
    level_labels = {**default_level_labels, **level_labels}

    if ax is None:
        ax = plt.gca()

    n_row = np.prod([df[row_factor].unique().size for row_factor in row_factors])
    n_col = np.prod([df[col_factor].unique().size for col_factor in col_factors])

    df_sorted = df.sort_values(row_factors + col_factors)
    values = df_sorted[value_var].values.reshape(n_row, n_col)

    # Make the heatmap
    im = ax.imshow(values, cmap=cmap, norm=norm)

    # x_labels = levels from last col_factor
    ax.set_xlabel(factor_labels[col_factors[-1]])
    ax.set_xticks(np.arange(n_col))
    ax.set_xticklabels(df_sorted[col_factors[-1]], rotation=xlabel_rotation)
    ax.set_xlim(-0.5, n_col - 0.5)

    # other factors across columns:
    # from second-to-last to first, so that the first factor is the uppermost level
    for f, col_factor in enumerate(col_factors[-2::-1]):
        levels = df_sorted[col_factor].values[:n_col]
        bar_y = n_row - 0.25 + f * pad_per_factor

        # Identify blocks of same levels: https://stackoverflow.com/a/6352456
        index = 0
        for level, block in groupby(levels):
            length = sum(1 for i in block)
            bar_xmin = index
            bar_xmax = index + length - 1
            index += length
            ax.plot(
                [bar_xmin - 0.4, bar_xmax + 0.4],
                [bar_y, bar_y],
                linewidth=0.75,
                color="k",
                clip_on=False,
            )
            ax.annotate(
                level_labels[factor_labels[col_factor]][level],
                xy=(bar_xmin + (bar_xmax - bar_xmin) / 2, bar_y + pad_label_bar),
                xycoords="data",
                ha="center",
                va="bottom",
                ma="center",
                annotation_clip=False,
            )

    # y_labels = levels from last row_factor
    ax.set_ylabel(factor_labels[row_factors[-1]])
    ax.set_yticks(np.arange(n_row))
    ax.set_yticklabels(df_sorted[row_factors[-1]][::n_col], rotation=ylabel_rotation)
    ax.set_ylim(-0.5, n_row - 0.5)

    # other factors across rows:
    # from second-to-last to first, so that the first factor is the uppermost level
    for f, row_factor in enumerate(row_factors[-2::-1]):
        levels = df_sorted[row_factor].values[::n_col][:n_row]
        bar_x = n_col - 0.25 + f * pad_per_factor

        index = 0
        for level, block in groupby(levels):
            length = sum(1 for i in block)
            bar_ymin = index
            bar_ymax = index + length - 1
            index += length
            ax.plot(
                [bar_x, bar_x],
                [bar_ymin - 0.4, bar_ymax + 0.4],
                linewidth=0.75,
                color="k",
                clip_on=False,
            )
            ax.annotate(
                level_labels[factor_labels[row_factor]][level],
                xy=(bar_x + pad_label_bar, bar_ymin + (bar_ymax - bar_ymin) / 2),
                xycoords="data",
                rotation=270,
                ha="left",
                va="center",
                ma="center",
                annotation_clip=False,
            )

    # colorbar legend
    cb = plt.colorbar(im, pad=len(row_factors) * pad_colorbar, ax=ax, **cb_args)
    cb.ax.set_title(value_var)
    cb.outline.set_linewidth(0.75)

    return ax, values


def break_after_nth_tick(ax, n, axis="x", occHeight=None, occWidth=None, where=0.5):
    """Visually break an axis x or y spine after the nth tick.
    Places a white occluding box and black diagonals onto the axis.
    Axis ticklabels must be changed manually.
    Parameters
    ----------
    ax : matplotlib.axis
        Axis object to plot on
    n : int
        Index of tick after which the break should be made
    axis : str, optional
        must be "x" or "y", by default "x"
    occHeight : float, optional
        Height of the occluding box, by default a third of the space between ticks
    occWidth : float, optional
        Width of the occluding box, by default a third of the space between ticks
    where : float, optional
        Fine tuning of occluder position between ticks, by default 0.5 (right in the middle)
    
    Returns
    -------
    matplotlib.axis
        Axis object with occluder
    Raises
    ------
    ValueError
        If axis keyword not in ['x', 'y']
    """
    # Save current x and y limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Determine occluder position
    if axis == "x":
        occPos = (
            ax.get_xticks()[n] + where * (ax.get_xticks()[n + 1] - ax.get_xticks()[n]),
            ylim[0],
        )
        if occHeight is None:
            occHeight = 1 / 10 * (ax.get_yticks()[n + 1] - ax.get_yticks()[n])
        if occWidth is None:
            occWidth = 1 / 3 * (ax.get_xticks()[n + 1] - ax.get_xticks()[n])
    elif axis == "y":
        occPos = (
            xlim[0],
            ax.get_yticks()[n] + where * (ax.get_yticks()[n + 1] - ax.get_yticks()[n]),
        )
        if occHeight is None:
            occHeight = 1 / 3 * (ax.get_yticks()[n + 1] - ax.get_yticks()[n])
        if occWidth is None:
            occWidth = 1 / 10 * (ax.get_xticks()[n + 1] - ax.get_xticks()[n])
    else:
        raise ValueError(f"'axis' must be 'x' or 'y' (is {axis})")

    # Build occlusion rectangles
    occBox = matplotlib.patches.Rectangle(
        (occPos[0] - occWidth / 2, occPos[1] - occHeight / 2),
        width=occWidth,
        height=occHeight,
        color="white",
        clip_on=False,
        zorder=8,
    )
    ax.add_patch(occBox)

    # Breaker lines
    if axis == "x":
        ax.scatter(
            x=[occPos[0] - occWidth / 2, occPos[0] + occWidth / 2],
            y=[ylim[0], ylim[0]],
            marker=(2, 0, -45),
            color="black",
            s=18,
            linewidth=0.75,
            clip_on=False,
            zorder=9,
        )
    elif axis == "y":
        ax.scatter(
            x=[xlim[0], xlim[0]],
            y=[occPos[1] - occHeight / 2, occPos[1] + occHeight / 2],
            marker=(2, 0, -45),
            color="black",
            s=18,
            linewidth=0.75,
            clip_on=False,
            zorder=9,
        )

    # Restore x and y limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax
