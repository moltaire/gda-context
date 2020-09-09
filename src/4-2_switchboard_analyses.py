#!/usr/bin/env python
# coding: utf-8

from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import DivergingNorm
from matplotlib.patches import Rectangle
from tqdm import tqdm_notebook as tqdm

from plotting.plot_share import factorial_heatmap
from plotting.plot_utils import cm2inch, set_mpl_defaults, break_after_nth_tick

matplotlib = set_mpl_defaults(matplotlib)
matplotlib.rcParams["font.size"] = 6

RESULTS_DIR = join("..", "results", "4-switchboard")
OUTPUT_DIR = RESULTS_DIR


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


sb_estimates = pd.read_csv(join(RESULTS_DIR, "estimates", "sb_estimates_de1.csv"))
sb_estimates.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)

# decode switches from model names
sb_estimates["integration"] = sb_estimates.model.str.split("_").str[1].str[4:]
sb_estimates["comparison"] = sb_estimates.model.str.split("_").str[2].str[5:]
sb_estimates["gb_att"] = sb_estimates.model.str.split("_").str[3].str[6:] == "true"
sb_estimates["gb_alt"] = sb_estimates.model.str.split("_").str[4].str[6:] == "true"
sb_estimates["leak"] = sb_estimates.model.str.split("_").str[5].str[3:]
sb_estimates["inhibition"] = sb_estimates.model.str.split("_").str[6].str[4:]

# set sub-random-fits to random:
## identify number of parameters
sb_estimates["n_params"] = (-2 * sb_estimates["nll"] + sb_estimates["bic"]) / np.log(
    225
)
random_nll = -225 * np.log(1 / 3)
sb_estimates["converged"] = ~(sb_estimates["nll"] > random_nll)
sb_estimates["non-converged"] = ~sb_estimates["converged"]
n_nonconverged = np.sum(sb_estimates["non-converged"])
sb_estimates.loc[sb_estimates["nll"] > random_nll, "beta"] = 0
sb_estimates.loc[
    sb_estimates["nll"] > random_nll, "bic"
] = 2 * random_nll + sb_estimates["n_params"] * np.log(225)
sb_estimates.loc[sb_estimates["nll"] > random_nll, "nll"] = random_nll

# The analysis comprised a total of 192 model variants, fitted to the data of each participant.
# Some of the variants should result in identical fits, because variants with constant (or no) leak or inhibition between alternatives are mathematically equivalent for `comparison` switch levels `absolute` and `vs-mean`. This is due to the fact that a softmax choice rule with a freely estimated inverse temperature parameter is applied over final accumulator values for all variants. Due to the linearity of models with constant (or no) leak or inhibition, the `vs-mean` transformation can be fully compensated by the softmax function, using a slightly transformed inverse temperature parameter.

# compute how many runs of the remaining 160 variants failed to converge
# remove duplicate models with comparative accumulation
# because they are mathematically equivalent to independent accumulation models
# when no or constant leak and inhibition are used
sb_estimates_noduplicates = sb_estimates.loc[
    ~(
        (sb_estimates["comparison"] == "vsmean")
        & (sb_estimates["leak"].isin(["none", "free"]))
        & (sb_estimates["inhibition"].isin(["none", "free"]))
    )
]

n_nonconverged = sb_estimates_noduplicates["non-converged"].sum()
print(
    "{} ({:.2f}%) estimation runs did not achieve performance of random model, indicating non-convergence.".format(
        n_nonconverged, 100 * n_nonconverged / len(sb_estimates_noduplicates)
    )
)

print("Non-converged runs are primarily using distance-dependent inhibition:")
print(sb_estimates_noduplicates.groupby(["inhibition"])["non-converged"].sum())


# Compute mean BICs for each model over participants
mean_bics = sb_estimates.groupby("model")["bic"].mean().reset_index()

# Decode switches from model names
mean_bics["integration"] = mean_bics.model.str.split("_").str[1].str[4:]
mean_bics["comparison"] = mean_bics.model.str.split("_").str[2].str[5:]
mean_bics["gb_att"] = mean_bics.model.str.split("_").str[3].str[6:] == "true"
mean_bics["gb_alt"] = mean_bics.model.str.split("_").str[4].str[6:] == "true"
mean_bics["leak"] = mean_bics.model.str.split("_").str[5].str[3:]
mean_bics["inhibition"] = mean_bics.model.str.split("_").str[6].str[4:]

# Recode switches to ordered Categoricals so I can control their order
mean_bics["integration"] = pd.Categorical(
    mean_bics["integration"], categories=["additive", "multiplicative"], ordered=True
).rename_categories([r"$+$", r"$\times$"])
mean_bics["comparison"] = pd.Categorical(
    mean_bics["comparison"], categories=["absolute", "vsmean", "n.d."], ordered=True
).rename_categories(["independent", "comparative", "n.d."])
mean_bics["gb_att"] = pd.Categorical(
    mean_bics["gb_att"], categories=[False, True], ordered=True
)
mean_bics["gb_alt"] = pd.Categorical(
    mean_bics["gb_alt"], categories=[False, True], ordered=True
)
mean_bics["leak"] = pd.Categorical(
    mean_bics["leak"], categories=["none", "free", "gaze-dependent"], ordered=True
).rename_categories(["None", "Constant", "Gaze"])
mean_bics["inhibition"] = pd.Categorical(
    mean_bics["inhibition"],
    categories=["none", "free", "distance-dependent", "gaze-dependent"],
    ordered=True,
).rename_categories(["None", "Constant", "Distance", "Gaze"])
mean_bics.rename({"bic": "BIC"}, axis=1, inplace=True)

# Factorial heatmap plot
fig, ax = plt.subplots(figsize=cm2inch(12, 5), dpi=300)
ax, values = factorial_heatmap(
    mean_bics,
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
## MAKE SURE TO SAVE OR SHOW THIS

# mark duplicate models with comparative accumulation
# because they are mathematically equivalent to independent accumulation models
# when no or constant leak and inhibition are used
mean_bics.loc[
    (
        (mean_bics["comparison"] == "comparative")
        & (mean_bics["leak"].isin(["Constant", "None"]))
        & (mean_bics["inhibition"].isin(["Constant", "None"]))
    ),
    "BIC",
] = np.nan

mean_bics.loc[
    (
        (mean_bics["leak"].isin(["Constant", "None"]))
        & (mean_bics["inhibition"].isin(["Constant", "None"]))
    ),
    "comparison",
] = "n.d."

print(f"{pd.isnull(mean_bics['BIC']).sum()} were marked as duplicate.")


# Identify overall best fitting model (lowest mean BIC)
top10_variants_pooled_bic = (
    mean_bics.sort_values("BIC").head(10).drop("model", axis=1).reset_index(drop=True)
)
print("Top 10 variants (based on mean BIC across participants)")
print(top10_variants_pooled_bic)
top10_variants_pooled_bic.to_csv(join(OUTPUT_DIR, "top10_variants_pooled_bic.csv"))

# Single best variant:
top1_variant_pooled_estimate_summary = mean_bics.sort_values("BIC")["model"].values[0]

# Investigate winning model parameters
print("Best variant (average across participants) parameter estimates:")
top1_variant_pooled_estimate_summary = sb_estimates.loc[
    sb_estimates["model"] == top1_variant_pooled_estimate_summary,
    ["alpha", "beta", "lam", "theta"],
].describe()
print(top1_variant_pooled_estimate_summary)
top1_variant_pooled_estimate_summary.to_csv(
    join(OUTPUT_DIR, "top1_variant_pooled_estimate-summary.csv")
)


# # Switch levels Mean BIC
#
# Next, we investigate model performance at the level of individual mechanism switches. This analysis aims to quantify the relative contribution to average model fit of each mechanism and specific implementation.

# mark where independent and comparative accumulation cannot be distinguished
sb_estimates_noduplicates.loc[
    (sb_estimates_noduplicates["leak"].isin(["none", "free"]))
    & (sb_estimates_noduplicates["inhibition"].isin(["none", "free"])),
    "comparison",
] = "n.d."

# identify best switch levels
switch_levels_bic = []
for switch in switches.keys():
    switch_mean_bics = sb_estimates_noduplicates.groupby(switch)["bic"].agg(
        ["mean", "sem"]
    )
    df = pd.DataFrame(switch_mean_bics)
    df.index.name = "level"
    df["switch"] = switch
    switch_levels_bic.append(df.reset_index())

switch_levels_bic = pd.concat(switch_levels_bic)[
    ["switch", "level", "mean", "sem"]
].reset_index(drop=True)
level_labels = {
    "comparison": {
        "absolute": "Independent",
        "vsmean": "Comparative",
        "n.d.": r"$n.d.$",
    },
    "integration": {"additive": r"$+$", "multiplicative": r"$\times$"},
    "gb_alt": {True: "with", False: "without"},
    "gb_att": {True: "with", False: "without"},
    "leak": {"none": "None", "free": "Constant", "gaze-dependent": "Gaze"},
    "inhibition": {
        "none": "None",
        "free": "Constant",
        "distance-dependent": "Distance",
        "gaze-dependent": "Gaze",
    },
}

switch_levels_bic["label"] = switch_levels_bic.apply(
    lambda x: level_labels[x["switch"]][x["level"]], axis=1
)
switch_levels_bic.to_csv(join(OUTPUT_DIR, "switch-levels_bic.csv"))


# Switch-level average BIC plot
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

# # Individual best fitting models
# Next, we investigate which variants provide the best fit on the level of individual participants:


# identify individually best fitting models
ind_best_variants = sb_estimates_noduplicates.loc[
    sb_estimates_noduplicates.groupby("subject")["bic"].idxmin()
][
    [
        "subject",
        "model",
        "bic",
        "integration",
        "comparison",
        "gb_alt",
        "gb_att",
        "leak",
        "inhibition",
    ]
]
ind_best_variants.drop("model", axis=1).to_csv(
    join(OUTPUT_DIR, "individual_best-variants_bic.csv")
)


top10_variants_individual_counts = (
    ind_best_variants["model"]
    .value_counts()
    .reset_index()
    .rename({"index": "model", "model": "count"}, axis=1)
)

# decode switches from model names
top10_variants_individual_counts["integration"] = (
    top10_variants_individual_counts.model.str.split("_").str[1].str[4:]
)
top10_variants_individual_counts["comparison"] = (
    top10_variants_individual_counts.model.str.split("_").str[2].str[5:]
)
top10_variants_individual_counts["gb_att"] = (
    top10_variants_individual_counts.model.str.split("_").str[3].str[6:] == "true"
)
top10_variants_individual_counts["gb_alt"] = (
    top10_variants_individual_counts.model.str.split("_").str[4].str[6:] == "true"
)
top10_variants_individual_counts["leak"] = (
    top10_variants_individual_counts.model.str.split("_").str[5].str[3:]
)
top10_variants_individual_counts["inhibition"] = (
    top10_variants_individual_counts.model.str.split("_").str[6].str[4:]
)
top10_variants_individual_counts = top10_variants_individual_counts.drop(
    "model", axis=1
)
top10_variants_individual_counts.to_csv(
    join(OUTPUT_DIR, "top10_variants_individual-counts.csv")
)
print(top10_variants_individual_counts)

# Count individually best fitting switches

best_switches_individual_counts = []
for switch in switches.keys():
    best_counts = ind_best_variants[switch].value_counts()
    best_counts.name = "count"
    df = pd.DataFrame(best_counts)
    df.index.name = "level"
    df["switch"] = switch
    best_switches_individual_counts.append(df.reset_index())

best_switches_individual_counts = pd.concat(best_switches_individual_counts)[
    ["switch", "level", "count"]
].reset_index(drop=True)
best_switches_individual_counts["label"] = best_switches_individual_counts.apply(
    lambda x: level_labels[x["switch"]][x["level"]], axis=1
)
print(best_switches_individual_counts)
best_switches_individual_counts.to_csv(
    join(RESULTS_DIR, "switch-levels_individual-counts.csv")
)


# Make a figure
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

