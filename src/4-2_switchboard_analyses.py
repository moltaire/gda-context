#!/usr/bin/env python
# coding: utf-8
"""
This script performs analysis on the results from the switchboard analysis, performed in `4-1_switchboard_fitting`.

    1) Read and process switchboard estimate data
    2) Compute mean BIC per model variant and save as `variants_mean_bic.csv`
    3) Identify overall best fitting variants (lowest mean BIC) and save as `top10_variants_pooled_bic`
    4) Summarise winning model parameters and save as `top1_variant_pooled_estimate-summary.csv`
    5) Compute Switch-levels Mean BIC and save as `switch-levels_bic.csv`
    6) Count individual best fitting model variants, save list of best variant per participant (`individual_best-variants_bic.csv`) and counts of individually best variants (`top10_variants_individual-counts.csv`)
    7) Count individually best fitting switches, save as `switch-levels_individual-counts.csv`
"""

from os.path import join

import numpy as np
import pandas as pd

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

# %% 1) Read and process the data
# -------------------------

# Read parameter estimates from switchboard analysis
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
## re-calculate number of parameters
sb_estimates["n_params"] = (-2 * sb_estimates["nll"] + sb_estimates["bic"]) / np.log(
    225
)
random_nll = -225 * np.log(1 / 3)  # NLL of random model
sb_estimates["non-converged"] = (
    sb_estimates["nll"] > random_nll
)  # mark all fits worse than the (nested) random model
# Set their NLLs and soft-max parameter estimates manually to a random-model's performance
sb_estimates.loc[sb_estimates["nll"] > random_nll, "beta"] = 0
sb_estimates.loc[
    sb_estimates["nll"] > random_nll, "bic"
] = 2 * random_nll + sb_estimates["n_params"] * np.log(225)
sb_estimates.loc[sb_estimates["nll"] > random_nll, "nll"] = random_nll

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

# Report number of non-converged estimation runs
n_nonconverged = sb_estimates_noduplicates["non-converged"].sum()
print(
    "{} ({:.2f}%) estimation runs did not achieve performance of random model, indicating non-convergence.".format(
        n_nonconverged, 100 * n_nonconverged / len(sb_estimates_noduplicates)
    )
)

print("Non-converged runs are primarily using distance-dependent inhibition:")
print(sb_estimates_noduplicates.groupby(["inhibition"])["non-converged"].sum())



# %% 2) Compute mean BICs for each model over participants
# --------------------------------------------------------
variants_mean_bic = sb_estimates.groupby("model")["bic"].mean().reset_index()

# Decode switches from model names
variants_mean_bic["integration"] = variants_mean_bic.model.str.split("_").str[1].str[4:]
variants_mean_bic["comparison"] = variants_mean_bic.model.str.split("_").str[2].str[5:]
variants_mean_bic["gb_att"] = (
    variants_mean_bic.model.str.split("_").str[3].str[6:] == "true"
)
variants_mean_bic["gb_alt"] = (
    variants_mean_bic.model.str.split("_").str[4].str[6:] == "true"
)
variants_mean_bic["leak"] = variants_mean_bic.model.str.split("_").str[5].str[3:]
variants_mean_bic["inhibition"] = variants_mean_bic.model.str.split("_").str[6].str[4:]

# Recode switches to ordered Categoricals so I can control their order
variants_mean_bic["integration"] = pd.Categorical(
    variants_mean_bic["integration"],
    categories=["additive", "multiplicative"],
    ordered=True,
).rename_categories([r"$+$", r"$\times$"])
variants_mean_bic["comparison"] = pd.Categorical(
    variants_mean_bic["comparison"],
    categories=["absolute", "vsmean", "n.d."],
    ordered=True,
).rename_categories(["independent", "comparative", "n.d."])
variants_mean_bic["gb_att"] = pd.Categorical(
    variants_mean_bic["gb_att"], categories=[False, True], ordered=True
)
variants_mean_bic["gb_alt"] = pd.Categorical(
    variants_mean_bic["gb_alt"], categories=[False, True], ordered=True
)
variants_mean_bic["leak"] = pd.Categorical(
    variants_mean_bic["leak"],
    categories=["none", "free", "gaze-dependent"],
    ordered=True,
).rename_categories(["None", "Constant", "Gaze"])
variants_mean_bic["inhibition"] = pd.Categorical(
    variants_mean_bic["inhibition"],
    categories=["none", "free", "distance-dependent", "gaze-dependent"],
    ordered=True,
).rename_categories(["None", "Constant", "Distance", "Gaze"])
variants_mean_bic.rename({"bic": "BIC"}, axis=1, inplace=True)
variants_mean_bic.to_csv(join(OUTPUT_DIR, "variants_mean_bic.csv"))

# mark duplicate models with comparative accumulation
# because they are mathematically equivalent to independent accumulation models
# when no or constant leak and inhibition are used
variants_mean_bic_noduplicates = variants_mean_bic.copy()
variants_mean_bic_noduplicates.loc[
    (
        (variants_mean_bic_noduplicates["comparison"] == "comparative")
        & (variants_mean_bic_noduplicates["leak"].isin(["Constant", "None"]))
        & (variants_mean_bic_noduplicates["inhibition"].isin(["Constant", "None"]))
    ),
    "BIC",
] = np.nan

variants_mean_bic_noduplicates.loc[
    (
        (variants_mean_bic_noduplicates["leak"].isin(["Constant", "None"]))
        & (variants_mean_bic_noduplicates["inhibition"].isin(["Constant", "None"]))
    ),
    "comparison",
] = "n.d."

print(
    f"{pd.isnull(variants_mean_bic_noduplicates['BIC']).sum()} were marked as duplicate."
)


# %% 3) Identify overall best fitting model (lowest mean BIC)
top10_variants_pooled_bic = (
    variants_mean_bic_noduplicates.sort_values("BIC")
    .head(10)
    .drop("model", axis=1)
    .reset_index(drop=True)
)
print("Top 10 variants (based on mean BIC across participants)")
print(top10_variants_pooled_bic)
top10_variants_pooled_bic.to_csv(join(OUTPUT_DIR, "top10_variants_pooled_bic.csv"))

# Single best variant:
top1_variant_pooled_estimate_summary = variants_mean_bic_noduplicates.sort_values(
    "BIC"
)["model"].values[0]

# %% 4) Summarise winning model parameters
print("Best variant (average across participants) parameter estimates:")
top1_variant_pooled_estimate_summary = sb_estimates.loc[
    sb_estimates["model"] == top1_variant_pooled_estimate_summary,
    ["alpha", "beta", "lam", "theta"],
].describe()
print(top1_variant_pooled_estimate_summary)
top1_variant_pooled_estimate_summary.to_csv(
    join(OUTPUT_DIR, "top1_variant_pooled_estimate-summary.csv")
)


# %% 5) Compute Switch-levels Mean BIC
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


# %% 6) Count individual best fitting model variants
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

# %% 7) Count individually best fitting switches
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
