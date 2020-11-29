#!~/anaconda3/bin/python

import argparse
import os
from datetime import datetime
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from bambi import Model
from pymc3 import summary, traceplot
from tqdm import tqdm


def runRegression():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", default=1, type=int, help="Set verbosity (0, 1, >1)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../results/0-clean_data",
        help="Relative path to preprocessed data directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results/",
        help="Relative path to results directory.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Analysis label (appended to output filename).",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Toggle overwriting of existing result files.",
    )
    parser.add_argument(
        "--timebinned",
        default=False,
        action="store_true",
        help="Toggle regressions for each time bin.",
    )
    parser.add_argument(
        "--mixed",
        default=False,
        action="store_true",
        help="Toggle random intercepts and random effects for each predictor.",
    )

    parser.add_argument(
        "--withoutchoice",
        default=False,
        action="store_true",
        help="Toggle exclusion of choice predictor.",
    )

    parser.add_argument("--nchains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument(
        "--nsamples", type=int, default=2000, help="Number of samples per MCMC chain."
    )
    parser.add_argument(
        "--ntune",
        type=int,
        default=500,
        help="Number of tuning samples per MCMC chain.",
    )

    parser.add_argument(
        "--center-time",
        default=False,
        action="store_true",
        help="Toggle centering of time predictor.",
    )

    parser.add_argument("--seed", type=int, default=2019, help="Random number seed.")

    args = parser.parse_args()
    VERBOSE = args.verbose
    DATADIR = args.data_dir
    RESULTS_DIR = args.results_dir
    LABEL = args.label
    if LABEL != "":
        LABEL = "_" + LABEL
    N_CHAINS = args.nchains
    N_TUNE = args.ntune
    N_SAMPLES = args.nsamples
    OVERWRITE = args.overwrite
    TIMEBINNED = args.timebinned
    MIXED = args.mixed
    WITHOUTCHOICE = args.withoutchoice
    CENTERTIME = args.center_time
    SEED = args.seed

    np.random.seed(SEED)

    OUTPUT_DIR = join(
        RESULTS_DIR, "S_supplemental-gaze-analyses", "regression_totalDwell" + LABEL
    )

    # Make directories if necessary
    makeDirIfNeeded(OUTPUT_DIR)
    makeDirIfNeeded(join(OUTPUT_DIR, "data"))
    makeDirIfNeeded(os.path.join(OUTPUT_DIR, "estimates"))
    makeDirIfNeeded(os.path.join(OUTPUT_DIR, "traceplots"))

    # Save arguments
    settings = vars(args)
    now = datetime.now()
    datestring = now.strftime("%d/%m/%Y %H:%M:%S")
    settings["time"] = datestring
    with open(join(OUTPUT_DIR, "settings.yaml"), "w") as file:
        settings = yaml.dump(settings, file)

    # Load data
    trials = pd.read_csv(join(DATADIR, "trials.csv"))
    dwells = pd.read_csv(join(DATADIR, "dwells.csv"))

    # merge relevant variables from trials-df into dwells-df
    dwells = dwells.merge(
        trials[
            [
                "subject",
                "trial",
                "pA",
                "pB",
                "pC",
                "mA",
                "mB",
                "mC",
                "effect",
                "target",
                "choice",
                "choice_tcd",
                "pos_A",
                "pos_B",
                "pos_C",
                "pA_top",
                "pB_top",
                "pC_top",
            ]
        ],
        on=["subject", "trial"],
    )

    # Reformat data to one AOI per line
    if VERBOSE > 0:
        print("Reformatting data...")

    if not TIMEBINNED:
        filename = "aois.csv"
        if os.path.isfile(join(OUTPUT_DIR, "data", filename)) and (not OVERWRITE):
            print(
                "\t\tSkipping (found preprocessed data in {})...".format(
                    join(OUTPUT_DIR, "data", filename)
                )
            )
        else:
            aois = dwells2aois(dwells)
            aois.to_csv(join(OUTPUT_DIR, "data", filename), index=False)

    else:
        filename = "aois_timebinned.csv"
        if os.path.isfile(join(OUTPUT_DIR, "data", filename)) and (not OVERWRITE):
            print(
                "\t\tSkipping (found preprocessed data in {})...".format(
                    join(OUTPUT_DIR, "data", filename)
                )
            )
        else:
            aois = []
            time_bins = dwells["time_bin"].unique()
            for time_bin in time_bins:
                if VERBOSE > 1:
                    print(f"\tTime bin {time_bin}...")
                dwells_t = dwells.loc[dwells["time_bin"] == time_bin].copy()
                aois_t = dwells2aois(dwells_t)
                aois_t["time_bin"] = time_bin
                aois.append(aois_t)
            aois = pd.concat(aois)
            aois = aois.sort_values(["subject", "trial", "time_bin", "aoi"])
            aois.to_csv(join(OUTPUT_DIR, "data", filename), index=False)

    # Run regression model
    if CENTERTIME:
        time_bin_pred = "time_bin_c"
    else:
        time_bin_pred = "time_bin"
    dependent = "dur_total"
    predictors = [
        "col_c",
        "row_c",
        "rank_c",
        "is_target",
        "is_decoy",
        "is_decoy",
        "is_probability",
    ]
    if not WITHOUTCHOICE:
        predictors.append("is_chosen")
    if MIXED:
        intercept = "0"
        if not TIMEBINNED:
            random_terms = [f"1|subject"] + ["{p}|subject" for p in predictors]
        else:  # timebinned
            predictors = predictors + [f"{p}:{time_bin_pred}" for p in predictors]
            predictors.append(time_bin_pred)
            random_terms = ["1|subject"] + [f"{p}|subject" for p in predictors]
    else:  # not mixed
        intercept = "1"
        random_terms = []
        if TIMEBINNED:
            predictors = predictors + [f"{p}:{time_bin_pred}" for p in predictors]
            predictors.append(time_bin_pred)

    formula = dependent + " ~ " + intercept + " + " + " + ".join(predictors)

    if VERBOSE > 0:
        print("Running regression model...")
        print(f"\t{formula}")
        print(
            "\trandom terms: " + ", ".join(term.split("|")[0] for term in random_terms)
        )

    # Reload data
    if not TIMEBINNED:
        data = pd.read_csv(join(OUTPUT_DIR, "data", "aois.csv"))
        data["time_bin"] = 0
    else:
        data = pd.read_csv(join(OUTPUT_DIR, "data", "aois_timebinned.csv"))

    if CENTERTIME:
        data["time_bin_c"] = data["time_bin"] - data["time_bin"].mean()

    for effect in ["attraction", "compromise"]:
        data_subset = data.loc[data["effect"] == effect].copy()
        ESTIMATES_FILE = join(
            OUTPUT_DIR, "estimates", f"regression_totalDwell{LABEL}_e-{effect}.csv",
        )

        if os.path.exists(ESTIMATES_FILE) and (not OVERWRITE):
            print(
                f"Found existing estimates in {ESTIMATES_FILE}. Skipping estimation..."
            )
        else:
            if VERBOSE > 0:
                print(f"Fitting model for {effect.capitalize()} trials")

            model = Model(data_subset.reset_index(drop=True))

            results = model.fit(
                formula,
                random=random_terms,
                family="gaussian",
                samples=N_SAMPLES,
                chains=N_CHAINS,
                tune=N_TUNE,
            )

            trace = model.backend.trace
            summary_df = summary(results, hdi_prob=0.95)
            for predictor in predictors:
                summary_df.loc[predictor + "[0]", "P>0"] = np.mean(
                    trace.get_values(predictor) > 0
                )

            summary_df.to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    "estimates",
                    f"regression_totalDwell{LABEL}_e-{effect}.csv",
                )
            )

            traceplot(model.backend.trace, var_names=predictors)
            plt.savefig(
                os.path.join(
                    OUTPUT_DIR,
                    "traceplots",
                    f"regression_totalDwell{LABEL}_e-{effect}_traceplot.png",
                ),
                dpi=100,
            )
            plt.close()


def dwells2aois(df):
    df["aoi"] = df["attribute"] + df["alternative"]

    # Compute total dwell duration toward each AOI in each trial
    dur = (
        df.groupby(["subject", "trial", "aoi"])["duration"]
        .sum()
        .rename("dur_total")
        .reset_index()
        .pivot_table(values="dur_total", columns="aoi", index=["subject", "trial"])
        .fillna(0)
        .reset_index()
        .melt(
            id_vars=["subject", "trial"],
            value_vars=["pA", "pB", "pC", "mA", "mB", "mC"],
            value_name="dur_total",
            var_name="aoi",
        )
        .sort_values(["subject", "trial", "aoi"])
        .reset_index(drop=True)
        .merge(
            df[
                [
                    "subject",
                    "trial",
                    "effect",
                    "target",
                    "choice",
                    "choice_tcd",
                    "pA",
                    "pB",
                    "pC",
                    "mA",
                    "mB",
                    "mC",
                    "pos_A",
                    "pos_B",
                    "pos_C",
                    "pA_top",
                    "pB_top",
                    "pC_top",
                ]
            ].drop_duplicates(),
            on=["subject", "trial"],
            how="left",
        )
    )
    dur["value"] = dur.apply(lambda x: x[x["aoi"]], axis=1)
    # Copmute number of dwells to each AOI in each trial
    count = (
        df.groupby(["subject", "trial", "aoi"])["duration"]
        .count()
        .rename("count")
        .reset_index()
        .pivot_table(values="count", columns="aoi", index=["subject", "trial"])
        .fillna(0)
        .reset_index()
        .melt(
            id_vars=["subject", "trial"],
            value_vars=["pA", "pB", "pC", "mA", "mB", "mC"],
            value_name="count",
            var_name="aoi",
        )
        .sort_values(["subject", "trial", "aoi"])
        .reset_index(drop=True)
        .merge(
            df[
                [
                    "subject",
                    "trial",
                    "pos_A",
                    "pos_B",
                    "pos_C",
                    "pA_top",
                    "pB_top",
                    "pC_top",
                ]
            ].drop_duplicates(),
            on=["subject", "trial"],
            how="left",
        )
    )
    # Combine these
    output = dur.merge(
        count[["subject", "trial", "aoi", "count"]], on=["subject", "trial", "aoi"]
    )

    output = output.merge(
        pd.DataFrame(
            output.apply(decode_pos, axis=1).to_list(), columns=["row", "col"]
        ),
        left_index=True,
        right_index=True,
    )[
        [
            "subject",
            "trial",
            "effect",
            "target",
            "aoi",
            "dur_total",
            "count",
            "row",
            "col",
            "choice",
            "choice_tcd",
            "value",
        ]
    ]

    # additional variables
    output["alternative"] = output["aoi"].str[1]
    output["attribute"] = output["aoi"].str[0]
    output["rank"] = output.groupby(["subject", "trial", "attribute"])["value"].rank()
    output["is_fixated"] = np.where(output["count"] > 0, 1, 0)
    output["is_target"] = np.where(
        output["effect"].isin(["attraction", "compromise"]),
        np.where(output["aoi"].str[1] == output["target"], 1, 0),
        np.nan,
    )
    output["is_decoy"] = np.where(
        output["effect"].isin(["attraction", "compromise"]),
        np.where(output["aoi"].str[1] == "C", 1, 0),
        np.nan,
    )
    output["is_probability"] = np.where(output["aoi"].str[0] == "p", 1, 0)
    output["is_chosen"] = np.where(output["choice"] == output["aoi"].str[1], 1, 0)

    # center some variables
    output["rank_c"] = output["rank"] - 2
    output["row_c"] = output["row"] - 0.5
    output["col_c"] = output["col"] - 1

    return output


def decode_pos(x):
    alt = x["aoi"][1]
    att = x["aoi"][0]
    col = x["pos_{}".format(alt)]
    p_top = x["p{}_top".format(alt)]
    if p_top:
        if att == "p":
            row = 0
        else:
            row = 1
    else:
        if att == "p":
            row = 1
        else:
            row = 0
    return row, col


def makeDirIfNeeded(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    runRegression()
