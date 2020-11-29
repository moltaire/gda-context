#!/usr/bin/python
"""
Gaze-dependent accumulation in context-dependent risky choice
This script performs preprocessing of behavioural and gaze data
    1. Exclude subjects
        a) marked to exclude (Experiment crash or no understanding of task)
        b) with bad eyetracking data
    2. Compute gender and age distribution of remaining sample
    3. Load raw behavioural data and process it, recoding choice variables, etc.
    4. Load eyetracking data (from eventdetector), process it, including AOI assignment
    5. Save preprocessed dataframes (trials, fixations, dwells)
Author: Felix Molter, felixmolter@gmail.com
"""

import argparse
from os import makedirs
from os.path import exists, join

import numpy as np
import pandas as pd


def preprocessData():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Toggle verbose output."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/",
        help="Relative path to raw data directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results/",
        help="Relative path to results directory.",
    )
    parser.add_argument(
        "--remove-last-fixation",
        default=False,
        action="store_true",
        help="Toggle removal of last fixation event in each trial.",
    )
    parser.add_argument(
        "--n_time_bins",
        default=5,
        type=int,
        help="Number of time bins for analyses of trial dynamics.",
    )

    # command line arguments
    args = parser.parse_args()
    VERBOSE = args.verbose
    RAWDATA_DIR = args.data_dir
    RESULTS_DIR = args.results_dir
    OUTPUT_DIR = join(OUTPUT_DIR, "0-clean_data")
    CALIBRATION_DIR = join(RAWDATA_DIR, "eyetracking_calibration")
    BEHAVIOURAL_DIR = join(RAWDATA_DIR, "behaviour")
    EYETRACKING_DIR = join(RAWDATA_DIR, "eyetracking")
    REMOVE_LAST_FIXATION = args.remove_last_fixation
    N_TIME_BINS = args.n_time_bins

    if REMOVE_LAST_FIXATION:
        rmlastfixstr = "_nolastfix"
    else:
        rmlastfixstr = ""

    if VERBOSE:
        print("Preprocessing behavioural and eyetracking data...")

    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR)

    # Load subject overview
    subject_overview = pd.read_csv(
        join(RAWDATA_DIR, "subject_overview.csv"), parse_dates=["date"]
    )

    # 1. Exclude subjects with problematic data
    # -----------------------------------------

    # a) Subjects manually marked for exclusion
    excluded_subjects = subject_overview.loc[
        subject_overview["exclude"] == True, "subject_id"
    ].values
    exclusion_reasons = subject_overview.loc[
        subject_overview["exclude"] == True, "comment"
    ].values
    if len(excluded_subjects) > 0:
        if VERBOSE:
            for subject, reason in zip(excluded_subjects, exclusion_reasons):
                print("\tSubject {} removed.\tReason: {}".format(subject, reason))
    subjects = subject_overview["subject_id"][subject_overview["exclude"] != True]

    # b) Subjects with bad eyetracking calibrations
    # Load eyetracking calibration files and exclude
    bad_calibration = []
    for subject in subjects:
        calibration_subject = pd.read_csv(
            join(CALIBRATION_DIR, "comp_calibration_{}.csv".format(subject)),
            index_col=False,
        )
        bad_blocks = []
        for block in [1, 2, 3]:
            if not np.any(
                calibration_subject.loc[
                    calibration_subject["block"] == block, "success"
                ]
            ):
                bad_blocks.append(block)
        if len(bad_blocks) > 0:
            bad_calibration.append(subject)
            if VERBOSE:
                print(
                    "\tSubject {} removed.\tReason: Bad eyetracking calibration".format(
                        subject
                    )
                )

    subjects = [subject for subject in subjects if subject not in bad_calibration]

    # 2. Compute gender and age distribution of remaining sample
    # ----------------------------------------------------------
    if VERBOSE:
        print("{} subjects remaining.".format(len(subjects)))
        print(
            "\t{} female, {} male".format(
                (
                    subject_overview.loc[
                        subject_overview["subject_id"].isin(subjects), "gender"
                    ]
                    == "f"
                ).sum(),
                (
                    subject_overview.loc[
                        subject_overview["subject_id"].isin(subjects), "gender"
                    ]
                    == "m"
                ).sum(),
            )
        )
        print(
            "\tMean ± S.D. age = {:.2f} ± {:.2f}".format(
                subject_overview.loc[
                    subject_overview["subject_id"].isin(subjects), "age"
                ].mean(),
                subject_overview.loc[
                    subject_overview["subject_id"].isin(subjects), "age"
                ].std(),
            )
        )

    # 3. Load raw behavioural data and process it, recoding choice variables, etc.
    # ----------------------------------------------------------------------------

    trials = []
    for s, subject in enumerate(subjects):

        # Read behavioural data
        trials_s = pd.read_csv(
            join(BEHAVIOURAL_DIR, "comp_behaviour_{}.csv".format(subject))
        )
        # Drop practice and indifference estimation trials
        trials_s = (
            trials_s.loc[~trials_s["block"].str.contains("Practice|Estimation")]
            .reset_index(drop=True)
            .copy()
        )

        # Add trial index over blocks
        trials_s.loc[:, "trial"] = np.arange(1, 226)

        # Make all timestamps relative to the first block onset
        first_block_onset = trials_s["block_onset"][0]
        trials_s["trial_onset"] -= first_block_onset
        trials_s["fixation_cross_onset"] -= first_block_onset
        trials_s["block_onset"] -= first_block_onset

        trials.append(trials_s)

    trials = pd.concat(trials)

    # Recode response key from 'left', 'mid', 'right' to 'L', 'M', 'R'
    trials.loc[:, "key"] = trials["key"].str.upper().apply(lambda x: x[0])

    # Extract effect and target alternative
    tmp = pd.Categorical(
        trials["condition"], categories=[1, 2, 31, 32, 99]
    ).rename_categories(
        {
            1: "compromise_A",
            2: "compromise_B",
            31: "attraction_A",
            32: "attraction_B",
            99: "filler",
        }
    )
    trials["effect"] = pd.Categorical(
        pd.Series(tmp).str.split("_").str.get(0),
        categories=["attraction", "compromise", "filler"],
    )
    trials["target"] = pd.Categorical(
        pd.Series(tmp).str.split("_").str.get(1), categories=["A", "B"]
    )
    del tmp

    # Recode all timing variables to ms
    trials["trial_onset"] *= 1000
    trials["fixation_cross_onset"] *= 1000
    trials["block_onset"] *= 1000
    trials["rt"] *= 1000

    # Compute a trial-end variable to truncate fixation data
    trials["trial_end"] = trials["trial_onset"] + trials["rt"]

    # Select the correct pC and mC:
    trials["pC"] = np.where(trials["target"] == "B", trials["pCB"], trials["pCA"])
    trials["mC"] = np.where(trials["target"] == "B", trials["mCB"], trials["mCA"])

    # Was the focal alternative chosen?
    conditions = [
        pd.isnull(trials["target"]),  # filler trials don't have a target
        trials["choice"] == "C",  # C choices are always decoy choices
        trials["choice"] == trials["target"],  # target choices
        trials["choice"] != trials["target"],
    ]  # competitor choices
    choices = [np.nan, "decoy", "target", "competitor"]
    trials["choice_tcd"] = pd.Categorical(
        np.select(conditions, choices), categories=["target", "competitor", "decoy"]
    )

    # Recode alternative positions (0 = left, 1 = mid, 2 = right)
    trials["pos_A"] = trials["position_A"] - 1
    trials["pos_B"] = trials["position_B"] - 1
    trials["pos_C"] = trials["position_C"] - 1

    # Recode attribute positions (whether an option's probability was shown top or bottom)
    trials["attribute_positions"] = trials["attribute_positions"].apply(
        lambda x: "{0:03d}".format(x)
    )
    trials["pA_top"] = (
        trials.apply(lambda x: x["attribute_positions"][int(x["pos_A"])], axis=1) == "1"
    )
    trials["pB_top"] = (
        trials.apply(lambda x: x["attribute_positions"][int(x["pos_B"])], axis=1) == "1"
    )
    trials["pC_top"] = (
        trials.apply(lambda x: x["attribute_positions"][int(x["pos_C"])], axis=1) == "1"
    )

    trials = trials[
        [
            "subject",
            "block",
            "trial",
            "effect",
            "target",
            "trial_onset",
            "trial_end",
            "key",
            "choice",
            "choice_tcd",
            "rt",
            "pA",
            "mA",
            "pB",
            "mB",
            "pC",
            "mC",
            "pos_A",
            "pos_B",
            "pos_C",
            "pA_top",
            "pB_top",
            "pC_top",
        ]
    ]

    # 4. Load eyetracking data (from eventdetector), process it, including AOI assignment
    fixations = []
    for s, subject in enumerate(subjects):

        fix_s = []

        for block in [1, 2, 3]:
            fix_b = pd.read_csv(
                join(
                    EYETRACKING_DIR,
                    "comp_eyetracking_{}_block-{}.txt".format(subject, block),
                ),
                sep="\t",
                skiprows=20,
            ).reset_index()
            # only keep fixations and user UserMessages
            fix_b = fix_b[
                fix_b["level_0"].str.contains("Fixation|UserEvent")
            ].reset_index()

            # remove columns containing saccade information
            # and set new column names
            column_names = [
                "event_type",
                "trial",  # this trial variable is constant and does not track the actual trials
                "event_number",
                "event_start",
                "event_end",  # this column also contains user messages, i.e., triggers
                "event_duration",
                "x",
                "y",
            ]
            fix_b = fix_b.iloc[:, 1 : len(column_names) + 1]
            fix_b.columns = column_names

            # Identify first block onset timestamp
            if block == 1:
                # The first block onset message occurs twice in the eyetracking file:
                # Once before the estimation trials, and another time before the actual first block.
                # Therefore we need the last of these two occurences.
                first_block_onset = fix_b.loc[
                    fix_b["event_end"].str.contains("Block1"), "event_start"
                ].values[-1]

            # Extract trigger messages, remove everything that is not within
            # the choice phase, and code trial variable
            messages = fix_b["event_end"].copy()
            # set non-message data to nan and fill messages downwards
            messages.loc[~fix_b["event_end"].str.contains("# Message:")] = np.nan
            fix_b["message"] = messages.fillna(method="ffill")
            # Remove data before first trigger
            fix_b = fix_b.loc[~pd.isnull(fix_b["message"])]
            # Only keep data from within choice phases
            fix_b = fix_b.loc[fix_b["message"].str.contains("Choice")]
            # Only keep fixation data
            fix_b = fix_b.loc[fix_b["event_type"].str.contains("Fixation")]
            # Remove data from estimation trials
            fix_b = fix_b.loc[~fix_b["message"].str.contains("E")]
            # Identify correct trial numbers
            fix_b["trial"] = (
                fix_b.message.str.split("Choice")
                .str.get(-1)
                .str.split(".png")
                .str.get(0)
                .astype(int)
            )
            fix_b["trial"] = (block - 1) * 75 + fix_b["trial"]

            # Add subject & block
            fix_b["block"] = block
            fix_b["subject"] = subject

            # Reduce to relevant columns
            fix_b = fix_b[
                [
                    "subject",
                    "block",
                    "trial",
                    "event_start",
                    "event_duration",
                    "event_end",
                    "x",
                    "y",
                ]
            ]

            fix_s.append(fix_b)
        fix_s = pd.concat(fix_s)
        # Recode timestamps relative to first block onset
        # and convert to ms (SMI log is in ns)
        fix_s["event_start"] = (
            fix_s["event_start"].astype(float) - first_block_onset
        ) / 1000
        fix_s["event_duration"] = (fix_s["event_duration"].astype(float) / 1000).astype(
            int
        )
        fix_s["event_end"] = (
            fix_s["event_end"].astype(float) - first_block_onset
        ) / 1000

        fixations.append(fix_s)

    fixations = pd.concat(fixations)

    # Convert eyetracking coordinates (0, 0) at top left of the screen
    # to more intuitive coordinate system (0, 0) at bottom left of the screen
    screen_width = 1280  # horizontal screen resulution in px
    screen_height = 1024  # vertical screen resulution in px
    fixations["y"] = screen_height - fixations["y"]

    # Extract AOI hits
    # First, define AOIs
    distance = 391  # distance in px between AOI centers, horizontal and vertical
    aoi_width = 385  # box width in px
    aoi_height = 240  # box height in px
    aois = {
        "top-left": [
            0.5 * screen_width - distance - 0.5 * aoi_width,  # x of left boundary
            0.5 * screen_width - distance + 0.5 * aoi_width,  # x of right boundary
            0.5 * screen_height
            + 0.5 * distance
            - 0.5 * aoi_height,  # y of bottom boundary
            0.5 * screen_height + 0.5 * distance + 0.5 * aoi_height,
        ],  # y of bottom boundary
        "top-mid": [
            0.5 * screen_width - 0.5 * aoi_width,  # x of left boundary
            0.5 * screen_width + 0.5 * aoi_width,  # x of right boundary
            0.5 * screen_height
            + 0.5 * distance
            - 0.5 * aoi_height,  # y of bottom boundary
            0.5 * screen_height + 0.5 * distance + 0.5 * aoi_height,
        ],  # y of bottom boundary
        "top-right": [
            0.5 * screen_width + distance - 0.5 * aoi_width,  # x of left boundary
            0.5 * screen_width + distance + 0.5 * aoi_width,  # x of right boundary
            0.5 * screen_height
            + 0.5 * distance
            - 0.5 * aoi_height,  # y of bottom boundary
            0.5 * screen_height + 0.5 * distance + 0.5 * aoi_height,
        ],  # y of bottom boundary
        "bot-left": [
            0.5 * screen_width - distance - 0.5 * aoi_width,  # x of left boundary
            0.5 * screen_width - distance + 0.5 * aoi_width,  # x of right boundary
            0.5 * screen_height
            - 0.5 * distance
            - 0.5 * aoi_height,  # y of bottom boundary
            0.5 * screen_height - 0.5 * distance + 0.5 * aoi_height,
        ],  # y of bottom boundary
        "bot-mid": [
            0.5 * screen_width - 0.5 * aoi_width,  # x of left boundary
            0.5 * screen_width + 0.5 * aoi_width,  # x of right boundary
            0.5 * screen_height
            - 0.5 * distance
            - 0.5 * aoi_height,  # y of bottom boundary
            0.5 * screen_height - 0.5 * distance + 0.5 * aoi_height,
        ],  # y of bottom boundary
        "bot-right": [
            0.5 * screen_width + distance - 0.5 * aoi_width,  # x of left boundary
            0.5 * screen_width + distance + 0.5 * aoi_width,  # x of right boundary
            0.5 * screen_height
            - 0.5 * distance
            - 0.5 * aoi_height,  # y of bottom boundary
            0.5 * screen_height - 0.5 * distance + 0.5 * aoi_height,
        ],  # y of bottom boundary
    }

    # AOI hit conditions
    conditions = [
        (
            (fixations["x"] >= xmin)
            & (fixations["x"] <= xmax)
            & (fixations["y"] >= ymin)
            & (fixations["y"] <= ymax)
        )
        for aoi, (xmin, xmax, ymin, ymax) in aois.items()
    ]
    options = [aoi for aoi, _ in aois.items()]
    fixations["aoi"] = pd.Categorical(
        np.select(conditions, options, default=np.nan), categories=options
    )

    # Include columns from trial dataframe
    fixations = fixations.merge(
        trials[
            [
                "subject",
                "trial",
                "pA",
                "mA",
                "pB",
                "mB",
                "pC",
                "mC",
                "pos_A",
                "pos_B",
                "pos_C",
                "pA_top",
                "pB_top",
                "pC_top",
                "trial_onset",
                "trial_end",
            ]
        ],
        on=["subject", "trial"],
        how="inner",
    )

    # Truncate last fixation if it overlaps with the trial end (response)
    fixations["event_duration"] = np.where(
        fixations["event_end"].astype(int) > fixations["trial_end"],
        fixations["trial_end"] - fixations["event_start"],
        fixations["event_duration"],
    )

    # remove fixations that have negative durations after truncation
    # these fixations can exist because of a timing difference between
    # response and the feedback-phase trigger
    fixations = fixations[fixations["event_duration"] > 0]

    # exclude pre-trial fixations completely, do not truncate at the beginning of a trial.
    # this needs to be done, because the eyetracking trigger is sent before the screen flip and reaction time start!
    # So, basically we are counting only from the first fixation AFTER the trial is shown.
    # in effect, the "delete first fixation option" in the iView event
    # extractor does not do anything.
    fixations = fixations[fixations["event_start"] > fixations["trial_onset"]]

    # Clean fixations according to Krajbich & Rangel, 2011 (PNAS) rules:
    # 1) if a sequence is A -> NaN -> A, this is recoded to A -> A -> A
    # 2) if a sequence is A -> NaN -> B, this is recoded to A -> B (the NaN is dropped)
    fixations["prev_aoi"] = fixations.groupby("trial")["aoi"].shift(1)
    fixations["next_aoi"] = fixations.groupby("trial")["aoi"].shift(-1)

    fixations["aoi_cleaned"] = np.where(
        ~pd.isnull(fixations["aoi"]),  # non-NaN AOIs
        fixations["aoi"],  # stay what they are
        np.where(
            (pd.isnull(fixations["aoi"]))
            & (  # NaN AOIs
                fixations["prev_aoi"] == fixations["next_aoi"]
            ),  # where the previous and next AOI are the same
            fixations["prev_aoi"],  # are assigned to that AOI
            np.nan,
        ),
    )  # and NaN otherwise (to be dropped)
    fixations.dropna(subset=["aoi_cleaned"], inplace=True)
    fixations["number"] = fixations.groupby(["subject", "trial"]).cumcount()

    # Compute time bins
    fixations["trial_time"] = (fixations["event_start"] - fixations["trial_onset"]) / (
        fixations["trial_end"] - fixations["trial_onset"]
    )
    fixations["time_bin"] = pd.cut(
        fixations["trial_time"],
        bins=np.linspace(0, 1, N_TIME_BINS + 1),
        labels=np.arange(N_TIME_BINS),
    )

    # Optional: Remove last fixation
    if REMOVE_LAST_FIXATION:
        if VERBOSE:
            print("Removing last fixation event from each trial...")
        fixations = fixations[
            fixations.groupby(["subject", "trial"]).cumcount(ascending=False) > 0
        ].reset_index(drop=True)

    # Extract row and column
    fixations["row"] = np.where(
        fixations["aoi_cleaned"].str.split("-").str.get(0) == "bot", 0, 1
    )
    fixations["col"] = np.where(
        fixations["aoi_cleaned"].str.split("-").str.get(1) == "left",
        0,
        np.where(fixations["aoi_cleaned"].str.split("-").str.get(1) == "mid", 1, 2),
    )
    # Extract attribute and alternative
    # alternative
    conditions = [
        fixations["pos_A"] == fixations["col"],
        fixations["pos_B"] == fixations["col"],
        fixations["pos_C"] == fixations["col"],
    ]
    choices = ["A", "B", "C"]
    fixations["alternative"] = pd.Categorical(
        np.select(conditions, choices), categories=["A", "B", "C"]
    )
    # attribute
    conditions = [
        fixations["alternative"] == "A",
        fixations["alternative"] == "B",
        fixations["alternative"] == "C",
    ]
    choices = [fixations["pA_top"], fixations["pB_top"], fixations["pC_top"]]
    fixations["pFix_top"] = np.select(conditions, choices)
    conditions = [
        (fixations["pFix_top"]) & (fixations["row"] == 1),  # p
        (fixations["pFix_top"]) & (fixations["row"] == 0),  # m
        (~fixations["pFix_top"]) & (fixations["row"] == 1),  # m
        (~fixations["pFix_top"]) & (fixations["row"] == 0),
    ]  # p
    choices = ["p", "m", "m", "p"]
    fixations["attribute"] = pd.Categorical(
        np.select(conditions, choices), categories=["p", "m"]
    )

    # Dwell data (cumulation of subsequent fixations towards the same AOI)
    dwells = fixations.copy()
    dwells["prev_aoi"] = dwells.groupby("trial")["aoi_cleaned"].shift(1)
    dwells["next_aoi"] = dwells.groupby("trial")["aoi_cleaned"].shift(-1)
    dwells["new_dwell"] = dwells["aoi_cleaned"] != dwells["prev_aoi"]
    dwells["new_dwell_idx"] = (
        dwells.groupby(["subject", "trial"])["new_dwell"]
    ).cumsum()
    dwell_duration = dwells.groupby(
        ["subject", "trial", "new_dwell_idx"], as_index=False
    )["event_duration"].sum()
    dwells = (
        dwells.groupby(["subject", "trial", "new_dwell_idx"])
        .head(1)
        .drop("event_duration", axis=1)
    )
    dwells = dwells.merge(
        dwell_duration[["subject", "trial", "new_dwell_idx", "event_duration"]],
        on=["subject", "trial", "new_dwell_idx"],
        how="inner",
    )
    dwells["number"] = dwells.groupby(["subject", "trial"]).cumcount()

    # 5. Save preprocessed data (trials, fixations, dwells)
    (
        trials[
            [
                "subject",
                "block",
                "trial",
                "effect",
                "target",
                "key",
                "choice",
                "choice_tcd",
                "rt",
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
        ].to_csv(join(OUTPUT_DIR, "trials{}.csv".format(rmlastfixstr)))
    )

    (
        fixations[
            [
                "subject",
                "block",
                "trial",
                "number",
                "event_duration",
                "x",
                "y",
                "aoi_cleaned",
                "row",
                "col",
                "alternative",
                "attribute",
                "time_bin",
            ]
        ]
        .rename({"event_duration": "duration", "aoi_cleaned": "aoi"}, axis="columns")
        .to_csv(join(OUTPUT_DIR, "fixations{}.csv".format(rmlastfixstr)))
    )

    (
        dwells[
            [
                "subject",
                "block",
                "trial",
                "number",
                "event_duration",
                "x",
                "y",
                "aoi_cleaned",
                "row",
                "col",
                "alternative",
                "attribute",
                "time_bin",
            ]
        ]
        .rename({"event_duration": "duration", "aoi_cleaned": "aoi"}, axis="columns")
        .to_csv(join(OUTPUT_DIR, "dwells{}.csv".format(rmlastfixstr)))
    )

    if VERBOSE:
        print("Created three output files in '{}':".format(OUTPUT_DIR))
        print(
            "\t1. trials{}.csv – Each entry corresponds to a single trial.".format(
                rmlastfixstr
            )
        )
        print(
            "\t2. fixations{}.csv – Each entry corresponds to a single fixation event, extracted from the SMI Event Detector.".format(
                rmlastfixstr
            )
        )
        print(
            "\t3. dwells{}.csv – Each entry corresponds to a single dwell (i.e., summed consecutive fixations towards the same AOI).".format(
                rmlastfixstr
            )
        )


if __name__ == "__main__":
    preprocessData()
