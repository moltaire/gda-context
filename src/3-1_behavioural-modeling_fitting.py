#!/usr/bin/python
"""
Gaze-dependent accumulation in context-dependent risky choice
This script performs maximum likelihood estimation and prediction of the behavioural models
    0. Processing of behavioural and fixation data, so that they fit the model code.
    1. (Parallel, if specified) maximum likelihood estimation and prediction of all models listed in `models`
Author: Felix Molter, felixmolter@gmail.com
"""

import argparse
from functools import partial
from os import makedirs
from os.path import exists, join

import numpy as np
import pandas as pd

import behavioural_models as bm


def modelFitting():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", default=0, type=int, help="Set verbosity (0, 1, >1)."
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
        "--optmethod",
        type=str,
        default="minimize",
        help="scipy optimizer. Use 'minimize' or 'differential_evolution'.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Toggle overwriting of existing result files.",
    )
    parser.add_argument(
        "--ncores", type=int, default=1, help="Number of cores for parallel processing."
    )
    parser.add_argument(
        "--nruns", type=int, default=10, help="Number of optimization runs per model."
    )
    parser.add_argument(
        "--nsims",
        type=int,
        default=1,
        help="Number of simulation repetitions per trial.",
    )
    parser.add_argument("--seed", type=int, default=2019, help="Random number seed.")

    args = parser.parse_args()
    VERBOSE = args.verbose
    DATADIR = args.data_dir
    RESULTS_DIR = args.results_dir
    OUTPUT_DIR = join(RESULTS_DIR, "3-behavioural-modeling")
    LABEL = args.label
    if LABEL != "":
        LABEL = "_" + LABEL
    N_CORES = args.ncores
    N_RUNS = args.nruns
    N_SIMS = args.nsims
    OPTMETHOD = args.optmethod
    OVERWRITE = args.overwrite
    SEED = args.seed

    np.random.seed(SEED)

    # Define models to be fit
    models = [
        bm.models.ExpectedUtility,
        bm.mdft.MDFT,
        bm.models.GazeBaselineStat,
        bm.models.GazeBaselineDyn,
        bm.models.Glickman1Layer,
    ]

    # Load trial data
    trials = pd.read_csv(join(DATADIR, "trials_with-dwells.csv"), index_col=0)

    # Load fixation data
    fixations = pd.read_csv(join(DATADIR, "fixations.csv"), index_col=0)
    fixations["duration"] /= 1000

    # 0. Processing of behavioural and fixation data, so that they fit the model code
    # -------------------------------------------------------------------------------

    # For each trial, extract fixated alternatives and fixation durations
    # Note that a trial might have missing eyetracking data
    # fixations['alternative'] is coded as "A", "B", "C". We need to change this to 0, 1, 2
    fixations["alternative"] = pd.Categorical(
        fixations["alternative"], categories=["A", "B", "C"]
    ).codes
    fixations["attribute"] = pd.Categorical(
        fixations["attribute"], categories=["p", "m"]
    ).codes
    # For each trial, make a list of fixated_alternatives and fixation durations
    fixated_alternatives = (
        fixations.groupby(["subject", "trial"])["alternative"]
        .apply(lambda x: np.array(x))
        .rename("fixated_alternatives")
        .reset_index()
    )
    # For each trial, make a list of fixated_alternatives and fixation durations
    fixated_attributes = (
        fixations.groupby(["subject", "trial"])["attribute"]
        .apply(lambda x: np.array(x))
        .rename("fixated_attributes")
        .reset_index()
    )
    fixation_durations = (
        fixations.groupby(["subject", "trial"])["duration"]
        .apply(lambda x: np.array(x))
        .rename("fixation_durations")
        .reset_index()
    )
    # And merge them into the trial data
    trials = trials.merge(fixated_alternatives, on=["subject", "trial"], how="left")
    trials = trials.merge(fixated_attributes, on=["subject", "trial"], how="left")
    trials = trials.merge(fixation_durations, on=["subject", "trial"], how="left")
    # Recode choices into numeric
    trials["choice"] = pd.Categorical(
        trials["choice"], categories=["A", "B", "C"]
    ).codes

    # 1. (Parallel, if specified) maximum likelihood estimation and prediction of all models listed in `models`
    # ---------------------------------------------------------------------------------------------------------

    subjects = np.unique(trials["subject"])
    n_subjects = len(subjects)

    def input_generator(data, subjects, models, verbose=False):
        "Generates subject-wise inputs to fit_subject_model function."
        for model in models:
            if VERBOSE > 0:
                _ = model(data=data)
                print(
                    "{}: Parameter estimation and choice prediction...".format(_.label)
                )
                del _
            for subject in subjects:
                # Subset subject data
                subject_data = data[data["subject"] == subject].copy()
                # Initialize model instance
                subject_model = model(data=subject_data)
                subject_model.subject = subject
                # Yield everything as a tuple to input into fit_subject_model function
                yield subject, subject_model

    # Fix some arguments of the fitting function, so that it only takes subject data as argument
    fit_predict = partial(
        fit_predict_subject_model,
        n_runs=N_RUNS,
        optmethod=OPTMETHOD,
        n_sims=N_SIMS,
        output_dir=OUTPUT_DIR,
        label=LABEL,
        seed=SEED,
        overwrite=OVERWRITE,
        verbose=VERBOSE,
    )

    # Run fitting in parallel if multiple cores specified
    if N_CORES > 1:
        from multiprocessing import Pool

        pool = Pool(N_CORES)
        # next line can use `map`, then outputs work fine, or `imap_unordered` which maps better, but messes with output concatenation.
        results = pool.map(
            fit_predict, input_generator(trials, subjects, models, VERBOSE)
        )
        all_estimates = [result[0] for result in results]
        all_predictions = [result[1] for result in results]
    # otherwise sequential
    else:
        all_estimates = []
        all_predictions = []
        for s, (subject, subject_model) in enumerate(
            input_generator(trials, subjects, models, VERBOSE)
        ):
            if VERBOSE > 0:
                print("Subject {} ({}/{})".format(subject, s + 1, n_subjects))
            (estimates, predictions) = fit_predict((subject, subject_model))
            all_estimates.append(estimates)
            all_predictions.append(predictions)

    # Collect and save results
    all_estimates_output_path = join(OUTPUT_DIR, "estimates", "estimates{}.csv").format(
        LABEL
    )
    all_estimates = pd.concat(all_estimates, sort=True)
    all_estimates.to_csv(all_estimates_output_path, index=False)
    all_predictions_output_path = join(
        OUTPUT_DIR, "predictions", "predictions{}.csv"
    ).format(LABEL)
    all_predictions = pd.concat(all_predictions, sort=True)
    all_predictions.to_csv(all_predictions_output_path, index=False)


def fit_predict_subject_model(
    variable_input,
    n_runs=1,
    optmethod="minimize",
    n_sims=1,
    output_dir="",
    label="",
    seed=None,
    overwrite=False,
    verbose=False,
):
    """
    Fit single subject model and predict choices from it.
    variable_input is generated by input_generator() and contains
    the subject ID, the model instance with attached data.
    """
    # Unpack input
    subject, subject_model = variable_input
    model_label = subject_model.label.replace(" ", "-").lower()

    # Organize output folders
    estimates_output_folder = join(output_dir, "estimates", model_label)
    predictions_output_folder = join(output_dir, "predictions", model_label)
    if not exists(estimates_output_folder):
        makedirs(estimates_output_folder)
    if not exists(predictions_output_folder):
        makedirs(predictions_output_folder)

    # Create output filenames
    estimates_output_file = "estimates_{}_{}{}.csv".format(model_label, subject, label)
    estimates_output_path = join(estimates_output_folder, estimates_output_file)
    predictions_output_file = "predictions_{}_{}{}.csv".format(
        model_label, subject, label
    )
    predictions_output_path = join(predictions_output_folder, predictions_output_file)

    # Parameter estimation
    if not overwrite and exists(estimates_output_path):
        if verbose > 0:
            print(
                "Found existing estimates at '{}'. Skipping estimation...".format(
                    estimates_output_path
                )
            )
        estimates_df = pd.read_csv(estimates_output_path)
        estimates = [
            estimates_df[parameter].values[0]
            for parameter in subject_model.parameter_names
        ]
        estimates_df["bic"] = (
            2 * estimates_df["nll"]
            + np.log(subject_model.n_trials) * subject_model.n_parameters
        )
    else:
        estimates, nll = subject_model.fit(
            method=optmethod, n_runs=n_runs, seed=seed, verbose=verbose
        )
        bic = 2 * nll + np.log(subject_model.n_trials) * subject_model.n_parameters
        # Put fitting results into DataFrame row
        estimates_df = pd.DataFrame(
            dict(subject=subject, nll=nll, bic=bic, model=model_label), index=[subject]
        )
        for parameter, estimate in zip(subject_model.parameter_names, estimates):
            estimates_df[parameter] = estimate
        estimates_df.to_csv(estimates_output_path, index=False)

    # Prediction
    if not overwrite and exists(predictions_output_path):
        if verbose:
            print(
                "Found existing predictions at '{}'. Skipping prediction...".format(
                    predictions_output_path
                )
            )
        predictions_df = pd.read_csv(predictions_output_path)
    else:
        predictions_df = []
        predicted_choiceprobs = subject_model.predict_choiceprobs(parameters=estimates)
        for rep in range(n_sims):
            predicted_choices = subject_model.simulate_choices(parameters=estimates)
            predictions_rep_df = subject_model.data.copy()
            predictions_rep_df["predicted_choice"] = predicted_choices
            for i, option in enumerate(["A", "B", "C"]):
                predictions_rep_df[
                    f"predicted_choiceprob_{option}"
                ] = predicted_choiceprobs[:, i]
            predictions_rep_df["model"] = model_label
            predictions_rep_df["rep"] = rep
            predictions_df.append(predictions_rep_df)
        predictions_df = pd.concat(predictions_df)
        predictions_df.to_csv(predictions_output_path, index=False)
    return estimates_df, predictions_df


if __name__ == "__main__":
    modelFitting()
