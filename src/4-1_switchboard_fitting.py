#!/usr/bin/python
"""
Gaze-dependent accumulation in context-dependent risky choice
  Switchboard Model Fitting

Author: Felix Molter, felixmolter@gmail.com
"""

import argparse
import itertools
import os
from functools import partial

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
    parser.add_argument("--seed", type=int, default=2020, help="Random number seed.")

    args = parser.parse_args()
    VERBOSE = args.verbose
    DATADIR = args.data_dir
    RESULTS_DIR = args.results_dir
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

    # Define model variants to be fit
    model = bm.switchboard.FullModel
    switches = dict(
        integration=["multiplicative", "additive"],
        comparison=["vsmean", "absolute"],
        attributeGazeBias=[False, True],
        alternativeGazeBias=[False, True],
        leak=["none", "free", "gaze-dependent"],
        inhibition=["none", "free", "distance-dependent", "gaze-dependent"],
    )

    # Load trial data
    trials = pd.read_csv(os.path.join(DATADIR, "trials.csv"), index_col=0)

    # Load fixation data
    fixations = pd.read_csv(os.path.join(DATADIR, "fixations.csv"), index_col=0)
    fixations["duration"] /= 1000

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

    # Iterate over subjects and fit all models
    subjects = np.unique(trials["subject"])
    n_subjects = len(subjects)

    def dict_product(dicts):
        """
        >>> list(dict_product(dict(number=[1,2], character='ab')))
        [{'character': 'a', 'number': 1},
        {'character': 'a', 'number': 2},
        {'character': 'b', 'number': 1},
        {'character': 'b', 'number': 2}]
        """
        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

    def input_generator(data, subjects, model, switches, results_dir="", verbose=False):
        "Generates subject-wise inputs to fit_subject_model function."
        for switchdict in dict_product(switches):
            if VERBOSE > 0:
                # Build a dummy model to get the right label to print...
                _ = model(data=data, switches=switchdict)
                print(f"{_.label}: Parameter estimation and choice prediction...")
                del _
            for subject in subjects:
                # Subset subject data
                subject_data = data[data["subject"] == subject].copy()
                # Initialize model instance
                subject_model = model(data=subject_data, switches=switchdict)
                subject_model.subject = subject
                # Yield everything as a tuple to input into fit_subject_model function
                yield subject, subject_model

    # Fix some arguments of the fitting function, so that it only takes subject data as argument
    fit_predict = partial(
        fit_predict_subject_model,
        n_runs=N_RUNS,
        optmethod=OPTMETHOD,
        n_sims=N_SIMS,
        results_dir=RESULTS_DIR,
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
            fit_predict,
            input_generator(trials, subjects, model, switches, RESULTS_DIR, VERBOSE),
        )
        all_estimates = [result[0] for result in results]
        all_predictions = [result[1] for result in results]
    # otherwise sequential
    else:
        all_estimates = []
        all_predictions = []
        for s, (subject, subject_model) in enumerate(
            input_generator(trials, subjects, model, switches, RESULTS_DIR, VERBOSE)
        ):
            if VERBOSE > 0:
                print("Subject {} ({}/{})".format(subject, s + 1, n_subjects))
            (estimates, predictions) = fit_predict((subject, subject_model))
            all_estimates.append(estimates)
            all_predictions.append(predictions)

    # Collect and save results
    all_estimates_output_path = os.path.join(
        RESULTS_DIR, "4-switchboard", "estimates", f"sb_estimates{LABEL}.csv"
    )
    all_estimates = pd.concat(all_estimates, sort=True)
    all_estimates.to_csv(all_estimates_output_path)
    all_predictions_output_path = os.path.join(
        RESULTS_DIR, "4-switchboard", "predictions", f"sb_predictions{LABEL}.csv"
    )
    all_predictions = pd.concat(all_predictions, sort=True)
    all_predictions.to_csv(all_predictions_output_path)


def fit_predict_subject_model(
    variable_input,
    n_runs=1,
    optmethod="differential_evolution",
    n_sims=1,
    results_dir="",
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
    estimates_output_folder = os.path.join(
        results_dir, "4-switchboard", "estimates", model_label
    )
    predictions_output_folder = os.path.join(
        results_dir, "4-switchboard", "predictions", model_label
    )
    if not os.path.exists(estimates_output_folder):
        os.makedirs(estimates_output_folder)
    if not os.path.exists(predictions_output_folder):
        os.makedirs(predictions_output_folder)

    # Create output filenames
    estimates_output_file = "estimates_{}_{}{}.csv".format(model_label, subject, label)
    estimates_output_path = os.path.join(estimates_output_folder, estimates_output_file)
    predictions_output_file = "predictions_{}_{}{}.csv".format(
        model_label, subject, label
    )
    predictions_output_path = os.path.join(
        predictions_output_folder, predictions_output_file
    )

    # Parameter estimation
    if not overwrite and os.path.exists(estimates_output_path):
        if verbose > 0:
            print(
                f"Found existing estimates at '{estimates_output_path}'. Skipping estimation..."
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
        for parameter, estimate in zip(subject_model.parameters.keys(), estimates):
            estimates_df[parameter] = estimate
        estimates_df.to_csv(estimates_output_path)

    # Prediction
    if not overwrite and os.path.exists(predictions_output_path):
        if verbose:
            print(
                f"Found existing predictions at '{predictions_output_path}'. Skipping prediction..."
            )
        predictions_df = pd.read_csv(predictions_output_path)
    else:
        predictions_df = []
        for rep in range(n_sims):
            predicted_choices = subject_model.simulate_choices(parameters=estimates)
            predictions_rep_df = subject_model.data.copy()
            predictions_rep_df["predicted_choice"] = predicted_choices
            predictions_rep_df["model"] = model_label
            predictions_rep_df["rep"] = rep
            predictions_df.append(predictions_rep_df)
        predictions_df = pd.concat(predictions_df)
        predictions_df.to_csv(predictions_output_path)
    return estimates_df, predictions_df


if __name__ == "__main__":
    modelFitting()
