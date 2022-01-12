#!/usr/bin/python
"""
Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour
This script performs parameter estimation for parameter- and model-recovery analyses
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
    # parser.add_argument(
    #     "--data-dir",
    #     type=str,
    #     default="../results/0-clean_data",
    #     help="Relative path to preprocessed data directory.",
    # )
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
    # parser.add_argument(
    #     "--nsims",
    #     type=int,
    #     default=1,
    #     help="Number of simulation repetitions per trial.",
    # )
    parser.add_argument("--seed", type=int, default=2019, help="Random number seed.")

    args = parser.parse_args()
    VERBOSE = args.verbose
    RESULTS_DIR = args.results_dir
    OUTPUT_DIR = join(RESULTS_DIR, "S_recoveries")
    LABEL = args.label
    if LABEL != "":
        LABEL = "_" + LABEL
    N_CORES = args.ncores
    N_RUNS = args.nruns
    OPTMETHOD = args.optmethod
    OVERWRITE = args.overwrite
    SEED = args.seed

    np.random.seed(SEED)

    # Define models to be fit
    models = [
        bm.models.ExpectedUtility,
        bm.models.ProspectTheory,
        bm.mdft.MDFT,
        bm.models.GazeBaselineStat,
        bm.models.GazeBaselineDyn,
        bm.models.Glickman1Layer,
    ]

    # Load trial data
    trials = pd.read_csv(
        join("..", "results", "S_recoveries", "trials_synth.csv"), index_col=0
    )

    # Fix eye tracking columns which have arrays in them
    trials["fixated_alternatives"] = trials["fixated_alternatives"].apply(
        lambda x: np.fromstring(x[1:-1], sep=" ", dtype=int)
    )

    trials["fixated_attributes"] = trials["fixated_attributes"].apply(
        lambda x: np.fromstring(x[1:-1], sep=" ", dtype=int)
    )

    trials["fixation_durations"] = trials["fixation_durations"].apply(
        lambda x: np.fromstring(x[1:-1], sep=" ", dtype=float)
    )

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
                if not isinstance(subject, int):
                    subject_model.subject = int(subject.split("-")[0])
                else:
                    subject_model.subject = subject
                # Yield everything as a tuple to input into fit_subject_model function
                yield subject, subject_model

    # Fix some arguments of the fitting function, so that it only takes subject data as argument
    fit = partial(
        fit_subject_model,
        n_runs=N_RUNS,
        optmethod=OPTMETHOD,
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
        all_estimates = pool.map(
            fit, input_generator(trials, subjects, models, VERBOSE)
        )
        # all_estimates = [result[0] for result in results]

    # otherwise sequential
    else:
        all_estimates = []
        for s, (subject, subject_model) in enumerate(
            input_generator(trials, subjects, models, VERBOSE)
        ):
            if VERBOSE > 0:
                print("Subject {} ({}/{})".format(subject, s + 1, n_subjects))
            (estimates, predictions) = fit((subject, subject_model))
            all_estimates.append(estimates)

    # Collect and save results
    all_estimates_output_path = join(OUTPUT_DIR, "estimates", "estimates{}.csv").format(
        LABEL
    )
    all_estimates = pd.concat(all_estimates, sort=True)
    all_estimates.to_csv(all_estimates_output_path, index=False)


def fit_subject_model(
    variable_input,
    n_runs=1,
    optmethod="minimize",
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
    if not exists(estimates_output_folder):
        makedirs(estimates_output_folder)

    # Create output filenames
    estimates_output_file = "estimates_{}_{}{}.csv".format(model_label, subject, label)
    estimates_output_path = join(estimates_output_folder, estimates_output_file)

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

    return estimates_df


if __name__ == "__main__":
    modelFitting()
