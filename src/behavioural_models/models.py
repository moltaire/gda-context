#!usr/bin/python
import numpy as np
from scipy.optimize import minimize, differential_evolution
from .utils import choose, softmax


class ChoiceModel(object):
    """Base class for probabilistic choice models
    
    Contains methods shared across models to
        1) simulate choices (`simulate_choices`)
        2) compute negative log-likelihood (`compute_nll`)
        3) perform parameter estimation (`fit`)
    """

    def __init__(self):
        super(ChoiceModel, self).__init__()

    def simulate_choices(self, parameters):
        """For given parameters, predict choice probabilities and generate choices from them.
        """
        choices = choose(self.predict_choiceprobs(parameters))
        return choices

    def compute_nll(self, parameters, verbose=False, nonzeroconst=1e-6):
        """Compute negative log-likelihood of the data, given parameters.
        """
        choiceprobs = self.predict_choiceprobs(parameters)
        chosenprobs = choiceprobs[
            np.arange(choiceprobs.shape[0]).astype(int), self.choices.astype(int)
        ]
        nll = -np.sum(np.log(chosenprobs + nonzeroconst))
        if verbose > 1:
            print(
                "\t",
                "Subject",
                self.subject,
                "\t",
                *np.round(parameters, 2),
                "\tNLL",
                np.round(nll, 2)
            )
        return nll

    def fit(
        self, method="minimize", n_runs=1, seed=None, verbose=False, **method_kwargs
    ):
        """Estimate best fitting parameters using maximum log-likelihood.

        Parameters:
        -----------
        method : str, optional
            Optimization method to use. Must be one of ['minimize', 'differential_evolution'], defaults to 'minimize'.
        n_runs : int, optional
            Number of optimization runs. Should probably be more than 1 if method='minimize'. Defaults to 1.
        seed : int, optional
            Random seed. Defaults to no seed.
        verbose : int, optional
            Verbosity toggle. Prints some stuff if > 0. Prints more stuff if > 1... Defaults to 0.
        **method_kwargs : optional
            Additional keyword arguments to be passed on to the optimizer.

        Returns:
        -------
        tuple
            (maximum-likelihood estimates, minimum negative log-likelihood)
        """
        best_nll = np.inf
        best_x = np.zeros(self.n_parameters) * np.nan
        for run in range(n_runs):
            if verbose > 0:
                print(
                    "{}\tSubject {}\tRun {} of {} ({:.0f}%)".format(
                        self.label,
                        self.subject,
                        run + 1,
                        n_runs,
                        100 * (run + 1) / n_runs,
                    )
                )
            if seed is not None:
                np.random.seed(seed * self.subject + seed * run)
            if method == "minimize":
                x0 = [
                    np.random.uniform(*self.parameter_bounds[p])
                    for p in range(self.n_parameters)
                ]
                result = minimize(
                    self.compute_nll,
                    x0=x0,
                    bounds=self.parameter_bounds,
                    **method_kwargs
                )
            elif method == "differential_evolution":
                result = differential_evolution(
                    self.compute_nll, bounds=self.parameter_bounds, **method_kwargs
                )
            else:
                raise ValueError(
                    'Unknown method "{}". Use "minimize" or "differential_evolution".'.format(
                        method
                    )
                )
            if result.fun < best_nll:
                best_nll = result.fun
                best_x = result.x
        return best_x, best_nll


class ExpectedUtility(ChoiceModel):
    """Expected Utility model
    
    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["pA", "pB", "pC"],
        outcome_cols=["mA", "mB", "mC"],
        label="EU",
        parameter_names=["alpha", "beta"],
        parameter_bounds=[(0, 5), (0, 50)],
    ):
        super(ExpectedUtility, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, beta = parameters
        utilities = self.probabilities * self.outcomes ** alpha
        choiceprobs = softmax(beta * utilities)
        return choiceprobs


class ProspectTheory(ChoiceModel):
    """Prospect Theory model.
    Assumes that objective probabilities are transformed into decision weights (using weighting function with parameter $\gamma$), and outcome utilities are computed with a power-function with parameter $\alpha$. Choice probabilities are derived from subjective expected utilities via a softmax function with inverse temperature parameter $\beta$.
    
    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["pA", "pB", "pC"],
        outcome_cols=["mA", "mB", "mC"],
        label="PT",
        parameter_names=["alpha", "gamma", "beta"],
        parameter_bounds=[(0, 5), (0.28, 1), (0, 50)],
    ):
        super(ProspectTheory, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, gamma, beta = parameters
        p = self.probabilities
        w = p ** gamma / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        SU = w * self.outcomes ** alpha
        choiceprobs = softmax(beta * SU)
        return choiceprobs


class GazeBaselineStat(ChoiceModel):
    def __init__(
        self,
        data,
        gaze_cols=["dwell_A", "dwell_B", "dwell_C"],
        label="gaze-baseline-stat",
        parameter_names=["beta"],
        parameter_bounds=[(0, 50)],
    ):
        super(GazeBaselineStat, self).__init__()
        self.data = data
        self.gaze_cols = gaze_cols
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)
        self.n_items = len(gaze_cols)

    def predict_choiceprobs(self, parameters):

        beta = parameters[0]
        choiceprobs = softmax(beta * self.data[self.gaze_cols].values)
        return choiceprobs


class GazeBaselineDyn(ChoiceModel):
    """Second baseline model that only uses gaze data,
    but also uses the sequence and duration of fixations 
    and a leak parameter

    Parameters
    ----------
    beta (beta > 0)
        Inverse temperature parameter
    lambda (0 < lambda < 1)
        Leak parameter (1 = full retention, 0 = full leak)
    """

    def __init__(
        self,
        data,
        probability_cols=["pA", "pB", "pC"],
        outcome_cols=["mA", "mB", "mC"],
        label="gaze-baseline-dyn",
        parameter_names=["beta", "lam"],
        parameter_bounds=[(0, 50), (0, 1)],
    ):
        super(GazeBaselineDyn, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.fixated_alternatives = data["fixated_alternatives"].values  # 0 = p, 1 = m
        self.fixated_attributes = data["fixated_attributes"].values
        self.fixation_durations = data["fixation_durations"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)
        self.n_items = len(probability_cols)

    def predict_choiceprobs(self, parameters):

        beta, lam = parameters

        Y = np.zeros((self.n_trials, self.n_items))

        for trial in range(self.n_trials):

            # If fixation data present
            if isinstance(self.fixation_durations[trial], np.ndarray):
                for dur, alt, att in zip(
                    self.fixation_durations[trial],
                    self.fixated_alternatives[trial],
                    self.fixated_attributes[trial],
                ):
                    x = np.zeros(self.n_items)
                    x[alt] = 1
                    Y[trial, :] = (1 - lam) * Y[trial, :] + x

        choiceprobs = softmax(beta * Y)
        return choiceprobs


class Glickman1Layer(ChoiceModel):
    """Three alternative adaptation from the winning model from Glickman et al., 2019
    Assumes that in each fixation, gaze-biased subjective utilities (see PT) are accumulated and all accumulators (irrespective of fixation) are subject to leak over individual fixations.

    Parameters
    ----------
    alpha (alpha > 0)
        Utility function parameter
    gamma (0.28 < gamma < 1)
        Probability weighting parameter
    beta (beta > 0)
        Inverse temperature parameter
    lambda (0 < lambda < 1)
        Leak parameter (0 = perfect memory, 1 = full leak)
    theta (0 < theta < 1)
        Gaze bias parameter
    """

    def __init__(
        self,
        data,
        probability_cols=["pA", "pB", "pC"],
        outcome_cols=["mA", "mB", "mC"],
        label="glickman1layer",
        parameter_names=["alpha", "gamma", "beta", "lam", "theta"],
        parameter_bounds=[(0, 5), (0.2, 1), (0, 50), (0, 1), (0, 1)],
    ):
        super(Glickman1Layer, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.fixated_alternatives = data["fixated_alternatives"].values  # 0 = p, 1 = m
        self.fixated_attributes = data["fixated_attributes"].values
        self.fixation_durations = data["fixation_durations"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)
        self.n_items = len(probability_cols)

    def predict_choiceprobs(self, parameters):

        alpha, gamma, beta, lam, theta = parameters
        p = self.probabilities
        w = p ** gamma / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        SU = w * self.outcomes ** alpha
        Y = np.zeros((self.n_trials, self.n_items))

        for trial in range(self.n_trials):

            # If fixation data present
            if isinstance(self.fixation_durations[trial], np.ndarray):
                for dur, alt, att in zip(
                    self.fixation_durations[trial],
                    self.fixated_alternatives[trial],
                    self.fixated_attributes[trial],
                ):

                    # Option wise gaze discount
                    theta_vector = np.ones(self.n_items) * theta
                    theta_vector[alt] = 1.0

                    Y[trial, :] = (1 - lam) * Y[trial, :] + theta_vector * SU[trial, :]

        choiceprobs = softmax(beta * Y)
        return choiceprobs


class Glickman2Layer(ChoiceModel):
    """Three alternative adaption from 2-layer model from Glickman et al., 2019
    Also assumes that over fixations, subjective utilities (see PT) are accumulated. However, in contrast to the 1-layer model, here, the subjective stimulus attributes (decision weights and subjective utilities) also accumulate across fixations. The gaze-bias acts on the input to these lower-level accumulators (decision weights and subjective utilities), which are then combined *after the gaze bias was applied* in the next level. 
    Accumulators on both levels are subject to leak.

    For a reference, see Glickman et al., 2019 (Fig. 6A)
    
    Parameters
    ----------
    alpha (alpha > 0)
        Utility function parameter
    gamma (0.28 < gamma < 1)
        Probability weighting parameter
    beta (beta > 0)
        Inverse temperature parameter
    lambda (0 < lambda < 1)
        Leak parameter (0 = perfect memory, 1 = full leak)
    theta (0 < theta < 1)
        Gaze bias parameter
    """

    def __init__(
        self,
        data,
        probability_cols=["pA", "pB", "pC"],
        outcome_cols=["mA", "mB", "mC"],
        label="Glickman2Layer",
        parameter_names=["alpha", "gamma", "beta", "lam", "theta"],
        parameter_bounds=[(0, 5), (0.2, 1), (0, 50), (0, 1), (0, 1)],
    ):
        super(Glickman2Layer, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.fixated_alternatives = data["fixated_alternatives"].values  # 0 = p, 1 = m
        self.fixated_attributes = data["fixated_attributes"].values
        self.fixation_durations = data["fixation_durations"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)
        self.n_items = len(probability_cols)
        self.n_attributes = 2

    def predict_choiceprobs(self, parameters):

        alpha, gamma, beta, lam, theta = parameters
        p = self.probabilities
        w = p ** gamma / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        m = self.outcomes ** alpha
        L1w = np.zeros((self.n_trials, self.n_items))
        L1m = np.zeros((self.n_trials, self.n_items))
        L2 = np.zeros((self.n_trials, self.n_items))

        for trial in range(self.n_trials):

            # If fixation data present
            if isinstance(self.fixation_durations[trial], np.ndarray):
                for dur, alt, att in zip(
                    self.fixation_durations[trial],
                    self.fixated_alternatives[trial],
                    self.fixated_attributes[trial],
                ):

                    # AOI wise gaze discount
                    theta_vector = np.ones((self.n_items, self.n_attributes)) * theta
                    theta_vector[alt, att] = 1.0

                    L1w[trial, :] = (1 - lam) * L1w[trial, :] + theta_vector[:, 0] * w[
                        trial, :
                    ]
                    L1m[trial, :] = (1 - lam) * L1m[trial, :] + theta_vector[:, 1] * m[
                        trial, :
                    ]
                    L2[trial, :] = (1 - lam) * L2[trial, :] + L1w[trial, :] * L1m[
                        trial, :
                    ]

        choiceprobs = softmax(beta * L2)
        return choiceprobs
