#!usr/bin/python
import numpy as np
import pandas as pd
from behavioural_models.utils import softmax, choose
from scipy.optimize import minimize, differential_evolution


class ChoiceModel(object):
    """Base class for probabilistic choice models"""

    def __init__(self):
        super(ChoiceModel, self).__init__()

    def simulate_choices(self, parameters):
        "Predict choice probabilities and generate choices from them."
        choices = choose(self.predict_choiceprobs(parameters))
        return choices

    def compute_nll(self, parameters, verbose=99, nonzeroconst=1e-6):
        "Compute negative log-likelihood of the data, given parameters."
        choiceprobs = self.predict_choiceprobs(parameters)
        chosenprobs = choiceprobs[
            np.arange(choiceprobs.shape[0]).astype(int), self.choices.astype(int)
        ]
        chosenprobs[np.isnan(chosenprobs)] = 0
        nll = -np.sum(np.log(chosenprobs + nonzeroconst))
        if np.isnan(nll):
            nll = np.inf
        if verbose:
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
        "Estimate best fitting parameters using maximum log-likelihood."
        best_nll = np.inf
        best_x = np.zeros(self.n_parameters) * np.nan
        for run in range(n_runs):
            if verbose > -1:
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
            bounds = [self.parameters[p]["bounds"] for p in self.parameter_names]
            if method == "minimize":
                x0 = [
                    np.random.uniform(*self.parameters[p]["bounds"])
                    for p in self.parameter_names
                ]
                result = minimize(
                    self.compute_nll, x0=x0, bounds=bounds, **method_kwargs
                )
            elif method == "differential_evolution":
                result = differential_evolution(
                    self.compute_nll, bounds=bounds, **method_kwargs
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


class FullModel(ChoiceModel):
    """
    The parent class for all switchboard model variants.
    """

    def __init__(
        self,
        data,
        probability_cols=["pA", "pB", "pC"],
        outcome_cols=["mA", "mB", "mC"],
        switches=dict(
            integration="multiplicative",  # multiplicative / additive
            comparison="vsmean",  # vsmean / absolute
            attributeGazeBias=False,  # True / False
            alternativeGazeBias=False,  # True / False,
            leak="none",  # none, free, gaze-dependent
            inhibition="none",  # none, free, gaze-dependent, distance-dependent
        ),
    ):
        super(FullModel, self).__init__()
        self.data = data
        self.n_trials = len(data)
        self.n_items = len(probability_cols)
        self.n_attributes = 2
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.fixated_alternatives = data["fixated_alternatives"].values
        self.fixated_attributes = data["fixated_attributes"].values  # 0 = p, 1 = m
        self.fixation_durations = data["fixation_durations"].values

        # Switches

        if switches["comparison"] == "vsmean":
            if (switches["leak"] in ["none", "free"]) and (
                switches["inhibition"] in ["none", "free"]
            ):
                print(
                    "/!\ Item-vs-mean comparison is equivalent to absolute comparison when combined with a softmax function, if feedback matrix is constant."
                )

        self.switches = switches
        self.label = "sb_int-{integration}_comp-{comparison}_gbatt-{attGazeBias}_gbalt-{altGazeBias}_lk-{leak}_inh-{inhibition}".format(
            integration=switches["integration"],
            comparison=switches["comparison"],
            attGazeBias=switches["attributeGazeBias"],
            altGazeBias=switches["alternativeGazeBias"],
            leak=switches["leak"],
            inhibition=switches["inhibition"],
        )

        # Parameters
        parameters = {}

        if switches["integration"] == "additive":
            parameters["w_p"] = dict(bounds=[0, 1])
        elif switches["integration"] == "multiplicative":
            parameters["alpha"] = dict(bounds=[0, 3])
            parameters["gamma"] = dict(bounds=[0.27, 1])

        if switches["attributeGazeBias"]:
            parameters["eta"] = dict(bounds=[0, 1])
        if switches["alternativeGazeBias"]:
            parameters["theta"] = dict(bounds=[0, 1])

        if switches["leak"] in ["free", "gaze-dependent"]:
            parameters["lam"] = dict(
                bounds=[0, 1]
            )  # 0 = full leak, 1 = no leak, perfect memory, note this is opposite to how it's coded in the GLA

        if switches["inhibition"] in ["free", "gaze-dependent"]:
            parameters["phi"] = dict(
                bounds=[0, 1]
            )  # 0 = no inhibition, 1 = maximum inhibition, e.g., accumulator A with value 5 inhibits all other accumulators with -5)
        elif switches["inhibition"] == "distance-dependent":
            parameters["phi"] = dict(bounds=[0, 1])
            parameters["wd"] = dict(bounds=[1, 50])
            parameters["w_p"] = dict(
                bounds=[0, 1]
            )  # this might already be here, if integration == 'additive'
            self.stimuli = transform_stims(
                self.data,
                attribute1_cols=["pA", "pB", "pC"],
                attribute2_cols=["mA", "mB", "mC"],
                log_transform=True,
                normalize=True,
                bounds=(0, 1),
            )

        if switches["comparison"] == "vsmean":
            C = np.ones((self.n_items, self.n_items)) * (-1 / (self.n_items - 1))
            np.fill_diagonal(C, 1)
        elif switches["comparison"] == "absolute":
            C = np.eye(self.n_items)
        else:
            raise ValueError(
                'Invalid argument "{}" for comparison switch'.format(
                    switches["comparison"]
                )
            )
        self.C = C

        # Softmax parameter
        parameters["beta"] = dict(bounds=[0, 50])

        self.parameters = parameters
        self.parameter_names = list(parameters.keys())
        self.n_parameters = len(parameters.keys())

    def predict_choiceprobs(self, parameters):

        param_dict = {
            param: parameters[i] for i, param in enumerate(self.parameter_names)
        }

        # Attribute processing
        p = self.probabilities
        m = self.outcomes

        if self.switches["integration"] == "multiplicative":
            gamma = param_dict["gamma"]
            p = p ** gamma / (p ** gamma + (1 - p) ** gamma) ** (
                1 / gamma
            )  # https://faculty.psy.ohio-state.edu/myung/personal/JRUApr2013.pdf

            V = np.stack([p, m], axis=-1)
        else:
            # divisive normalization
            p_n = p / p.sum(axis=1, keepdims=True)
            m_n = m / m.sum(axis=1, keepdims=True)
            V = np.stack(
                [p_n, m_n], axis=-1
            )  # stack them      (n_trials x n_items x n_attributes)

        # Initialize storage
        A = np.zeros((self.n_trials, self.n_items))  # absolute

        for trial in range(self.n_trials):

            # If fixation data present
            if isinstance(self.fixation_durations[trial], np.ndarray):
                for dur, alt, att in zip(
                    self.fixation_durations[trial],
                    self.fixated_alternatives[trial],
                    self.fixated_attributes[trial],
                ):

                    # Gaze bias
                    if self.switches["attributeGazeBias"]:
                        eta_v = np.ones(self.n_attributes) * param_dict["eta"]
                        eta_v[att] = 1.0
                    else:
                        eta_v = np.ones(self.n_attributes)
                    if self.switches["alternativeGazeBias"]:
                        theta_v = np.ones(self.n_items) * param_dict["theta"]
                        theta_v[alt] = 1.0
                    else:
                        theta_v = np.ones(self.n_items)

                    # Leak
                    if self.switches["leak"] == "none":
                        lam_v = np.ones(self.n_items)
                    elif self.switches["leak"] == "free":
                        lam_v = np.ones(self.n_items) * param_dict["lam"]
                    elif self.switches["leak"] == "gaze-dependent":
                        lam_v = np.ones(self.n_items) * param_dict["lam"]
                        lam_v[alt] = 1.0
                    else:
                        raise ValueError(
                            'Did not understand leak argument "{}".'.format(
                                self.switches["leak"]
                            )
                        )

                    # Inhibition
                    if self.switches["inhibition"] == "none":
                        inh = np.zeros((self.n_items, self.n_items))
                    elif self.switches["inhibition"] == "free":
                        inh = (
                            np.ones((self.n_items, self.n_items))
                            * (-1)
                            * param_dict["phi"]
                        )
                        np.fill_diagonal(inh, 0)
                    elif self.switches["inhibition"] == "distance-dependent":
                        inh = distance_feedback(
                            self.stimuli[trial],
                            phi1=param_dict["phi"],
                            w=param_dict["w_p"],
                            wd=param_dict["wd"],
                        )
                    elif self.switches["inhibition"] == "gaze-dependent":
                        inh = np.zeros((self.n_items, self.n_items))
                        inh[:, alt] = (-1) * param_dict["phi"]
                        np.fill_diagonal(inh, 0)
                    else:
                        raise ValueError(
                            'Did not understand inhibition argument "{}".'.format(
                                self.switches["inhibition"]
                            )
                        )

                    if self.switches["integration"] == "multiplicative":
                        v = (
                            theta_v
                            * (eta_v[0] * V[trial, :, 0])
                            * (eta_v[1] * V[trial, :, 1]) ** param_dict["alpha"]
                        )
                    else:
                        w = np.array([param_dict["w_p"], 1 - param_dict["w_p"]])
                        v = np.sum(w * eta_v * theta_v[:, None] * V[trial], axis=1)

                    # Combine leak and inhibition
                    S = np.diag(lam_v) + inh

                    # Accumulation with inhibition and leak
                    try:
                        A[trial, :] = S.dot(A[trial, :]) + self.C.dot(v)
                    except Exception as e:
                        print("switches:")
                        print(self.switches)
                        print("")
                        print("A[trial, :]:", A[trial, :])
                        print("S:", S)
                        print("v:", v)
                        raise e

        choiceprobs = softmax(param_dict["beta"] * A)

        return choiceprobs


def distance_feedback(m, phi1, w, wd):
    """
    m: single trial stimuli
    phi1: distance/inhibition parameter
    w: first attribute weight
    wd: dominance overweighting parameter
    """
    n_items = m.shape[0]
    W = np.array([w, 1.0 - w])

    # indifference vector
    iv = np.array([-W[1] / W[0], 1])
    # dominance vector
    dv = W / W[0]
    # basis B
    B_star = np.vstack([iv, dv])
    # normalize
    B = B_star / np.linalg.norm(B_star, axis=0)

    # initialize distance matrix
    D = np.ones((n_items, n_items)) * np.nan

    B_inv = np.linalg.inv(B)

    A = np.diag([1, wd])

    dists_std = m[:, np.newaxis] - m
    dists_trns = dists_std @ B_inv

    D = np.sum((dists_trns ** 2 @ A), axis=2)

    feedback_matrix = np.eye(n_items) - np.exp(
        -(1 / phi1) * D ** 2
    )  # 0s on diagonal, negative values off diagonal. larger absolute values for larger distances. steeper dropoff and lower inhibition for smaller phi1

    return feedback_matrix


def transform_stims(
    data,
    attribute1_cols,
    attribute2_cols,
    log_transform=True,
    normalize=True,
    bounds=(0, 10),
):
    if data is None:
        return None
    else:
        stimuli = np.stack(
            [data[attribute1_cols].values, data[attribute2_cols]], axis=-1
        )
        if log_transform:
            stimuli = np.log(stimuli)
        if normalize:
            minimum, maximum = bounds
            n_trials, n_items, n_attributes = stimuli.shape
            for attribute in range(n_attributes):
                attribute_min = stimuli[:, :, attribute].min()
                attribute_max = stimuli[:, :, attribute].max()
                stimuli[:, :, attribute] = minimum + (
                    (stimuli[:, :, attribute] - attribute_min) * (maximum - minimum)
                ) / (attribute_max - attribute_min)
        return stimuli
