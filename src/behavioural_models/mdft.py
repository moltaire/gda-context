import numpy as np
from scipy.stats import norm as normal
from scipy.integrate import quad
from .models import ChoiceModel


class MDFT(ChoiceModel):
    """docstring for MDFT"""
    def __init__(self, data=None,
                 attribute1_cols=['pA', 'pB', 'pC'],
                 attribute2_cols=['mA', 'mB', 'mC'],
                 label='MDFT',
                 parameter_names=['w', 'wd', 'phi1', 'phi2', 'sig2'],
                 parameter_bounds=[(0, 1), (1, 50), (0.01, 1000), (0, 1), (0, 1000)], # from Berkowitsch et al. 2014
                 log_transform=True,
                 normalize=True,
                 normalize_bounds=[0, 1]):
        super(MDFT, self).__init__()
        self.data = data
        self.choices = data['choice'].values
        self.attribute1_cols = attribute1_cols
        self.attribute2_cols = attribute2_cols
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)
        self.log_transform = log_transform
        self.normalize = normalize
        self.normalize_bounds = normalize_bounds
        self.stimuli = transform_stims(self.data,
                                       attribute1_cols=attribute1_cols,
                                       attribute2_cols=attribute2_cols,
                                       log_transform=log_transform,
                                       normalize=normalize,
                                       bounds=normalize_bounds)

    def predict_choiceprobs(self, parameters):
        n_trials, n_items, n_attributes = self.stimuli.shape
        w, wd, phi1, phi2, sig2 = parameters

        W = np.array([w, 1 - w])
        C = make_C(n_items)

        choiceprobs = np.zeros((n_trials, n_items))

        for trial in range(n_trials):
            m = self.stimuli[trial, :, :]
            try:
                s = make_S(m, phi1, phi2, w, wd)
                p_trial = np.zeros(n_items)
                eta, omega = dyn(s, C, m, W, sig2)
                # A
                L = np.array([[1, -1, 0], [1, 0, -1]])
                Leta = L @ eta
                m1 = Leta[0]
                m2 = Leta[1]
                Lomega = L @ omega @ L.T
                s1 = np.sqrt(Lomega[0, 0])
                s2 = np.sqrt(Lomega[1, 1])
                r = Lomega[1, 0] / (s1 * s2)
                p31 = thur3(m1, m2, s1, s2, r)

                # B
                L = np.array([[-1, 1, 0], [0, 1, -1]])
                Leta = L @ eta
                m1 = Leta[0]
                m2 = Leta[1]
                Lomega = L @ omega @ L.T
                s1 = np.sqrt(Lomega[0, 0])
                s2 = np.sqrt(Lomega[1, 1])
                r = Lomega[1, 0] / (s1 * s2)
                p32 = thur3(m1, m2, s1, s2, r)

                # C
                L = np.array([[-1, 0, 1], [0, -1, 1]])
                Leta = L @ eta
                m1 = Leta[0]
                m2 = Leta[1]
                Lomega = L @ omega @ L.T
                s1 = np.sqrt(Lomega[0, 0])
                s2 = np.sqrt(Lomega[1, 1])
                r = Lomega[1, 0] / (s1 * s2)
                p33 = thur3(m1, m2, s1, s2, r)
                p_trial = np.array([p31, p32, p33])
            except np.linalg.LinAlgError as e:
                p_trial = np.zeros(n_items)

            if np.any(~np.isfinite(p_trial)):
                p_trial = np.zeros(n_items)
            else:
                p_trial[p_trial <= 0] = 0
                p_trial = p_trial / np.sum(p_trial)
            choiceprobs[trial, :] = p_trial
        return choiceprobs


def make_C(n_alternatives):
    C = np.ones((n_alternatives, n_alternatives)) * -1/(n_alternatives-1)
    np.fill_diagonal(C, 1.)
    return C


def make_S(m, phi1, phi2, w, wd):
    """
    m: single trial stimuli
    phi1: distance/inhibition parameter
    phi2: decay parameter (0 = no decay)
    w: first attribute weight
    wd: dominance overweighting parameter
    """
    n_items = m.shape[0]
    W = np.array([w, 1.-w])

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

    dists_std = (m[:, np.newaxis] - m)
    dists_trns = dists_std @ B_inv

    D = np.sum((dists_trns**2 @ A), axis=2)

    D_psy = np.eye(n_items) - phi2 * np.exp(-phi1 * D**2)

    return D_psy


def dyn(S, C, m, W, sig2, t=250):
    """
    there is a lot of weirdness in this function, all of which is courtesy of JB
    """
    n_items = S.shape[0]
    mu = C @ m @ W
    psi = np.diag(W) - np.outer(W, W.T)

    eta = mu
    sig1 = 1

    Phi = (sig1 * C  @ m @ psi @ m.T @ C.T) + (sig2 * C @ np.eye(n_items) @ C.T)

    # second term is initial state variability
    omega = Phi + 0 * np.eye(n_items)
    si = np.eye(n_items)

    for i in range(2, t+1):
        rt = 1. / (1 + np.exp((i - 202)) / 25)  # lolwhat
        si = S @ si  # does not do anything?
        eta = eta + rt * si @ mu
        omega = omega + (rt**2) * si @ Phi @ si.T

    return eta, omega


def thur3(m1, m2, s1, s2, r):
    """
    "Compute choice probabilities for 3-alternative thurstone model."
        What are the inputs?
        Returns choice probability p.
    """
    z1 = m1 / s1
    z2 = m2 / s2

    p1 = normal.cdf(z1) * normal.cdf(z2)
    p2, abserror = quad(Fc, 0, r, args=(m1, m2, s1, s2))
    p = p1+p2
    return p


def Fc(t, m1, m2, s1, s2):
    """
    "Step in 3 alternative chocie"
    What are the inputs?
    What are the outputs?
    This is being integrated over, when computing choice probabilities.
    """
    z1 = m1 / s1
    z2 = m2 / s2

    x = z1**2 - 2*t*z1*z2 + z2**2
    x = x / (2 * (1 - t**2))
    x = np.exp(-x) / (2 * np.pi * np.sqrt(1 - t**2))
    return x


def transform_stims(data, attribute1_cols, attribute2_cols, log_transform=True, normalize=True, bounds=(0, 10)):
    if data is None:
        return None
    else:
        stimuli = np.stack([data[attribute1_cols].values, data[attribute2_cols]], axis=-1)
        if log_transform:
            stimuli = np.log(stimuli)
        if normalize:
            minimum, maximum = bounds
            n_trials, n_items, n_attributes = stimuli.shape
            for attribute in range(n_attributes):
                attribute_min = stimuli[:, :, attribute].min()
                attribute_max = stimuli[:, :, attribute].max()
                stimuli[:, :, attribute] = minimum + ((stimuli[:, :, attribute] - attribute_min) * (maximum - minimum)) / (attribute_max - attribute_min)
        return stimuli