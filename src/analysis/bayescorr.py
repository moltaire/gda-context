#!usr/bin/python

# Compute Bayesian correlation analysis
# Implementing the model from Lee & Wagenmakers (2014)
# Following junpenglaos implementation at
# https://github.com/pymc-devs/resources/blob/master/BCM/ParameterEstimation/DataAnalysis.ipynb

import numpy as np
import pymc3 as pm
import theano.tensor as tt


def runBayesCorr(y1, y2, seed=None, sample_kwargs={}):
    """
    Run Bayesian correlation analysis.

    Implementation from Lee & Wagenmakers textbook.

    Arguments:
    ----------
    y1 array like
    y2 array like
    seed : int, optional
        random seed passed on to pymc3.sample
    sample_kwargs : dict, optional
        additional keyword arguments passed on to pymc3.sample

    Returns:
    --------
    pymc3.trace
    """
    y = np.stack([y1, y2], axis=1)

    with pm.Model() as corrModel:

        # Priors
        r = pm.Uniform("r", lower=-1.0, upper=1.0)
        mu = pm.Normal("mu", mu=0, tau=0.001, shape=2)

        lambda1 = pm.Gamma("lambda1", alpha=0.001, beta=0.001)
        lambda2 = pm.Gamma("lambda2", alpha=0.001, beta=0.001)
        sigma1 = pm.Deterministic("sigma1", 1 / np.sqrt(lambda1))
        sigma2 = pm.Deterministic("sigma2", 1 / np.sqrt(lambda2))

        cov = pm.Deterministic(
            "cov",
            tt.stacklists(
                [
                    [lambda1 ** -1, r * sigma1 * sigma2],
                    [r * sigma1 * sigma2, lambda2 ** -1],
                ]
            ),
        )
        # Likelihood
        Y = pm.MvNormal("Y", mu=mu, cov=cov, shape=2, observed=y)

        # MCMC
        trace = pm.sample(random_seed=seed, **sample_kwargs)

    return trace
