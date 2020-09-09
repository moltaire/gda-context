#!usr/bin/python

# This module contains code to run BEST models
# John K. Kruschke, Journal of Experimental Psychology: General, 2013, v.142(2), pp.573-603. (doi: 10.1037/a0029146)
# Code is in part adapted from the PyMC3 implementation available at
# https://docs.pymc.io/notebooks/BEST.html

import numpy as np
import pymc3 as pm


def runBEST(y1, y2, sigma_low=1.0, sigma_high=10.0, sample_kwargs={"cores": 1}):
    """Run two-sample BEST model
    
    Args:
        y1 (array): Group 1 values
        y2 (array): Group 2 values
        sigma_low (float, optional): Lower bound of uniform prior on group standard deviation. Defaults to 1.0.
        sigma_high (float, optional): Upper bound of uniform prior on group. Defaults to 10.0.
        sample_kwargs : dict, optional
        additional keyword arguments passed on to pymc3.sample
    
    Returns:
        pymc3.MultiChain: MCMC trace from BEST model.
    """
    y = np.concatenate([y1, y2])
    mu_m = y.mean()
    mu_sd = y.std() * 2

    with pm.Model() as BEST:
        # Priors
        group1_mean = pm.Normal("group1_mean", mu=mu_m, sd=mu_sd)
        group2_mean = pm.Normal("group2_mean", mu=mu_m, sd=mu_sd)
        group1_sd = pm.Uniform("group1_sd", lower=sigma_low, upper=sigma_high)
        group2_sd = pm.Uniform("group2_sd", lower=sigma_low, upper=sigma_high)
        nu = pm.Exponential("nu_minus_one", 1.0 / 29.0) + 1.0

        # Deterministics
        lam1 = group1_sd ** -2
        lam2 = group2_sd ** -2
        diff_of_means = pm.Deterministic("diff_of_means", group1_mean - group2_mean)
        diff_of_sds = pm.Deterministic("diff_of_sds", group1_sd - group2_sd)
        pooled_sd = np.sqrt((group1_sd ** 2 + group2_sd ** 2) / 2)
        effect_size = pm.Deterministic("d", diff_of_means / pooled_sd)

        # Likelihood
        group1 = pm.StudentT("group1", nu=nu, mu=group1_mean, lam=lam1, observed=y1)
        group2 = pm.StudentT("group2", nu=nu, mu=group2_mean, lam=lam2, observed=y2)

        # MCMC
        trace = pm.sample(**sample_kwargs)

    return trace


def runBEST1G(y, mu=0.0, sigma_low=1.0, sigma_high=10.0, sample_kwargs={"cores": 1}):
    """Run one-sample BEST model.
    
    Args:
        y (array): Values
        mu (float, optional): Population mean. Defaults to 0.0.
        sigma_low (float, optional): Lower bound of uniform prior on group standard deviation. Defaults to 1.0.
        sigma_high (float, optional): Upper bound of uniform prior on group. Defaults to 10.0.
        sample_kwargs : dict, optional
            additional keyword arguments passed on to pymc3.sample
    
    Returns:
        pymc3.MultiChain: MCMC trace from BEST model.
    """
    with pm.Model() as BEST1G:
        # Priors
        mean = pm.Normal("mean", mu=y.mean(), sd=y.std() * 2)
        sd = pm.Uniform("sd", lower=sigma_low, upper=sigma_high)
        nu = pm.Exponential("nu_minus_one", 1.0 / 29.0) + 1.0

        # Deterministics
        lam = sd ** -2
        difference = pm.Deterministic("difference", mean - mu)
        effect_size = pm.Deterministic("d", (mean - mu) / sd)

        # Likelihood
        observed = pm.StudentT("observed", nu=nu, mu=mean, lam=lam, observed=y)

        # MCMC
        trace = pm.sample(**sample_kwargs)

    return trace
