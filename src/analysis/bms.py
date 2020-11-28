import numpy as np
import pymc3 as pm
import theano.tensor as tt


def run_bms(L, **sample_kwargs):
    """This function computes the exceedance probabilities (xp)
    and expected relative frequencies (r) from an array of log-evidences.
    
    Args:
        L (numpy.ndarray): Array of model log-evidences (higher is better fit).
            Array shape should be (K models; N subjects)

        **sample_kwargs: Additional arguments to the pymc.sample function.
            Currently `cores=1` seems to be necessary.
    
    Returns:
        dict: Dictionary with values xp and r.

    Reference:
        Stephan, K. E., Penny, W. D., Daunizeau, J., Moran, R. J., & Friston, K. J. (2009). Bayesian model selection for group studies. Neuroimage, 46(4), 1004-1017.
    """

    K, N = L.shape

    with pm.Model() as bms:

        def lookup_L(L, N):
            """This function looks up the log-evidences for all N subjects,
            given the current model labels m.
            """
            return L[tt.cast(m, dtype="int32"), tt.cast(tt.arange(N), dtype="int32")]

        # Priors
        alpha = pm.Uniform("alpha", 0, N, shape=K, testval=np.ones(K))

        # Model
        r = pm.Dirichlet("r", a=alpha, testval=np.ones(K) / K)
        m = pm.Categorical("m", p=r, shape=N, testval=0)

        # Look up log evidence
        ll = pm.DensityDist("ll", logp=lookup_L, observed=dict(L=L, N=N))

        # Sample
        trace = pm.sample(**sample_kwargs)

    # Build results
    result = {}
    result["summary"] = pm.summary(trace, var_names=["alpha", "r"])
    result["xp"] = np.array(
        [
            np.mean(trace.get_values("r")[:, k] == trace.get_values("r").max(axis=1))
            for k in range(K)
        ]
    )
    r_unscaled = np.array([np.mean(trace.get_values("r")[:, k]) for k in range(K)])
    result["r"] = r_unscaled / r_unscaled.sum()

    return result
