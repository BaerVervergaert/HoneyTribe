import numpy as np
from honeytribe.metrics.centrality import mirror_centrality, mean, quantile
from honeytribe.metrics.utils import _name_function

def mirror_variability(a, mirror_transform, inverse_transform, centrality_measure, variance_centrality_measure, bias_correction=None, **kwargs):
    midpoint = centrality_measure(a, **kwargs)
    return mirror_centrality(
        np.abs(a - midpoint),
        mirror_transform,
        inverse_transform,
        variance_centrality_measure,
        bias_correction=bias_correction,
        **kwargs
    )

@_name_function('std')
def std(a, sample_weight=None):
    if len(a) < 2:
        raise ValueError('std requires at least two observations (n>=2) for unbiased estimation.')
    return mirror_variability(
        a,
        mirror_transform=lambda x: x**2,
        inverse_transform=np.sqrt,
        centrality_measure=mean,
        variance_centrality_measure=mean,
        bias_correction= lambda a: len(a) / (len(a) - 1),
        sample_weight = sample_weight,
    )

def quantile_spread(a, q0, q1 = None, sample_weight=None):
    """Calculates the spread between given coverage or quantiles.

    Parameters
    ----------
    a: sequence
    q0: lower-quantile, or coverage if q1 is not given.
    q1: (default: None) upper-quantile, if not given, then q0 is assumed to be the coverage instead of the lower-quantile.

    Returns
    -------
    float: spread
    """
    if q1 is None:
        q0 = (1 - q0)/2
        q1 = 1 - q0
    val_0 = quantile(a, q0, sample_weight=sample_weight)
    val_1 = quantile(a, q1, sample_weight=sample_weight)
    return val_1 - val_0

@_name_function('iqr')
def iqr(a, sample_weight=None):
    return quantile_spread(a, 0.25, 0.75, sample_weight=sample_weight)
