import numpy as np
from honeytribe.metrics.centrality import mirror_centrality, mean, quantile, median
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

@_name_function('variance')
def variance(a, sample_weight=None):
    """Compute the variance (std squared).

    Parameters:
        a: array-like input
        sample_weight: optional sample weights

    Returns:
        float: Variance (unbiased)
    """
    return std(a, sample_weight=sample_weight) ** 2


@_name_function('mad')
def mad(a, sample_weight=None):
    """Compute Mean Absolute Deviation around the mean.

    More robust than std for detecting variability.

    Parameters:
        a: array-like input
        sample_weight: optional sample weights

    Returns:
        float: Mean absolute deviation
    """
    return mirror_variability(
        a,
        mirror_transform=lambda x: x,
        inverse_transform=lambda x: x,
        centrality_measure=mean,
        variance_centrality_measure=mean,
        sample_weight=sample_weight,
    )


@_name_function('median_absolute_deviation')
def median_absolute_deviation(a, sample_weight=None):
    """Compute Median Absolute Deviation around the median.

    Extremely robust to outliers. Note: standard MAD uses scale factor 1.4826
    to make it consistent with std for normal distributions.

    Parameters:
        a: array-like input
        sample_weight: optional sample weights (used for weighted median)

    Returns:
        float: Median absolute deviation
    """
    from honeytribe.metrics.centrality import median
    return mirror_variability(
        a,
        mirror_transform=lambda x: x,
        inverse_transform=lambda x: x,
        centrality_measure=median,
        variance_centrality_measure=median,
        sample_weight=sample_weight,
    )


@_name_function('coefficient_of_variation')
def coefficient_of_variation(a, sample_weight=None):
    """Compute Coefficient of Variation: std / mean.

    Normalized measure for comparing variability across different scales.

    Parameters:
        a: array-like input
        sample_weight: optional sample weights

    Returns:
        float: Coefficient of variation
    """
    a = np.asarray(a)
    mu = mean(a, sample_weight=sample_weight)
    if mu == 0:
        raise ValueError('Coefficient of Variation undefined for zero mean')
    s = std(a, sample_weight=sample_weight)
    return s / abs(mu)


@_name_function('data_range')
def data_range(a, sample_weight=None):
    """Compute the range: max - min.

    Simple spread measure. Ignores sample_weight.

    Parameters:
        a: array-like input
        sample_weight: ignored

    Returns:
        float: Range
    """
    return np.max(a) - np.min(a)


@_name_function('decile_range')
def decile_range(a, sample_weight=None):
    """Compute decile range: quantile(0.9) - quantile(0.1).

    More robust than full range by excluding extreme tails.

    Parameters:
        a: array-like input
        sample_weight: optional sample weights

    Returns:
        float: Decile range
    """
    return quantile_spread(a, 0.1, 0.9, sample_weight=sample_weight)


@_name_function('semi_interquartile_range')
def semi_interquartile_range(a, sample_weight=None):
    """Compute Semi-Interquartile Range: (Q3 - Q1) / 2.

    Half of IQR; scale-independent measure related to midhinge.

    Parameters:
        a: array-like input
        sample_weight: optional sample weights

    Returns:
        float: Semi-interquartile range
    """
    return iqr(a, sample_weight=sample_weight) / 2.0


@_name_function('normalized_iqr')
def normalized_iqr(a, sample_weight=None):
    """Compute normalized IQR: IQR / median.

    Scale-independent dispersion measure.

    Parameters:
        a: array-like input
        sample_weight: optional sample weights

    Returns:
        float: Normalized IQR
    """
    iqr_val = iqr(a, sample_weight=sample_weight)
    med = median(a, sample_weight=sample_weight)
    if med == 0:
        raise ValueError('Normalized IQR undefined for zero median')
    return iqr_val / abs(med)


@_name_function('gini_coefficient')
def gini_coefficient(a, sample_weight=None):
    """Compute Gini coefficient: measure of statistical dispersion/inequality.

    Ranges from 0 (perfect equality) to 1 (perfect inequality).
    Based on Lorenz curve; measures relative mean absolute difference.

    Parameters:
        a: array-like input (should be non-negative for meaningful interpretation)
        sample_weight: optional sample weights

    Returns:
        float: Gini coefficient
    """
    a = np.asarray(a)
    if sample_weight is not None:
        raise NotImplementedError('Weighted Gini coefficient not implemented yet.')
    else:
        # Unweighted Gini: sum(|a_i - a_j|) / (2 * n^2 * mean(a))
        n = int(len(a))
        idx = np.argsort(a)
        a_sorted = a[idx]
        cumulative_sum = 0.0
        for i in range(n):
            cumulative_sum += a_sorted[i] * (i + 1)
        return (2 * cumulative_sum) / (n * np.sum(a_sorted)) - (n + 1) / n
