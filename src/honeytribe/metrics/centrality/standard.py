import numpy as np
import pandas as pd
from math import ceil

from honeytribe.metrics.utils import _name_function


@_name_function('mean')
def mean(a, sample_weight=None):
    val = a.copy()
    if sample_weight is not None:
        N = len(a)
        norm = np.sum(sample_weight) / N
        val = val * sample_weight / norm
    return np.mean(val)

@_name_function('median')
def median(a, sample_weight=None):
    q = 0.5
    return quantile(a, q, sample_weight)

@_name_function('lower_quartile')
def lower_quartile(a, sample_weight=None):
    q = 0.25
    return quantile(a, q, sample_weight)

@_name_function('upper_quartile')
def upper_quartile(a, sample_weight=None):
    q = 0.75
    return quantile(a, q, sample_weight)

@_name_function('min')
def min(a, sample_weight=None):
    return np.min(a)

@_name_function('max')
def max(a, sample_weight=None):
    return np.max(a)

def quantile(a, q: float, sample_weight = None):
    idx = np.argsort(a)
    val = a[idx]
    q = float(q)
    q = 0.0 if q < 0 else (1.0 if q > 1 else q)
    N = len(a)
    if N == 0:
        raise ValueError('quantile of empty array')
    if sample_weight is not None:
        w = np.asarray(sample_weight)[idx]
        if np.any(w < 0):
            raise ValueError('sample_weight must be non-negative')
        if np.all(w == 0):
            # fall back to unweighted
            raise ValueError('sample_weight cannot be all zero')
        else:
            c = np.cumsum(w) / np.sum(w)
            i = np.searchsorted(c, q, side='left')
            if i == 0:
                return val[0]
            if i >= N:
                return val[-1]
            c0 = c[i-1]
            c1 = c[i]
            if c1 == c0:
                return val[i]
            t = (q - c0) / (c1 - c0)
            return (1 - t) * val[i-1] + t * val[i]
    if N == 1:
        return val[0]
    k = q * (N - 1)
    i0 = int(np.floor(k))
    i1 = int(np.ceil(k))
    if i0 == i1:
        return val[i0]
    t = k - i0
    return (1 - t) * val[i0] + t * val[i1]

@_name_function('mode')
def mode(a, sample_weight=None):
    df = pd.DataFrame(data = {'val':a})
    if sample_weight is not None:
        df['sample_weight'] = sample_weight / np.sum(sample_weight)
    else:
        df['sample_weight'] = 1 / len(df)
    return df.groupby('val').sample_weight.sum().idxmax()

def mirror_centrality(a, mirror_transform, inverse_transform, centrality_measure, bias_correction=None, **kwargs):
    if bias_correction is None:
        bias_correction = lambda a: 1.
    return inverse_transform(bias_correction(a) * centrality_measure(mirror_transform(a), **kwargs))

@_name_function('geometric_mean')
def geometric_mean(a, sample_weight=None):
    return mirror_centrality(
        a,
        np.log,
        np.exp,
        mean,
        sample_weight = sample_weight,
    )


def trimmed_centrality(a, centrality_measure, trim=0.1, sample_weight=None):
    """Compute a trimmed centrality measure by removing extreme values.

    Parameters:
        a: array-like input
        centrality_measure: function to compute centrality (e.g., mean, median)
        trim: fraction to remove from each tail (default 0.1 = 10%
        sample_weight: optional sample weights
    Returns:
        float: Trimmed centrality value
    """
    if trim < 0 or trim > 0.5:
        raise ValueError('trim must be between 0 and 0.5')

    a = np.asarray(a)
    n = len(a)
    k = int(np.floor(n * trim))

    idx = np.argsort(a)
    trimmed_idx = idx[k:n-k]
    trimmed = a[trimmed_idx]

    if sample_weight is not None:
        trimmed_weights = np.asarray(sample_weight)[trimmed_idx]
        return centrality_measure(trimmed, sample_weight=trimmed_weights)

    return centrality_measure(trimmed)


@_name_function('trimmed_mean')
def trimmed_mean(a, trim=0.1, sample_weight=None):
    """Compute the mean after removing a fraction of extreme values from both tails.

    Parameters:
        a: array-like input
        trim: fraction to remove from each tail (default 0.1 = 10% total)
        sample_weight: optional sample weights

    Returns:
        float: Trimmed mean value
    """
    return trimmed_centrality(a, mean, trim=trim, sample_weight=sample_weight)


def winsorized_centrality(a, centrality_measure, limits=0.1, sample_weight=None):
    """Compute a winsorized centrality measure by capping extreme values.

    Parameters:
        a: array-like input
        centrality_measure: function to compute centrality (e.g., mean, median)
        limits: fraction to cap at each tail (default 0.1)
        sample_weight: optional sample weights

    Returns:
        float: Winsorized centrality value
    """
    if limits < 0 or limits > 0.5:
        raise ValueError('limits must be between 0 and 0.5')

    lower_q = limits
    upper_q = 1.0 - limits

    lower_val = quantile(a, lower_q, sample_weight)
    upper_val = quantile(a, upper_q, sample_weight)

    winsorized = np.clip(a, lower_val, upper_val)

    return centrality_measure(winsorized, sample_weight=sample_weight)

@_name_function('winsorized_mean')
def winsorized_mean(a, limits=0.1, sample_weight=None):
    """Compute the mean after capping extreme values at percentiles.

    Parameters:
        a: array-like input
        limits: fraction to cap at each tail (default 0.1)
        sample_weight: optional sample weights

    Returns:
        float: Winsorized mean value
    """
    return winsorized_centrality(a, mean, limits=limits, sample_weight=sample_weight)

def generalized_midhinge(a, q1=0.25, q3=0.75, sample_weight=None):
    """Compute the generalized midhinge: average of specified lower and upper quantiles.

    Parameters:
        a: array-like input
        q1: lower quantile (default 0.25)
        q3: upper quantile (default 0.75)
        sample_weight: optional sample weights

    Returns:
        float: Generalized midhinge value
    """
    lower_q = quantile(a, q1, sample_weight)
    upper_q = quantile(a, q3, sample_weight)
    return (lower_q + upper_q) / 2.0

@_name_function('midhinge')
def midhinge(a, sample_weight=None):
    """Compute the average of the lower and upper quartiles.

    Parameters:
        a: array-like input
        sample_weight: optional sample weights

    Returns:
        float: Midhinge value (Q1 + Q3) / 2
    """
    return generalized_midhinge(a, q1=0.25, q3=0.75, sample_weight=sample_weight)


@_name_function('power_mean')
def power_mean(a, p=1.0, sample_weight=None):
    """Compute the power mean (generalized mean) with exponent p.

    Special cases:
    - p → -∞: minimum
    - p = -1: harmonic mean
    - p → 0: geometric mean
    - p = 1: arithmetic mean
    - p = 2: quadratic mean (RMS)
    - p → ∞: maximum

    Parameters:
        a: array-like input (must be positive for p < 0 or p → 0)
        p: exponent parameter (default 1.0 for arithmetic mean)
        sample_weight: optional sample weights

    Returns:
        float: Power mean value
    """
    a = np.asarray(a)

    if p == 0:
        return geometric_mean(a, sample_weight)
    elif p == 1:
        return mean(a, sample_weight)
    elif p == np.inf:
        return max(a, sample_weight)
    elif p == -np.inf:
        return min(a, sample_weight)
    else:
        if np.any(a <= 0) and p < 0:
            raise ValueError('power_mean requires positive values for p < 0')

        return mirror_centrality(
            a,
            lambda x: np.power(x, p),
            lambda x: np.power(x, 1.0 / p),
            mean,
            sample_weight=sample_weight,
        )


@_name_function('hodges_lehmann')
def hodges_lehmann(a, sample_weight=None):
    """Compute the Hodges-Lehmann estimator: median of all pairwise averages.

    Very robust to outliers. Note: ignores sample_weight as this is not standard
    for this estimator.

    Parameters:
        a: array-like input
        sample_weight: ignored

    Returns:
        float: Hodges-Lehmann estimator value
    """
    a = np.asarray(a)
    n = len(a)

    # Compute all pairwise averages (including self-pairs)
    pairwise_means = np.zeros(n * (n + 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            pairwise_means[idx] = (a[i] + a[j]) / 2.0
            idx += 1

    return np.median(pairwise_means)


@_name_function('midrange')
def midrange(a, sample_weight=None):
    """Compute the midrange: (min + max) / 2.

    Quick rough center estimate. Ignores sample_weight.

    Parameters:
        a: array-like input
        sample_weight: ignored

    Returns:
        float: Midrange value
    """
    return (np.min(a) + np.max(a)) / 2.0
