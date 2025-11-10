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