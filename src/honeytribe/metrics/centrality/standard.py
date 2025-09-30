import numpy as np
import pandas as pd

def mean(a, sample_weight=None):
    val = a.copy()
    if sample_weight is not None:
        N = len(a)
        norm = np.sum(sample_weight) / N
        val = val * sample_weight / norm
    return np.mean(val)

def median(a, sample_weight=None):
    q = 0.5
    return quantile(a, q, sample_weight)


def quantile(a, q: float, sample_weight = None):
    idx = np.argsort(a)
    val = a[idx]
    if sample_weight is not None:
        p = sample_weight[idx] / np.sum(sample_weight)
        i = np.searchsorted(p, q)
        if i > 0:
            p_1 = p[i]
            p_0 = p[i - 1]
            d_1 = p_1 - q
            d_0 = q - p_0
            q_1 = d_1 / (d_1 + d_0)
            q_0 = 1 - q_1
        else:
            q_1 = 1
            q_0 = 0
    else:
        N = len(a)
        i = N // 2
        if i % 2 == 1:
            q_1 = 1
            q_0 = 0
        else:
            q_1 = .5
            q_0 = .5
    if q_0 > 0:
        val_1 = val[i]
        val_0 = val[i - 1]
        return q_1 * val_1 + q_0 * val_0
    else:
        return val[i]

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

def geometric_mean(a, sample_weight=None):
    return mirror_centrality(
        a,
        np.log,
        np.exp,
        mean,
        sample_weight = sample_weight,
    )