import numpy as np

from honeytribe.metrics.variability.standard import mirror_variability, std as std_fn
from honeytribe.metrics.centrality import standard as cen


def test_mirror_variability_std_matches_std_function():
    rng = np.random.RandomState(123)
    a = rng.randn(200) * 3 + 5
    got_std = mirror_variability(
        a,
        mirror_transform=lambda x: x**2,
        inverse_transform=np.sqrt,
        centrality_measure=cen.mean,
        variance_centrality_measure=cen.mean,
        bias_correction=lambda arr: len(arr) / (len(arr) - 1),
    )
    exp_std = std_fn(a)
    assert np.isclose(got_std, exp_std, rtol=1e-12, atol=1e-12)


def test_mirror_variability_mean_absolute_deviation():
    rng = np.random.RandomState(0)
    a = rng.randn(100) + 2
    mad = mirror_variability(
        a,
        mirror_transform=lambda x: x,  # identity
        inverse_transform=lambda x: x,
        centrality_measure=cen.mean,
        variance_centrality_measure=cen.mean,
    )
    exp_mad = np.mean(np.abs(a - np.mean(a)))
    assert np.isclose(mad, exp_mad, rtol=1e-12, atol=1e-12)


def test_mirror_variability_weighted_mean_absolute_deviation():
    a = np.array([1.0, 2.0, 3.0, 10.0])
    w = np.array([1.0, 1.0, 1.0, 5.0])  # heavier outlier
    weighted_mean = np.average(a, weights=w)
    mad_w = mirror_variability(
        a,
        mirror_transform=lambda x: x,
        inverse_transform=lambda x: x,
        centrality_measure=cen.mean,
        variance_centrality_measure=cen.mean,
        sample_weight=w,
    )
    exp_mad_w = np.sum(np.abs(a - weighted_mean) * w) / np.sum(w)
    assert np.isclose(mad_w, exp_mad_w, rtol=1e-12, atol=1e-12)


def test_mirror_variability_median_absolute_deviation_less_than_mean_with_outlier():
    # Outlier inflates mean-based deviation more than median-based
    a = np.array([1, 2, 2, 2, 2, 50])
    mad_mean = mirror_variability(
        a,
        mirror_transform=lambda x: x,
        inverse_transform=lambda x: x,
        centrality_measure=cen.mean,
        variance_centrality_measure=cen.mean,
    )
    mad_median = mirror_variability(
        a,
        mirror_transform=lambda x: x,
        inverse_transform=lambda x: x,
        centrality_measure=cen.median,
        variance_centrality_measure=cen.mean,
    )
    assert mad_median < mad_mean


def test_mirror_variability_custom_p_norm():
    rng = np.random.RandomState(99)
    a = rng.randn(150)
    p = 3
    lp_metric = mirror_variability(
        a,
        mirror_transform=lambda x, p=p: x**p,
        inverse_transform=lambda y, p=p: y**(1/p),
        centrality_measure=cen.mean,
        variance_centrality_measure=cen.mean,
    )
    exp_lp = (np.mean(np.abs(a - np.mean(a))**p))**(1/p)
    assert np.isclose(lp_metric, exp_lp, rtol=1e-12, atol=1e-12)

