import numpy as np
import pandas as pd

from honeytribe.metrics.centrality import standard as cen


def test_mean_unweighted_and_weighted():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 2.0, 2.0])
    assert np.isclose(cen.mean(a), np.mean(a))
    expected_w = np.average(a, weights=w)
    assert np.isclose(cen.mean(a, sample_weight=w), expected_w)


def test_quantile_unweighted_matches_numpy_linear():
    rng = np.random.RandomState(0)
    a = rng.randn(50)
    for q in [0.0, 0.1, 0.25, 0.33, 0.5, 0.7, 0.75, 0.9, 1.0]:
        got = cen.quantile(a, q)
        exp = np.quantile(a, q, method='linear')
        assert np.isclose(got, exp)


def test_quantile_weighted_simple_case():
    # Values 0,1,2,3 with weights emphasizing upper values
    a = np.array([0.0, 1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 3.0, 5.0])
    # Compute expected via cumulative weight interpolation
    idx = np.argsort(a)
    a_sorted = a[idx]
    w_sorted = w[idx]
    c = np.cumsum(w_sorted) / np.sum(w_sorted)
    def wq(q):
        i = np.searchsorted(c, q, side='left')
        if i == 0:
            return a_sorted[0]
        if i >= len(a):
            return a_sorted[-1]
        p0 = c[i-1]
        p1 = c[i]
        if p1 == p0:
            return a_sorted[i]
        t = (q - p0) / (p1 - p0)
        return (1 - t) * a_sorted[i-1] + t * a_sorted[i]
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        got = cen.quantile(a, q, sample_weight=w)
        exp = wq(q)
        assert np.isclose(got, exp)


def test_median_lower_upper_quartile_consistency():
    rng = np.random.RandomState(1)
    a = rng.randn(101)
    assert np.isclose(cen.median(a), cen.quantile(a, 0.5))
    assert np.isclose(cen.lower_quartile(a), cen.quantile(a, 0.25))
    assert np.isclose(cen.upper_quartile(a), cen.quantile(a, 0.75))


def test_min_max():
    a = np.array([3.0, -1.0, 2.5])
    assert cen.min(a) == np.min(a)
    assert cen.max(a) == np.max(a)


def test_mode_unweighted_and_weighted():
    a = np.array([1, 2, 2, 3, 3, 3, 4])
    assert cen.mode(a) == 3
    # Weight 2 heavier than 3
    w = np.array([1, 5, 5, 1, 1, 1, 1])
    assert cen.mode(a, sample_weight=w) == 2


def test_geometric_mean_unweighted_and_weighted():
    a = np.array([1.0, 2.0, 4.0, 8.0])
    # Unweighted geometric mean = exp(mean(log(a)))
    gm = np.exp(np.mean(np.log(a)))
    assert np.isclose(cen.geometric_mean(a), gm)
    # Weighted geometric mean = exp( sum(w*log(a))/sum(w) )
    w = np.array([1.0, 2.0, 1.0, 0.0])
    gm_w = np.exp(np.average(np.log(a), weights=w))
    assert np.isclose(cen.geometric_mean(a, sample_weight=w), gm_w)


def test_mirror_centrality_geometric_mean_matches_builtin():
    a = np.array([1.0, 2.0, 4.0, 8.0])
    gm_mc = cen.mirror_centrality(a, np.log, np.exp, cen.mean)
    gm_builtin = cen.geometric_mean(a)
    exp_val = np.exp(np.mean(np.log(a)))
    assert np.isclose(gm_mc, gm_builtin)
    assert np.isclose(gm_mc, exp_val)


def test_mirror_centrality_harmonic_mean():
    a = np.array([1.0, 2.0, 4.0, 8.0])
    # Harmonic mean = inverse(mean(1/a))
    hm_mc = cen.mirror_centrality(a, lambda x: 1.0 / x, lambda z: 1.0 / z, cen.mean)
    hm_expected = len(a) / np.sum(1.0 / a)
    assert np.isclose(hm_mc, hm_expected)


def test_mirror_centrality_std_population_unweighted():
    rng = np.random.RandomState(123)
    a = rng.randn(100) * 2 + 1
    # std via mirror: square centered values, average, sqrt
    mu = a.mean()
    std_mc = cen.mirror_centrality(a, lambda x: (x - mu) ** 2, np.sqrt, cen.mean)
    std_np = np.std(a)  # population std (ddof=0)
    assert np.isclose(std_mc, std_np, rtol=1e-12, atol=1e-12)


def test_mirror_centrality_std_population_weighted():
    rng = np.random.RandomState(42)
    a = rng.randn(80) + 3
    w = rng.rand(80) + 0.1
    mu_w = np.average(a, weights=w)
    # Use mean with sample_weight to average squared deviations
    std_mc = cen.mirror_centrality(
        a,
        lambda x: (x - mu_w) ** 2,
        np.sqrt,
        cen.mean,
        sample_weight=w,
    )
    std_expected = np.sqrt(np.average((a - mu_w) ** 2, weights=w))
    assert np.isclose(std_mc, std_expected, rtol=1e-12, atol=1e-12)


def test_mode_ties_choose_smallest_numeric():
    a = np.array([1, 1, 2, 2])
    # Both 1 and 2 occur twice; pandas groupby sorts keys, idxmax picks first -> 1
    assert cen.mode(a) == 1


def test_mode_ignores_nans():
    a = np.array([np.nan, np.nan, 1, np.nan, np.nan, 2, 2])
    # NaNs are excluded by groupby key; mode should be 2
    assert cen.mode(a) == 2


def test_mode_non_numeric_values():
    a = np.array(['a', 'b', 'b', 'c', 'a', 'b'], dtype=object)
    assert cen.mode(a) == 'b'


def test_mode_weighted_affects_result_on_tie():
    a = np.array([1, 2])
    w_equal = np.array([1.0, 1.0])
    # With equal weights and values occurring once each, smallest key chosen
    assert cen.mode(a, sample_weight=w_equal) == 1
    # Heavier weight for 2 flips the mode
    w_two_heavy = np.array([1.0, 2.0])
    assert cen.mode(a, sample_weight=w_two_heavy) == 2

