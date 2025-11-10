import numpy as np

from honeytribe.metrics.variability import standard as var
from honeytribe.metrics.centrality import standard as cen


def test_std_matches_numpy_unbiased():
    rng = np.random.RandomState(0)
    a = rng.randn(100) * 2 + 1
    # Our std uses unbiased correction via mirror_variability
    got = var.std(a)
    exp = np.std(a, ddof=1)
    assert np.isclose(got, exp, rtol=1e-12, atol=1e-12)


def test_std_weighted_matches_weighted_unbiased_when_weights_equal():
    # For equal weights, weighted unbiased should match numpy ddof=1
    a = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones_like(a)
    got = var.std(a, sample_weight=w)
    exp = np.std(a, ddof=1)
    assert np.isclose(got, exp)


def test_std_raises_for_too_few_points():
    import pytest
    with pytest.raises(ValueError):
        var.std(np.array([1.0]))


def test_quantile_spread_as_coverage():
    a = np.linspace(0.0, 1.0, 101)
    # 80% coverage -> between 10% and 90% quantiles -> ~0.8 range
    spread = var.quantile_spread(a, 0.8)
    assert np.isclose(spread, 0.8, atol=1e-6)


def test_quantile_spread_between_quantiles():
    rng = np.random.RandomState(2)
    a = np.sort(rng.randn(200))
    spread = var.quantile_spread(a, 0.25, 0.75)
    q25 = np.quantile(a, 0.25, method='linear')
    q75 = np.quantile(a, 0.75, method='linear')
    assert np.isclose(spread, q75 - q25)


def test_iqr_aliases_quantile_spread_25_75():
    rng = np.random.RandomState(3)
    a = rng.randn(500)
    assert np.isclose(var.iqr(a), var.quantile_spread(a, 0.25, 0.75))


def test_quantile_spread_weighted_vs_unweighted_behavior():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 3.0, 5.0])
    spread_u = var.quantile_spread(a, 0.25, 0.75)
    spread_w = var.quantile_spread(a, 0.25, 0.75, sample_weight=w)
    # Compute explicit quantiles to inspect shift
    q25_u = cen.quantile(a, 0.25)
    q75_u = cen.quantile(a, 0.75)
    q25_w = cen.quantile(a, 0.25, sample_weight=w)
    q75_w = cen.quantile(a, 0.75, sample_weight=w)
    # With heavier upper-tail weights, both quantiles should not decrease
    assert q25_w >= q25_u
    assert q75_w >= q75_u
    # Spreads can shrink or expand depending on distribution; just assert a change occurred
    assert not np.isclose(spread_w, spread_u)
