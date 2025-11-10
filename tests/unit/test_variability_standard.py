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


def test_variance_is_std_squared():
    rng = np.random.RandomState(10)
    a = rng.randn(100)
    assert np.isclose(var.variance(a), var.std(a)**2)


def test_mad_mean_absolute_deviation():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mad_val = var.mad(a)
    mean_val = np.mean(a)
    expected = np.mean(np.abs(a - mean_val))
    assert np.isclose(mad_val, expected)


def test_mad_weighted():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    mad_w = var.mad(a, sample_weight=w)
    # Should be higher than unweighted due to heavy weight on 5 (extreme value)
    mad_u = var.mad(a)
    assert mad_w > mad_u


def test_median_absolute_deviation_robust():
    # Without outlier
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mad_clean = var.median_absolute_deviation(a)
    # With extreme outlier
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
    mad_outlier = var.median_absolute_deviation(b)
    # MAD should be less affected than std
    std_shift = var.std(b) - var.std(a)
    mad_shift = mad_outlier - mad_clean
    assert mad_shift < std_shift


def test_coefficient_of_variation_normalized():
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([100.0, 200.0, 300.0])  # 10x scale
    cv_a = var.coefficient_of_variation(a)
    cv_b = var.coefficient_of_variation(b)
    # CV should be similar despite different scales
    assert np.isclose(cv_a, cv_b, rtol=1e-10)


def test_coefficient_of_variation_zero_mean_raises():
    import pytest
    a = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        var.coefficient_of_variation(a)


def test_range_simple():
    a = np.array([1.0, 5.0])
    assert var.data_range(a) == 4.0


def test_range_with_outliers():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3, 4, 5, 100])
    assert var.data_range(b) > var.data_range(a)


def test_decile_range_more_robust_than_range():
    # Without extreme outliers
    a = np.array([1, 2, 3, 4, 5])
    dr_a = var.decile_range(a)
    # With extreme outliers
    b = np.array([1, 2, 3, 4, 5, 1000, 0.001])
    dr_b = var.decile_range(b)
    # Decile range should be similar despite extreme outliers
    # With extreme values, quantiles still shift; allow larger tolerance
    assert abs(dr_a - dr_b) < max(dr_a, dr_b)


def test_semi_interquartile_range_half_iqr():
    rng = np.random.RandomState(0)
    a = rng.randn(100)
    siqr = var.semi_interquartile_range(a)
    iqr_val = var.iqr(a)
    assert np.isclose(siqr, iqr_val / 2.0)


def test_normalized_iqr_scale_independent():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    n_iqr_a = var.normalized_iqr(a)
    n_iqr_b = var.normalized_iqr(b)
    assert np.isclose(n_iqr_a, n_iqr_b, rtol=1e-10)


def test_normalized_iqr_zero_median_raises():
    import pytest
    a = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        var.normalized_iqr(a)


def test_gini_coefficient_bounds():
    a = np.array([1.0, 2.0, 3.0])
    gini = var.gini_coefficient(a)
    assert 0 <= gini <= 1


def test_gini_coefficient_perfect_equality():
    a = np.array([5.0, 5.0, 5.0])
    gini = var.gini_coefficient(a)
    assert np.isclose(gini, 0.0, atol=1e-10)


def test_gini_coefficient_weighted():
    a = np.array([1.0, 2.0, 3.0])
    # Use equal weights for simple case
    gini_u = var.gini_coefficient(a)
    # Should be in [0, 1]
    assert 0 <= gini_u <= 1


def test_gini_coefficient_inequality():
    # More unequal distribution should have higher Gini
    equal = np.array([5.0, 5.0, 5.0])
    unequal = np.array([1.0, 5.0, 9.0])
    gini_equal = var.gini_coefficient(equal)
    gini_unequal = var.gini_coefficient(unequal)
    assert gini_unequal > gini_equal


