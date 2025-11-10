import numpy as np
import pandas as pd

from honeytribe.metrics.correlation import standard as corr


def test_basic_correlations_linear_monotonic():
    rng = np.random.RandomState(0)
    n = 200
    x = np.linspace(0, 10, n) + rng.randn(n) * 0.01
    y = 2.0 * x + rng.randn(n) * 0.01
    pear = corr.pearsonr(x, y)
    spear = corr.spearmanr(x, y)
    kend = corr.kendalltau(x, y)
    assert pear.value > 0.99
    assert spear.value > 0.99
    assert kend.value > 0.95


def test_chatterjeexi_independence_and_functional_relation():
    rng = np.random.RandomState(1)
    n = 300
    x = rng.rand(n)
    y_indep = rng.rand(n)
    xi_indep = corr.chatterjeexi(x, y_indep)
    assert xi_indep.value <= 1
    # For independence, xi should be small on average
    assert np.isclose(xi_indep.value, 0.0, atol=0.1)
    # Nonlinear deterministic relation with small noise
    y_func = x**2 + rng.randn(n) * 0.01
    xi_func = corr.chatterjeexi(x, y_func)
    assert xi_func.value > 0.7


def test_stepanovr_matches_definition():
    rng = np.random.RandomState(2)
    n = 150
    x = rng.randn(n)
    y = 0.5 * x + rng.randn(n) * 0.1
    r_step = corr.stepanovr(x, y).value
    tau = corr.kendalltau(x, y).value
    rho = corr.spearmanr(x, y).value
    expected = (3 * tau - rho) / 2
    assert np.isclose(r_step, expected, rtol=1e-12, atol=1e-12)
    # Symmetry
    assert np.isclose(corr.stepanovr(y, x).value, r_step)


def test_somersd_on_contingency_table():
    # 2x2 contingency table
    table = np.array([[10, 5], [2, 8]])
    out = corr.somersd(table)
    assert -1 <= out.value <= 1
    assert out.p_value is not None


def test_correlation_matrix_non_symmetric():
    rng = np.random.RandomState(3)
    n = 100
    A = pd.DataFrame({
        'a': rng.randn(n),
        'b': rng.randn(n)
    })
    B = pd.DataFrame({'c': A['a'] * 0.3 + rng.randn(n) * 0.7})
    M = corr.correlation_matrix(A, B, metric=corr.pearsonr)
    assert M.value.shape == (2, 1)
    # Compare with direct pearson for (a, c)
    direct = corr.pearsonr(A['a'], B['c']).value
    assert np.isclose(M.value[0, 0], direct, atol=1e-12)


def test_auto_correlation_series_ar1_positive_lag1():
    rng = np.random.RandomState(4)
    n = 300
    x = np.zeros(n)
    phi = 0.85
    eps = rng.randn(n) * 0.2
    for t in range(1, n):
        x[t] = phi * x[t-1] + eps[t]
    ac = corr.auto_correlation_series(x, lags=3, metric=corr.pearsonr)
    assert len(ac.value) == 3
    assert ac.value[0] > 0.6


def test_auto_correlation_matrix_shapes():
    rng = np.random.RandomState(5)
    n = 200
    def ar1(phi):
        x = np.zeros(n)
        e = rng.randn(n) * 0.1
        for t in range(1, n):
            x[t] = phi * x[t-1] + e[t]
        return x
    A = pd.DataFrame({
        'x1': ar1(0.8),
        'x2': ar1(0.5),
        'x3': ar1(0.2),
    })
    M = corr.auto_correlation_matrix(A, lags=4, metric=corr.pearsonr)
    assert M.value.shape == (4, 3)
    # Stronger AR(1) should have higher lag1 autocorrelation
    assert M.value[0, 0] > M.value[0, 1] > M.value[0, 2]


def test_relative_model_improvement_coefficient_cases():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    baseline = np.array([0.0, 0.0, 0.0])
    assert np.isclose(corr.relative_model_improvement_coefficient(a, b, baseline), 1.0)
    # No improvement over baseline
    assert np.isclose(corr.relative_model_improvement_coefficient(baseline, b, baseline), 0.0)
    # Opposite signal implies -1
    assert np.isclose(corr.relative_model_improvement_coefficient(-a, b, baseline), -1.0)


def test_error_correlation_for_model_improvement_equals_pearson_on_errors():
    rng = np.random.RandomState(6)
    n = 120
    a = rng.randn(n)
    b = 0.7 * a + rng.randn(n) * 0.3
    baseline = rng.randn(n) * 0.1
    out = corr.error_correlation_for_model_improvement(a, b, baseline, metric=corr.pearsonr)
    # Should equal pearsonr(a-baseline, b-baseline)
    r_direct = corr.pearsonr(a - baseline, b - baseline)
    assert np.isclose(out.value, r_direct.value, atol=1e-12)

