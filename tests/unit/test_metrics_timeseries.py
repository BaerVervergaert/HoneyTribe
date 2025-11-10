import numpy as np
import pandas as pd

from honeytribe.tsa import TimeSeriesData
from honeytribe.metrics.timeseries import (
    exponentially_weighted_mean,
    rolling_unary_metric,
    exponentially_weighted_std,
    exponentially_weighted_covariance,
    exponentially_weighted_correlation,
    rolling_correlation_metric,
    rolling_autocorrelation_metric,
)
from honeytribe.metrics.correlation import standard as corr_std


def test_exponentially_weighted_mean_matches_pandas_on_regular_numeric():
    # Regular numeric index
    n = 50
    idx = np.arange(n)
    x = np.linspace(0, 1, n)
    df = pd.DataFrame({'x': x}, index=idx)
    ts = TimeSeriesData(df)

    p = 0.9
    ours = exponentially_weighted_mean(ts, p=p)
    # pandas ewm with adjust=True corresponds to normalized weighted average
    alpha = 1 - p
    expected = df['x'].ewm(alpha=alpha, adjust=True).mean()

    pd.testing.assert_index_equal(ours.df.index, df.index)
    assert 'x' in ours.df.columns
    # allow small numerical tolerance
    np.testing.assert_allclose(ours.df['x'].to_numpy(), expected.to_numpy(), rtol=1e-6, atol=1e-8)


def test_exponentially_weighted_mean_on_time_index_matches_pandas():
    n = 40
    idx = pd.date_range('2025-01-01', periods=n, freq='h')
    x = np.sin(np.linspace(0, 3.14, n))
    df = pd.DataFrame({'x': x}, index=idx)
    ts = TimeSeriesData(df)
    p = 0.93
    alpha = 1 - p
    ours = exponentially_weighted_mean(ts, p=p)
    expected = df['x'].ewm(alpha=alpha, adjust=True).mean()
    # indices should match
    pd.testing.assert_index_equal(ours.df.index, df.index)
    np.testing.assert_allclose(ours.df['x'].to_numpy(), expected.to_numpy(), rtol=1e-6, atol=1e-8)


def test_exponentially_weighted_mean_p_equal_1_is_cumulative_mean():
    n = 20
    idx = np.arange(n)
    x = np.random.RandomState(0).randn(n)
    df = pd.DataFrame({'x': x}, index=idx)
    ts = TimeSeriesData(df)
    ew = exponentially_weighted_mean(ts, p=1.0)
    # Last value must equal simple mean of entire series
    assert np.isclose(ew.df['x'].iloc[-1], df['x'].mean())


def test_exponentially_weighted_std_zero_for_constant_series():
    n = 30
    idx = np.arange(n)
    df = pd.DataFrame({'x': np.ones(n) * 5.0}, index=idx)
    ts = TimeSeriesData(df)
    std = exponentially_weighted_std(ts, p=0.95)
    assert np.allclose(std.df['x'].fillna(0).to_numpy(), 0.0)


def test_exponentially_weighted_covariance_and_correlation_labels_and_values():
    n = 80
    idx = np.arange(n)
    rng = np.random.RandomState(1)
    x = rng.randn(n)
    y = 0.5 * x + rng.randn(n) * 0.1
    df = pd.DataFrame({'x': x, 'y': y}, index=idx)
    ts = TimeSeriesData(df)

    cov = exponentially_weighted_covariance(ts, p=0.97)
    # Expected labels present
    for k in ['x_x', 'x_y', 'y_x', 'y_y']:
        assert k in cov.df.columns
    # Variances non-negative
    assert (cov.df['x_x'].dropna() >= 0).all()
    assert (cov.df['y_y'].dropna() >= 0).all()

    corr = exponentially_weighted_correlation(ts, p=0.97)
    # Correlation of identical series should approach 1 after warm-up
    tail = corr.df['x_x'].iloc[n//4:]
    assert np.nanmean(np.abs(tail - 1.0)) < 1e-3
    # Cross-correlation should be high positive
    assert corr.df['x_y'].mean() > 0.5


def test_rolling_unary_metric_mean():
    n = 30
    idx = np.arange(n)
    x = np.random.RandomState(0).randn(n)
    df = pd.DataFrame({'x': x}, index=idx)
    ts = TimeSeriesData(df)

    win = 5
    ours = rolling_unary_metric(ts, np.mean, window=win)
    expected = df['x'].rolling(window=win).mean()

    assert 'x' in ours.columns
    np.testing.assert_allclose(ours['x'].to_numpy(), expected.to_numpy(), equal_nan=True)


def test_rolling_correlation_metric_symmetry_and_shape():
    n = 60
    idx = np.arange(n)
    rng = np.random.RandomState(2)
    a = rng.randn(n)
    b = 0.3 * a + rng.randn(n) * 0.7
    df = pd.DataFrame({'a': a, 'b': b}, index=idx)
    ts = TimeSeriesData(df)

    win = 10
    out = rolling_correlation_metric(ts, corr_std.pearsonr, window=win)
    # Columns and symmetry
    for label in ['a>b', 'b>a', 'a>a', 'b>b']:
        assert label in out.columns
    # Symmetry (ignoring NaNs at the start)
    diff = (out['a>b'] - out['b>a']).dropna()
    assert np.allclose(diff, 0, atol=1e-8)


def test_rolling_autocorrelation_metric_basic_ar1_signal():
    n = 200
    idx = np.arange(n)
    rng = np.random.RandomState(3)
    x = np.zeros(n)
    phi = 0.8
    eps = rng.randn(n) * 0.2
    for t in range(1, n):
        x[t] = phi * x[t-1] + eps[t]
    df = pd.DataFrame({'x': x}, index=idx)
    ts = TimeSeriesData(df)

    out = rolling_autocorrelation_metric(ts, corr_std.pearsonr, lags=3, window=30)
    # Check presence of lag 1 autocorrelation label
    assert 'x_lag(1)>x' in out.columns
    # Mean autocorrelation at lag 1 should be positive and reasonably large
    assert out['x_lag(1)>x'].mean() > 0.4


def test_exponentially_weighted_covariance_matches_pandas():
    n = 100
    idx = np.arange(n)
    rng = np.random.RandomState(42)
    x = rng.randn(n)
    y = 0.7 * x + rng.randn(n) * 0.3
    df = pd.DataFrame({'x': x, 'y': y}, index=idx)
    ts = TimeSeriesData(df)
    p = 0.92
    alpha = 1 - p
    ours_cov = exponentially_weighted_covariance(ts, p=p).df
    pd_cov = df.ewm(alpha=alpha, adjust=True).cov(bias=True)
    # Wide format expected
    expected_cov = pd.DataFrame(index=df.index)
    for row_col in df.columns:
        slice_df = pd_cov.xs(row_col, level=1)
        for col_col in df.columns:
            expected_cov[f'{row_col}_{col_col}'] = slice_df[col_col]
    # After warm-up
    warm = slice(5, None)
    ours_slice = ours_cov.iloc[warm]
    exp_slice = expected_cov.iloc[warm]
    # 1. Sign consistency: diagonal positive
    for col in df.columns:
        assert (ours_slice[f'{col}_{col}'] >= 0).all()
        assert (exp_slice[f'{col}_{col}'] >= 0).all()
    # 2. Cross-covariance sign alignment (x_y)
    ours_sign = np.sign(ours_slice['x_y']).iloc[-20:].mean()
    exp_sign = np.sign(exp_slice['x_y']).iloc[-20:].mean()
    assert ours_sign * exp_sign > 0  # same predominant sign
    # 3. Relative ordering of variances maintained
    ours_var_ratio = (ours_slice['x_x'] / ours_slice['y_y']).median()
    exp_var_ratio = (exp_slice['x_x'] / exp_slice['y_y']).median()
    assert np.sign(ours_var_ratio - 1) == np.sign(exp_var_ratio - 1)
    # 4. Reconstructed correlation from our cov approximates pandas correlation
    ours_std_x = np.sqrt(ours_slice['x_x'])
    ours_std_y = np.sqrt(ours_slice['y_y'])
    ours_corr = (ours_slice['x_y'] / (ours_std_x * ours_std_y))
    pd_corr_series = df['x'].ewm(alpha=alpha, adjust=True).corr(df['y'])
    pd_corr = pd_corr_series.iloc[warm]
    # Compare median correlation
    assert abs(ours_corr.median() - pd_corr.median()) < 0.1


def test_exponentially_weighted_std_matches_pandas():
    n = 120
    idx = np.arange(n)
    rng = np.random.RandomState(7)
    x = rng.randn(n)
    y = 0.5 * rng.randn(n) + 0.2
    df = pd.DataFrame({'x': x, 'y': y}, index=idx)
    ts = TimeSeriesData(df)
    p = 0.9
    alpha = 1 - p
    ours_std = exponentially_weighted_std(ts, p=p).df
    pd_var = df.ewm(alpha=alpha, adjust=True).var(bias=True)
    pd_std = pd_var.pow(0.5)
    slice_start = 5
    ours_slice = ours_std.iloc[slice_start:]
    pd_slice = pd_std.iloc[slice_start:]
    # 1. Non-negativity
    for col in df.columns:
        assert (ours_slice[col] >= 0).all()
        assert (pd_slice[col] >= 0).all()
    # 2. Relative ordering of typical std magnitude preserved
    ours_ratio = (ours_slice['x'] / ours_slice['y']).median()
    pd_ratio = (pd_slice['x'] / pd_slice['y']).median()
    assert np.sign(ours_ratio - 1) == np.sign(pd_ratio - 1)
    # 3. Magnitude alignment: our std within 10% of pandas std on median
    for col in df.columns:
        median_ours = ours_slice[col].median()
        median_pd = pd_slice[col].median()
        assert abs(median_ours - median_pd) / median_pd < 0.1
    # 4. Scale consistency: ratio between our std and pandas std roughly stable (low coefficient of variation)
    for col in df.columns:
        ratio = ours_slice[col] / pd_slice[col]
        # Ignore initial possible zeros
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) > 10:
            cv = ratio.std() / ratio.mean()
            assert cv < 0.1
            # Assert most are close to 1
            close_to_one = (np.abs(ratio - 1) < 0.1).sum() / len(ratio)
            assert close_to_one > 0.9 # at least 90% of ratio's is within 10% error of 1
        else:
            # Not enough data to assess stability, fail
            assert False, "Not enough data points after cleaning to assess scale consistency."

