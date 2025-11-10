import numpy as np
import pandas as pd

from honeytribe.tsa import TimeSeriesData
from honeytribe.metrics.timeseries import (
    exponentially_weighted_mean,
    exponentially_weighted_correlation,
)


def test_practical_monotone_increasing_series_ew_mean_tracks_signal_and_smooths():
    n = 200
    idx = np.arange(n)
    signal = np.linspace(0, 10, n)
    noise = np.random.RandomState(1).randn(n) * 0.2
    x = signal + noise
    df = pd.DataFrame({'x': x}, index=idx)
    ts = TimeSeriesData(df)

    p = 0.98
    ew = exponentially_weighted_mean(ts, p=p)
    ew_series = ew.df['x']

    # Property 1: High correlation with underlying trend
    corr_with_trend = np.corrcoef(ew_series, signal)[0, 1]
    assert corr_with_trend > 0.995

    # Property 2: EW series reduces high frequency noise: first difference variance lower than raw series
    raw_diff_std = np.diff(x).std()
    ew_diff_std = np.diff(ew_series).std()
    assert ew_diff_std < raw_diff_std

    # Property 3: Final value not too far from final trend point (allow some lag)
    # Calibrated threshold: allow lag-induced deviation up to 2.5 for large p
    assert abs(ew_series.iloc[-1] - signal[-1]) < 2.5

    # Additional property: increasing p increases terminal lag error monotonically (more smoothing / inertia)
    errors = []
    for p_val in [0.9, 0.95, 0.97, 0.98]:
        ewp = exponentially_weighted_mean(ts, p=p_val).df['x']
        errors.append(abs(ewp.iloc[-1] - signal[-1]))
    assert all(errors[i] <= errors[i+1] + 1e-9 for i in range(len(errors)-1))


def test_practical_ew_correlation_detects_relationship():
    rng = np.random.RandomState(2)
    n = 300
    idx = np.arange(n)
    x = rng.randn(n)
    y = 0.8 * x + 0.2 * rng.randn(n)
    df = pd.DataFrame({'x': x, 'y': y}, index=idx)
    ts = TimeSeriesData(df)

    corr = exponentially_weighted_correlation(ts, p=0.99)

    # Practical expectation: correlation series mostly high positive
    mean_corr = corr.df['x_y'].mean()
    assert mean_corr > 0.6
