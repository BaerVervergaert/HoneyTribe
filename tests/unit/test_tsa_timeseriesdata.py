import pandas as pd
import numpy as np

from honeytribe.tsa import TimeSeriesData

def test_time_index_frequency_inference_regular():
    dates = pd.date_range('2025-01-01', periods=5, freq='D')
    df = pd.DataFrame({'val': range(5)}, index=dates)
    ts = TimeSeriesData(df)
    assert ts.freq == 'D'
    assert ts.is_regular


def test_numeric_index_frequency_inference_regular():
    df = pd.DataFrame({'val': range(5)}, index=range(0, 10, 2))
    ts = TimeSeriesData(df)
    assert ts.freq == 2
    assert ts.is_regular


def test_lag_numeric_regular():
    df = pd.DataFrame({'val': range(5)}, index=range(0, 10, 2))
    ts = TimeSeriesData(df)
    lagged = ts.lag(2)
    assert (lagged.index == np.array([2,4,6,8,10])).all()


def test_lag_time_regular():
    dates = pd.date_range('2025-01-01', periods=5, freq='D')
    df = pd.DataFrame({'val': range(5)}, index=dates)
    ts = TimeSeriesData(df)
    lagged = ts.lag(1)
    assert lagged.index[0] == dates[0] + pd.Timedelta('1D')


def test_lag_irregular_time_errors():
    dates = pd.to_datetime(['2025-01-01','2025-01-02','2025-01-04','2025-01-05'])
    df = pd.DataFrame({'val': range(4)}, index=dates)
    ts = TimeSeriesData(df)
    assert ts.freq == 'irregular time'
    import pytest
    with pytest.raises(ValueError):
        ts.lag(1)

