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


def test_string_index_unknown_dimension():
    """Test TimeSeriesData with string index (non-numeric, non-time)."""
    df = pd.DataFrame({'val': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])
    ts = TimeSeriesData(df)
    assert ts._index_dimension == 'unknown'
    assert ts.freq == 'unknown'


def test_categorical_index_unknown_dimension():
    """Test TimeSeriesData with categorical index."""
    cat_index = pd.CategoricalIndex(['cat1', 'cat2', 'cat3'])
    df = pd.DataFrame({'val': [1, 2, 3]}, index=cat_index)
    ts = TimeSeriesData(df)
    assert ts._index_dimension == 'unknown'
    assert ts.freq == 'unknown'


def test_multi_index_unknown_dimension():
    """Test TimeSeriesData with MultiIndex."""
    multi_idx = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)])
    df = pd.DataFrame({'val': [10, 20, 30]}, index=multi_idx)
    ts = TimeSeriesData(df)
    assert ts._index_dimension == 'unknown'
    assert ts.freq == 'unknown'


def test_unknown_index_can_access_df_attributes():
    """Test that unknown index types still allow access to df attributes."""
    df = pd.DataFrame({'val': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])
    ts = TimeSeriesData(df)

    # Should be able to access DataFrame attributes through delegation
    assert ts.shape == (4, 1)
    assert list(ts.columns) == ['val']
    assert len(ts) == 4


def test_unknown_index_lag_uses_shift():
    """Test that lag on unknown index uses simple shift."""
    df = pd.DataFrame({'val': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])
    ts = TimeSeriesData(df)

    lagged = ts.lag(1)
    assert pd.isna(lagged.iloc[0]['val'])  # First value should be NaN
    assert lagged.iloc[1]['val'] == 1  # Second value should be original first
    assert lagged.iloc[2]['val'] == 2


def test_timedelta_index_recognized_as_time():
    """Test that TimedeltaIndex is recognized as time dimension."""
    td_index = pd.TimedeltaIndex(['1 day', '2 days', '3 days', '4 days'])
    df = pd.DataFrame({'val': [1, 2, 3, 4]}, index=td_index)
    ts = TimeSeriesData(df)
    assert ts._index_dimension == 'time'


def test_interval_index_recognized_as_time():
    """Test that IntervalIndex is recognized as time dimension."""
    dates = pd.date_range('2025-01-01', periods=4, freq='D')
    interval_idx = pd.IntervalIndex.from_arrays(dates[:-1], dates[1:])
    df = pd.DataFrame({'val': [1, 2, 3]}, index=interval_idx)
    ts = TimeSeriesData(df)
    assert ts._index_dimension == 'time'
    # IntervalIndex doesn't support frequency inference
    assert ts.freq == 'unknown'


