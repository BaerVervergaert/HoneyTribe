from typing import Callable

from honeytribe.tsa import TimeSeriesData
from honeytribe.metrics.utils import _name_function
from honeytribe.metrics.correlation.standard import CorrelationOutput
import numpy as np
import pandas as pd

def exponentially_weighted_transform_estimator(data:TimeSeriesData, transform:Callable, p:float=1.) -> TimeSeriesData:
    """Compute exponentially weighted transform estimator.

    Parameters:
        data (TimeSeriesData): The time series data.
        transform (Callable): The transformation function to apply to each row.
        p (float): The decay factor (0 < p <= 1). A value of
                     1 means no decay, while values closer to 0 give more weight to recent observations.
    Returns:
        TimeSeriesData: The exponentially weighted transform estimator.
    """
    prev_idx = None
    value = 0
    prev_norm = 0
    out = []
    out_index = []
    for idx, row in data.iterrows():
        if prev_idx is None:
            prev_idx = idx
        if data._index_dimension == 'time':
            if data.freq != 'irregular time':
                freq = data.freq
                if not freq[0].isdigit():
                    freq = '1' + freq
                delta = pd.Timedelta(freq)
                diff = (idx - prev_idx)/delta
            else:
                diff = idx - prev_idx
        elif data._index_dimension == 'numeric':
            if data.freq != 'irregular numeric':
                delta = data.freq
                diff = (idx - prev_idx)/delta
            else:
                diff = idx - prev_idx
        else:
            diff = 1
        decay = p**diff
        norm = decay * prev_norm + 1
        new_value = transform(row)
        value = decay * value + new_value
        out.append(value/norm)
        out_index.append(idx)
        prev_idx = idx
        prev_norm = norm
    return TimeSeriesData(
        pd.DataFrame(
            data = out,
            index = out_index
        )
    )

def exponentially_weighted_moment_estimator(data:TimeSeriesData, p:float=1., power:float=1., abs:bool = False) -> TimeSeriesData:
    """Compute exponentially weighted moment estimator.

    Parameters:
        data (TimeSeriesData): The time series data.
        p (float): The decay factor (0 < p <= 1). A value of
                        1 means no decay, while values closer to 0 give more weight to recent observations
        power (float): The moment power to compute.
        abs (bool): Whether to use absolute values when computing the moment.
    Returns:
        TimeSeriesData: The exponentially weighted moment estimator.
    """
    if abs:
        transform = lambda row, power=power: np.abs(row)**power
    else:
        transform = lambda row, power=power: row**power
    return exponentially_weighted_transform_estimator(data, transform, p)

@_name_function('exponentially_weighted_mean')
def exponentially_weighted_mean(data:TimeSeriesData, p:float=1.) -> TimeSeriesData:
    """Compute exponentially weighted mean.

    Parameters:
        data (TimeSeriesData): The time series data.
        p (float): The decay factor (0 < p <= 1). A value of
                        1 means no decay, while values closer to 0 give more weight to recent observations
    Returns:
        TimeSeriesData: The exponentially weighted mean.
    """
    return exponentially_weighted_moment_estimator(data, p, power=1, abs=False)

@_name_function('exponentially_weighted_std')
def exponentially_weighted_std(data:TimeSeriesData, p:float=1.) -> TimeSeriesData:
    """Compute exponentially weighted standard deviation using covariance diagonal for consistent scaling."""
    mean = exponentially_weighted_mean(data, p)
    centered = TimeSeriesData(
        data.df - mean.df
    )
    def transform(row):
        d = {}
        for col, value in row.items():
            d[col] = value ** 2
        return pd.Series(d)
    return exponentially_weighted_transform_estimator(centered, transform, p).apply_rowwise(np.sqrt)

@_name_function('exponentially_weighted_covariance')
def exponentially_weighted_covariance(data:TimeSeriesData, p:float=1.) -> TimeSeriesData:
    """Compute exponentially weighted covariance.

    Parameters:
        data (TimeSeriesData): The time series data.
        p (float): The decay factor (0 < p <= 1). A value of
                        1 means no decay, while values closer to 0 give more weight to recent observations
    Returns:
        TimeSeriesData: The exponentially weighted covariance.
    """
    mean = exponentially_weighted_mean(data, p)
    centered = TimeSeriesData(
        data.df - mean.df
    )
    def transform(row):
        d = {}
        for col1, value1 in row.items():
            for col2, value2 in row.items():
                label = f'{col1}_{col2}'
                d[label] = value1 * value2
        return pd.Series(d)
    return exponentially_weighted_transform_estimator(centered, transform, p)

@_name_function('exponentially_weighted_correlation')
def exponentially_weighted_correlation(data:TimeSeriesData, p:float=1.):
    """Compute exponentially weighted correlation.

    Uses covariance and standard deviations derived from the covariance diagonal for consistent normalization.
    Adds a small epsilon to denominators to avoid division by zero in near-constant segments.
    """
    cov = exponentially_weighted_covariance(data, p)
    std = exponentially_weighted_std(data, p)
    eps = 1e-12
    df = cov.df.copy()
    for col1 in data.columns:
        for col2 in data.columns:
            label = f'{col1}_{col2}'
            denom = (std.df[col1] * std.df[col2]).replace(0, eps)
            df[label] = df[label] / denom
    return TimeSeriesData(df)

def rolling_unary_metric(data: TimeSeriesData, func: Callable, *args, **kwargs) -> pd.DataFrame:
    """Compute rolling unary metrics.

    Parameters:
        data (TimeSeriesData): The time series data.
        func (Callable): The unary metric function to use.
    Returns:
        pd.DataFrame: DataFrame containing rolling unary metrics.
    """
    out = {}
    for col in data.columns:
        out[col] = data.df[col].rolling(*args, **kwargs).apply(func)
    return pd.DataFrame(out)

def rolling_correlation_metric(data: TimeSeriesData, correlation_metric: Callable, *args, **kwargs) -> pd.DataFrame:
    """Compute rolling multivariate correlation metrics.

    Parameters:
        data (TimeSeriesData): The time series data.
        correlation_metric (Callable): The correlation metric function to use.
    Returns:
        pd.DataFrame: DataFrame containing rolling correlation metrics.

    Notes:
        - Additional arguments and keyword arguments are passed to the pandas rolling method.
        - The resulting DataFrame will have columns named in the format
        "{col1}>{col2}" representing the correlation between dependent variable col1 and independent variable col2.
    """
    def transform(series, df, col1, col2):
        idx = series.index
        return correlation_metric(df.loc[idx, col1], df.loc[idx, col2]).value
    out = {}
    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            label = f"{col1}>{col2}"
            if correlation_metric._symmetric and i>j:
                out[label] = out[f"{col2}>{col1}"]
            else:
                out[label] = data.df[col1].rolling(*args, **kwargs).apply(transform, args=(data.df, col1, col2))
    return pd.DataFrame(out)

def rolling_autocorrelation_metric(data: TimeSeriesData, correlation_metric: Callable, lags: int|list[int], *args, **kwargs) -> pd.DataFrame:
    """Compute rolling multivariate autocorrelation metrics for specified lags.

    Parameters:
        data (TimeSeriesData): The time series data.
        correlation_metric (Callable): The correlation metric function to use.
        lags (int|list[int]): The lag(s) to compute the autocorrelation for
    Returns:
        pd.DataFrame: DataFrame containing rolling autocorrelation metrics.

    Notes:
        - Additional arguments and keyword arguments are passed to the pandas rolling method.
        - The function computes the cross correlation between each pair of columns.
        - The resulting DataFrame will have columns named in the format
        "{col1}_lag({lag})>{col2}" representing the correlation between
        the lagged version of col1 and col2.
        - This is a direct correlation computation and does not account for partial correlations. So, expect exponential decay in correlations for increasing lags in many real-world datasets.
    """
    def transform(series, df, col1, col2):
        idx = series.index
        return correlation_metric(df.loc[idx, col1], df.loc[idx, col2]).value
    if isinstance(lags, int):
        lags = list(range(lags))
    out = {}
    for lag in lags:
        lagged_data = data.lag(lag)
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(lagged_data.columns):
                label = f"{col1}_lag({lag})>{col2}"
                if correlation_metric._symmetric and i>j:
                    out[label] = out[f"{col2}_lag({lag})>{col2}"]
                else:
                    lag_col = f"{col1}_lag({lag})"
                    lagged_series:pd.Series = lagged_data[col1].rename(lag_col)
                    target_series = data.df[col2]
                    combined_df = pd.concat([lagged_series, target_series], axis=1)
                    out[label] = target_series.rolling(*args, **kwargs).apply(transform, args=(combined_df, lag_col, col2))
    return pd.DataFrame(out)
