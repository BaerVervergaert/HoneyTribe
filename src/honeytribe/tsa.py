from typing import Literal, Self

import pandas as pd

class TimeSeriesData:
    def __init__(self, df: pd.DataFrame, time_column: str|None = None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f'Input data must be a pandas DataFrame, got {type(df)}')
        df = df.copy()
        if time_column is not None:
            df = df.set_index(time_column)
        self.df = df

        if self._index_dimension in ('time', 'numeric'):
            self.df.sort_index(inplace=True)

        self._init_frequency()

    @property
    def _index_dimension(self) -> Literal['time', 'numeric', 'unknown']:
        try:
            if isinstance(self.df.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.IntervalIndex)):
                return 'time'
            elif pd.api.types.is_timedelta64_dtype(self.df.index) or pd.api.types.is_timedelta64_ns_dtype(self.df.index):
                return 'time'
            elif pd.api.types.is_numeric_dtype(self.df.index):
                return 'numeric'
            else:
                print(f'Unknown dtype, got: {self.df.index.dtype}')
                return 'unknown'
        except AttributeError:
            raise ValueError(f'Unable to determine index dimension for index of type {type(self.df.index)}')

    def _init_frequency(self):
        # Guess frequency, if possible
        # Handle time indexes (DatetimeIndex, PeriodIndex)
        # Handle RangeIndex
        # Otherwise, mark as unknown and assumed regular
        if self._index_dimension == 'time':
            # IntervalIndex and TimedeltaIndex cannot be processed by infer_freq
            if isinstance(self.df.index, (pd.IntervalIndex, pd.TimedeltaIndex)):
                print(f"Warning: Index of type {type(self.df.index).__name__} does not support automatic frequency inference.")
                self.freq = 'unknown'
            else:
                inferred_freq = pd.infer_freq(self.df.index)
                if inferred_freq is None:
                    print("Warning: Time index has irregular intervals.")
                    self.freq = 'irregular time'
                else:
                    self.freq = inferred_freq
        elif self._index_dimension == 'numeric':
            if isinstance(self.df.index, pd.RangeIndex):
                self.freq = abs(self.df.index.step)
            else:
                diffs = self.df.index.to_series().diff().dropna().unique()
                if len(diffs) == 1:
                    self.freq = abs(diffs[0])
                else:
                    print("Warning: Numeric index has irregular intervals.")
                    self.freq = 'irregular numeric'
        elif isinstance(self.df.index, (pd.CategoricalIndex, pd.MultiIndex, pd.IntervalIndex, pd.TimedeltaIndex)):
            print(f"Warning: Index of type {type(self.df.index)} may not support frequency inference.")
            self.freq = 'unknown'
        else:
            print(f"Warning: Unable to infer frequency for index of type {type(self.df.index)}.")
            self.freq = 'unknown'
        if self.freq == 'unknown':
            print("Assuming regular time series with unknown frequency.")

    @property
    def is_regular(self) -> bool:
        return self.freq != 'irregular time' and self.freq != 'irregular numeric'

    def lag(self, lag):
        if self._index_dimension == 'time':
            if self.freq != 'irregular time':
                return self.df.shift(lag, freq=self.freq)
            else:
                raise ValueError('Cannot lag on irregular time index')
        elif self._index_dimension == 'numeric':
            if self.freq != 'irregular numeric':
                if lag % self.freq != 0:
                    raise ValueError(f'Can only lag on multiples of {self.freq}')
                out = self.df.copy()
                out.index += lag
                return out
            else:
                raise ValueError(f'Cannot lag on irregular numeric index')
        elif self._index_dimension == 'unknown':
            return self.df.shift(lag)
        else:
            raise NotImplementedError(f'No lag method implemented for _index_dimension {self._index_dimension}')

    @property
    def start(self):
        return self.df.index.min()

    @property
    def end(self):
        return self.df.index.max()

    def __getattr__(self, item):
        # Don't delegate private attributes or magic methods
        if item.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
        return getattr(self.df, item)

    def __len__(self):
        """Return the length of the DataFrame."""
        return len(self.df)

    def apply_rowwise(self, func) -> Self:
        """Apply a function row-wise to the DataFrame and return a new TimeSeriesData."""
        new_df = self.df.apply(func, axis=1)
        return TimeSeriesData(new_df)

    def fill_missing(
        self,
        method: Literal['mean', 'median', 'ffill', 'bfill', 'linear', 'spline'] = 'linear',
        order: int = 3,
        limit: int | None = None
    ) -> Self:
        """Fill missing values in the time series.

        Args:
            method: Filling method to use:
                - 'mean': Replace NaN with column mean
                - 'median': Replace NaN with column median
                - 'ffill': Forward fill from previous value
                - 'bfill': Backward fill from next value
                - 'linear': Linear interpolation
                - 'spline': Spline interpolation
            order: Order for spline interpolation (default: 3)
            limit: Maximum number of consecutive NaNs to fill

        Returns:
            New TimeSeriesData with missing values filled
        """
        df_filled = self.df.copy()

        if method == 'mean':
            df_filled = df_filled.fillna(df_filled.mean(), limit=limit)
        elif method == 'median':
            df_filled = df_filled.fillna(df_filled.median(), limit=limit)
        elif method == 'ffill':
            df_filled = df_filled.ffill(limit=limit)
        elif method == 'bfill':
            df_filled = df_filled.bfill(limit=limit)
        elif method == 'linear':
            df_filled = df_filled.interpolate(method='linear', limit=limit)
        elif method == 'spline':
            df_filled = df_filled.interpolate(method='spline', order=order, limit=limit)
        else:
            raise ValueError(f"Unknown filling method: {method}")

        return TimeSeriesData(df_filled)

    def smooth(
        self,
        method: Literal['rolling_mean', 'spline', 'kalman'] = 'rolling_mean',
        window: int = 3,
        center: bool = True,
        s: float | None = None,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2
    ) -> Self:
        """Apply smoothing to the time series.

        Args:
            method: Smoothing method to use:
                - 'rolling_mean': Rolling mean with centered window
                - 'spline': Spline smoothing
                - 'kalman': Kalman filter smoothing
            window: Window size for rolling mean (default: 3)
            center: Center the window for rolling mean (default: True)
            s: Smoothing parameter for spline (None = automatic)
            process_variance: Process noise covariance for Kalman filter
            measurement_variance: Measurement noise covariance for Kalman filter

        Returns:
            New TimeSeriesData with smoothed values
        """
        df_smoothed = self.df.copy()

        if method == 'rolling_mean':
            df_smoothed = df_smoothed.rolling(window=window, center=center, min_periods=1).mean()
        elif method == 'spline':
            from scipy.interpolate import UnivariateSpline
            for col in df_smoothed.columns:
                valid_mask = df_smoothed[col].notna()
                if valid_mask.sum() > 3:  # Need at least 4 points for spline
                    if self._index_dimension == 'time':
                        # Convert datetime index to numeric for spline fitting
                        x = (df_smoothed.index[valid_mask] - df_smoothed.index[0]).total_seconds()
                        x_all = (df_smoothed.index - df_smoothed.index[0]).total_seconds()
                    else:
                        x = df_smoothed.index[valid_mask].values
                        x_all = df_smoothed.index.values

                    y = df_smoothed.loc[valid_mask, col].values
                    spline = UnivariateSpline(x, y, s=s)
                    df_smoothed[col] = spline(x_all)
                else:
                    print(f"Warning: Column '{col}' has too few valid points for spline smoothing. Skipping.")
        elif method == 'kalman':
            for col in df_smoothed.columns:
                series = df_smoothed[col].values
                n = len(series)

                # Simple 1D Kalman filter implementation
                filtered = series.copy()
                P = 1.0  # Initial error covariance
                # Find first non-NaN value for initialization
                first_valid_idx = pd.Series(series).first_valid_index()
                if first_valid_idx is None:
                    continue  # Skip columns with all NaN
                x_hat = series[first_valid_idx]  # Initial state estimate

                for i in range(n):
                    if pd.isna(series[i]):
                        # If measurement is missing, just predict
                        filtered[i] = x_hat
                        P = P + process_variance
                        continue

                    # Prediction step (assuming state doesn't change)
                    x_hat_minus = x_hat
                    P_minus = P + process_variance

                    # Update step
                    K = P_minus / (P_minus + measurement_variance)  # Kalman gain
                    x_hat = x_hat_minus + K * (series[i] - x_hat_minus)
                    P = (1 - K) * P_minus

                    filtered[i] = x_hat

                df_smoothed[col] = filtered
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        return TimeSeriesData(df_smoothed)


