from typing import Literal, Self

import pandas as pd


class TimeSeriesData:
    def __init__(self, df: pd.DataFrame, time_column: str|None = None):
        df = df.copy()
        if time_column is not None:
            df = df.set_index(time_column)
        self.df = df

        if self._index_dimension in ('time', 'numeric'):
            self.df.sort_index(inplace=True)

        self._init_frequency()

    @property
    def _index_dimension(self) -> Literal['time', 'numeric', 'unknown']:
        if isinstance(self.df.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.IntervalIndex)):
            return 'time'
        elif pd.api.types.is_timedelta64_dtype(self.df.index) or pd.api.types.is_timedelta64_ns_dtype(self.df.index):
            return 'time'
        elif pd.api.types.is_numeric_dtype(self.df.index):
            return 'numeric'
        else:
            print(f'Unknown dtype, got: {self.df.dtype}')
            return 'unknown'

    def _init_frequency(self):
        # Guess frequency, if possible
        # Handle time indexes (DatetimeIndex, PeriodIndex)
        # Handle RangeIndex
        # Otherwise, mark as unknown and assumed regular
        if self._index_dimension == 'time':
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
        return getattr(self.df, item)

    def apply_rowwise(self, func) -> Self:
        """Apply a function row-wise to the DataFrame and return a new TimeSeriesData."""
        new_df = self.df.apply(func, axis=1)
        return TimeSeriesData(new_df)
