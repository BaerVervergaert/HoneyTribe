import matplotlib.pyplot as plt
from typing import Callable, Any

from honeytribe.eda.summary import EDASummaryBase, SavedResultsMixin
from honeytribe.metrics.timeseries import *

class TimeSeriesCentralitySummary(EDASummaryBase, SavedResultsMixin):
    """Summary of centrality metrics for time series data.

    Attributes:
        name (str): The name of the summary.
        data (TimeSeriesData): The time series data to analyze.
        metrics (list): List of centrality metric functions to compute.
    """

    def __init__(self, data: TimeSeriesData, metrics=None):
        super().__init__(name="Time Series Centrality Summary")
        self.data = data
        if metrics is None:
            metrics = [
                exponentially_weighted_mean,
            ]
        self.metrics = metrics
        self._results = None
    def compute(self):
        """Compute the centrality metrics for the time series data."""
        self._results = {}
        for metric in self.metrics:
            result = exponentially_weighted_transform_estimator(
                self.data,
                transform=lambda row: metric(row.values),
                p=0.9
            )
            self._results[metric.__name__] = result
    def plot(self, fig_kwargs:dict[str, Any]|None = None) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        if fig_kwargs is None:
            fig_kwargs = {}
        plots = {}
        for column in self.data.df.columns:
            fig, ax = plt.subplots(**fig_kwargs)
            ax.plot(self.data.df.index, self.data.df[column], label='Original', alpha=0.5)
            for metric_name, result in self._results.items():
                if isinstance(result, TimeSeriesData) and column in result.df.columns:
                    ax.plot(result.df.index, result.df[column], label=metric_name)
                elif isinstance(result, pd.DataFrame) and column in result.columns:
                    ax.plot(result.index, result[column], label=metric_name)
                else:
                    print(f'Warning: Result for metric {metric_name} does not contain column {column}.')
            ax.set_title(f'Time Series Centrality Metrics for {column}')
            ax.set_xlabel('Time')
            ax.set_ylabel(column)
            ax.legend()
            # Fix format timeseries x-axis with time index
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')))
            ax.xaxis.rotate(45)
            plots[column] = (fig, ax)
        return plots
