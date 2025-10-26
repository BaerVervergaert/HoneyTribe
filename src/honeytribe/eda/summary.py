import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal

import honeytribe.metrics as mt
import honeytribe.metrics.centrality
import honeytribe.metrics.correlation
import honeytribe.metrics.correlation.standard
import honeytribe.metrics.variability
from honeytribe.metrics.correlation.standard import CorrelationMatrixOutput


class EDACentralitySummary:
    def __init__(self, df: pd.DataFrame, metrics=None):
        self.df = df
        if metrics is None:
            metrics = [
                mt.centrality.mean,
                mt.centrality.median,
            ]
        self.metrics = metrics
        self._saved_results = None
    def compute(self) -> dict[str, dict[str, float]]:
        results = {}
        for column in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise ValueError(f'Column {column} is not numeric and cannot be used for centrality metrics.')
            results[column] = {
                metric.__name__: metric(self.df[column].dropna().values) for metric in self.metrics
            }
        self._saved_results = results
        return self._saved_results
    def get_results(self):
        if self._saved_results is None:
            raise ValueError('Compute results first by calling the `.compute` method.')
        return self._saved_results
    def plot(self, fig_kwargs = None) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        if self._saved_results is None:
            raise ValueError('Compute results first by calling the `.compute` method.')
        if fig_kwargs is None:
            fig_kwargs = {}
        plots = {}
        for column in self.df.columns:
            fig, ax = plt.subplots()
            column_values = self.df[column].dropna().values
            metrics = self._saved_results[column]
            ax.hist(column_values, bins=30, alpha=0.5, label=column)
            for metric_name, metric_value in metrics.items():
                ax.axvline(metric_value, linestyle='--', label=f'{column} {metric_name}: {metric_value:.2f}')
            plots[column] = (fig, ax)
        return plots
    def table(self):
        if self._saved_results is None:
            raise ValueError('Compute results first by calling the `.compute` method.')
        table_data = []
        for column, metrics in self._saved_results.items():
            row = {'column': column}
            row.update(metrics)
            table_data.append(row)
        return pd.DataFrame(table_data)

class EDACorrelationSummary:
    def __init__(self, df: pd.DataFrame, metrics=None):
        self.df = df
        if metrics is None:
            metrics = [
                mt.correlation.standard.pearsonr,
                mt.correlation.standard.spearmanr,
                mt.correlation.standard.kendalltau,
                mt.correlation.standard.chatterjeexi,
            ]
        self.metrics = metrics
        self._saved_results = None
    def compute(self) -> dict[str, CorrelationMatrixOutput]:
        columns = self.df.select_dtypes(include=['number']).columns
        sub_df = self.df[columns]
        results = {
            metric.__name__: mt.correlation.standard.correlation_matrix(sub_df, metric=metric) for metric in self.metrics
        }
        self._saved_results = results
        return self._saved_results
    def get_results(self) -> dict[str, CorrelationMatrixOutput]:
        if self._saved_results is None:
            raise ValueError('Compute results first by calling the `.compute` method.')
        return self._saved_results
    def _serialize_matrix(self, matrix: CorrelationMatrixOutput, matrix_format: Literal['full', 'upper_triangle', None] = None) -> pd.Series:
        df = pd.DataFrame(matrix.value, index=matrix.row_index, columns=matrix.column_index)
        labels = []
        serialized_data = []
        for i, row_label in enumerate(df.index):
            for j, col_label in enumerate(df.columns):
                if matrix_format == 'upper_triangle' and j <= i:
                    continue
                index_label = f"{row_label} -> {col_label}"
                labels.append(index_label)
                serialized_data.append(df.iloc[i, j])
        return pd.Series(serialized_data, index=labels)
    def table(self, matrix_format: Literal['full', 'upper_triangle', None] = 'upper_triangle') -> pd.DataFrame:
        if self._saved_results is None:
            raise ValueError('Compute results first by calling the `.compute` method.')
        results = {}
        for metric_name, result in self._saved_results.items():
            series = self._serialize_matrix(result, matrix_format=matrix_format)
            results[metric_name] = series
        return pd.DataFrame(results)
    def plot(self) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        if self._saved_results is None:
            raise ValueError('Compute results first by calling the `.compute` method.')
        plots = {}
        for metric_name, result in self._saved_results.items():
            fig, ax = plt.subplots()
            cax = ax.matshow(result.value, cmap='coolwarm')
            fig.colorbar(cax)
            ax.set_xticks(range(len(result.column_index)))
            ax.set_yticks(range(len(result.row_index)))
            ax.set_xticklabels(result.column_index, rotation=90)
            ax.set_yticklabels(result.row_index)
            ax.set_title(f'Correlation Matrix: {metric_name}')
            plots[metric_name] = (fig, ax)
        return plots
