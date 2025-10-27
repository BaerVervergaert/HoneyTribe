import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Any

from networkx.algorithms.bipartite import density

import honeytribe.metrics as mt
import honeytribe.metrics.centrality
import honeytribe.metrics.correlation
import honeytribe.metrics.correlation.standard
import honeytribe.metrics.variability
import honeytribe.metrics.variability.standard
from honeytribe.metrics.correlation.standard import CorrelationMatrixOutput
from honeytribe.metrics.utils import _name_function

class EDASummaryBase:
    def __init__(self, name: str, df: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.df = df
    def plot(self):
        raise NotImplementedError("No .plot method implemented.")
    def table(self):
        raise NotImplementedError("No .table method implemented.")
    def compute(self):
        ...

class SavedResultsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_results = None
    @property
    def results_available(self) -> bool:
        return self._saved_results is not None
    def set_results(self, results: Any):
        self._saved_results = results
    def get_results(self):
        return self._saved_results

class EDACentralitySummary(EDASummaryBase, SavedResultsMixin):
    def __init__(self, df: pd.DataFrame, metrics=None):
        super().__init__("Centrality", df)
        if metrics is None:
            metrics = [
                mt.centrality.mean,
                mt.centrality.median,
            ]
        self.metrics = metrics
    def compute(self) -> dict[str, dict[str, float]]:
        results = {}
        columns = self.df.select_dtypes(include=['number']).columns
        for column in columns:
            results[column] = {
                metric.__name__: metric(self.df[column].dropna().values) for metric in self.metrics
            }
        self.set_results(results)
        return self.get_results()
    def get_results(self):
        if (res := super().get_results()) is None:
            raise ValueError('Compute results first by calling the `.compute` method.')
        return res
    def plot(self, fig_kwargs:dict[str, Any]|None = None) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        if fig_kwargs is None:
            fig_kwargs = {}
        plots = {}
        for column in self.df.columns:
            fig, ax = plt.subplots(**fig_kwargs)
            column_values = self.df[column].dropna().values
            metrics = self.get_results()[column]
            ax.hist(column_values, bins=30, alpha=0.5, label=column, density=True)
            for metric_name, metric_value in metrics.items():
                ax.axvline(metric_value, linestyle='--', label=f'{column} {metric_name}: {metric_value:.2f}')
            ax.set_title(f'Histogram of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Density')
            ax.legend()
            plots[column] = (fig, ax)
        return plots
    def table(self):
        table_data = []
        for column, metrics in self.get_results().items():
            row = {'column': column}
            row.update(metrics)
            table_data.append(row)
        return pd.DataFrame(table_data)

class EDACorrelationSummary(EDASummaryBase, SavedResultsMixin):
    def __init__(self, df: pd.DataFrame, metrics=None):
        super().__init__("Correlation", df)
        if metrics is None:
            metrics = [
                mt.correlation.standard.pearsonr,
                mt.correlation.standard.spearmanr,
                mt.correlation.standard.kendalltau,
                mt.correlation.standard.chatterjeexi,
            ]
        self.metrics = metrics
    def compute(self) -> dict[str, CorrelationMatrixOutput]:
        columns = self.df.select_dtypes(include=['number']).columns
        sub_df = self.df[columns]
        results = {
            metric.__name__: mt.correlation.standard.correlation_matrix(sub_df, metric=metric) for metric in self.metrics
        }
        self.set_results(results)
        return self.get_results()
    def get_results(self) -> dict[str, CorrelationMatrixOutput]:
        if (res := super().get_results()) is None:
            raise ValueError('Compute results first by calling the `.compute` method.')
        return res
    def _serialize_matrix(self, matrix: CorrelationMatrixOutput) -> pd.Series:
        df = pd.DataFrame(matrix.value, index=matrix.row_index, columns=matrix.column_index)
        labels = []
        serialized_data = []
        symmetric = matrix.symmetric
        for i, row_label in enumerate(df.index):
            for j, col_label in enumerate(df.columns):
                if j == i:
                    continue
                if symmetric and j < i:
                    continue
                index_label = f"{row_label} > {col_label}"
                labels.append(index_label)
                serialized_data.append(df.iloc[i, j])
        return pd.Series(serialized_data, index=labels)
    def table(self, matrix_format: Literal['full', 'upper_triangle', None] = 'upper_triangle') -> pd.DataFrame:
        results = {}
        for metric_name, result in self.get_results().items():
            series = self._serialize_matrix(result)
            results[metric_name] = series
        return pd.DataFrame(results)
    def plot(self, fig_kwargs: dict[str, Any]|None = None) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        if fig_kwargs is None:
            fig_kwargs = {}
        plots = {}
        for metric_name, result in self.get_results().items():
            fig, ax = plt.subplots(**fig_kwargs)
            cax = ax.matshow(result.value, cmap='coolwarm')
            for (i, j), z in np.ndenumerate(result.value):
                ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
            fig.colorbar(cax)
            if result.column_index is not None:
                ax.set_xticks(range(len(result.column_index)))
                ax.set_xticklabels(result.column_index, rotation=90)
            if result.row_index is not None:
                ax.set_yticks(range(len(result.row_index)))
                ax.set_yticklabels(result.row_index)
            ax.set_title(f'Correlation Matrix: {metric_name}')
            plots[metric_name] = (fig, ax)
        columns = self.df.select_dtypes(include=['number']).columns
        for i, col_label in enumerate(columns):
            for j, col_label2 in enumerate(columns):
                if i >= j:
                    continue
                fig, ax = plt.subplots()
                ax.scatter(self.df[col_label], self.df[col_label2], alpha=0.5)
                ax.set_xlabel(col_label)
                ax.set_ylabel(col_label2)
                ax.set_title(f"Scatter Plot: x = '{col_label}' vs y = '{col_label2}'")
                label = f'{col_label} vs {col_label2}'
                plots[label] = (fig, ax)
        return plots

class EDAVariabilitySummary(EDASummaryBase, SavedResultsMixin):
    def __init__(self, df: pd.DataFrame, metrics=None):
        super().__init__("Variability", df)
        if metrics is None:
            metrics = [
                mt.variability.standard.std,
                _name_function('quantile_spread-90')(lambda a, sample_weight=None: mt.variability.standard.quantile_spread(a, 0.9, sample_weight)),
                mt.variability.standard.iqr,
            ]
        self.metrics = metrics
    def compute(self) -> dict[str, dict[str, float]]:
        results = {}
        columns = self.df.select_dtypes(include=['number']).columns
        for column in columns:
            results[column] = {
                metric.__name__: metric(self.df[column].dropna().values) for metric in self.metrics
            }
        self.set_results(results)
        return self.get_results()
    def get_results(self):
        if (res := super().get_results()) is None:
            raise ValueError('Compute results first by calling the `.compute` method.')
        return res
    def table(self):
        table_data = []
        for column, metrics in self.get_results().items():
            row = {'column': column}
            row.update(metrics)
            table_data.append(row)
        return pd.DataFrame(table_data)
    def plot(self, fig_kwargs:dict[str, Any]|None = None) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        if fig_kwargs is None:
            fig_kwargs = {}
        plots = {}
        for column in self.df.columns:
            fig, ax = plt.subplots(**fig_kwargs)
            column_values = self.df[column].dropna().values
            ax.boxplot(column_values, vert=True)
            ax.set_title(f'Boxplot of {column}')
            plots[column] = (fig, ax)
        return plots