from dataclasses import dataclass
from numbers import Number

import pandas as pd
import scipy.stats as st
import numpy as np

from honeytribe.metrics.utils import _name_function

@dataclass
class CorrelationOutput:
    name: str
    value: float
    p_value: float|None = None
    hypothesis_test: str|None = None
    hypothesis_test_description: str|None = None
    hypothesis_test_notes: str|None = None

@dataclass
class CorrelationMatrixOutput:
    name: str
    value: list[list[float]]
    p_value: list[list[float]]
    hypothesis_test: str|None = None
    hypothesis_test_description: str|None = None
    hypothesis_test_notes: str|None = None
    column_index: list[str]|None = None
    row_index: list[str]|None = None
    symmetric: bool = False

@dataclass
class AutoCorrelationOutput:
    name: str
    value: list[float]
    p_value: list[float]
    lags: list[int]
    hypothesis_test: str|None = None
    hypothesis_test_description: str|None = None
    hypothesis_test_notes: str|None = None

@dataclass
class AutoCorrelationMatrixOutput:
    name: str
    value: list[list[float]]
    p_value: list[list[float]]
    lags: list[int]
    hypothesis_test: str|None = None
    hypothesis_test_description: str|None = None
    hypothesis_test_notes: str|None = None
    column_index: list[str]|None = None

def _symmetric(symmetric: bool = True):
    def func(f):
        f._symmetric = symmetric
        return f
    return func

@_name_function('kendalltau')
@_symmetric()
def kendalltau(a, b):
    tau, p_value = st.kendalltau(a, b)
    return CorrelationOutput(
        name = "Kendall-Tau",
        value = tau,
        p_value = p_value,
        hypothesis_test = "H_0: tau = 0",
        hypothesis_test_description = """Test if there is a monotonic relation between the variables."""
    )

@_name_function('pearsonr')
@_symmetric()
def pearsonr(a, b):
    r, p_value = st.pearsonr(a, b)
    return CorrelationOutput(
        name = "Pearson-r",
        value = r,
        p_value = p_value,
        hypothesis_test = "H_0: r = 0",
        hypothesis_test_description = """Test if there is a linear trend between the variables."""
    )

@_name_function('spearmanr')
@_symmetric()
def spearmanr(a, b):
    r, p_value = st.spearmanr(a, b)
    return CorrelationOutput(
        name = "Spearman-r",
        value = r,
        p_value = p_value,
        hypothesis_test = "H_0: r = 0",
        hypothesis_test_description = """Test if there is a monotonic relation between the variables."""
    )

@_name_function('chatterjeexi')
@_symmetric(False)
def chatterjeexi(a, b):
    """Chatterjee's xi correlation coefficient.

    Rank-based correlation coefficient that can detect both linear, monotonic and non-monotonic relationships. See: https://arxiv.org/pdf/1909.10140

    Note: This correlation coefficient has three important properties (when dataset size converges to infinity):
    1. It always lies between 0 and 1.
    2. It is 0 if and only if the variables are independent.
    3. It is 1 if and only if there is a measurable function f such that Y = f(X) almost surely.
    """
    xi, p_value = st.chatterjeexi(a, b)
    return CorrelationOutput(
        name = "Chatterjee-xi",
        value = xi,
        p_value = p_value,
        hypothesis_test = "H_0: xi = 0",
        hypothesis_test_description = """Test if the variables are not indepedent. Low p-value implies no independence, and high p-value implies not enough evidence to refute independence (likely independent or not enough data)."""
    )

@_name_function('somersd')
def somersd(contingency_table):
    d, p_value = st.somersd(contingency_table)
    return CorrelationOutput(
        name = "Somers-d",
        value = d,
        p_value = p_value,
        hypothesis_test = "H_0: d = 0",
        hypothesis_test_description = """Test if the variables are ordinally associated."""
    )

@_name_function('stepanovr')
@_symmetric()
def stepanovr(a, b):
    """Stepanov's r correlation coefficient.

    Based on Kendall's tau and Spearman's rho. See: https://arxiv.org/pdf/2405.16469

    Stepanov has introduced an improved version, which is not yet implemented. See: https://arxiv.org/pdf/2506.06056
    """
    r = (3 * kendalltau(a, b).value - spearmanr(a, b).value) / 2
    return CorrelationOutput(
        name = "Stepanov-r",
        value = r,
    )

def relative_model_improvement_coefficient(a, b, baseline):
    """Generalization of Pearson's r to measure relative model improvement.

    If a is the predictions of a candidate model, b the actual values, and baseline the predictions of a baseline model, then this metric indicates the additional information provided by the candidate model over the baseline model.

    It is a normalized version of:
    mean((a - baseline) * (b - baseline))

    Particularly, it is computed as:
    mean((a - baseline) * (b - baseline)) / (||a - baseline|| * ||b - baseline||)

    As an exercise to reinforce our intuition, consider the following cases:
    1. If a == b, then the metric is 1.
    2. If a == baseline, then the metric is 0.
    3. If sign(a - baseline) == sign(b - baseline) always, then the metric is positive.
    4. If sign(a - baseline) != sign(b - baseline) always, then the metric is negative.
    5. If a - baseline is orthogonal to b - baseline, then the metric is 0.
    6. If the coefficient is significantly negative, then the candidate model can be used to improve the baseline model by inverting its predictions, as follows:
        a_improved = baseline - c*(a - baseline), where c is some positive scaling factor that can be tuned on a validation set.
    """
    a_err = a - baseline
    b_err = b - baseline
    a_norm = np.sqrt(np.mean(a_err**2))
    b_norm = np.sqrt(np.mean(b_err**2))
    return np.mean(a_err * b_err) / (a_norm * b_norm)

def error_correlation_for_model_improvement(a, b, baseline, metric=pearsonr):
    """Correlation of errors for model improvement.

    If a is the predictions of a candidate model, b the actual values, and baseline the predictions of a baseline model, then this metric indicates how correlated the errors of the candidate model are with the errors of the actual values, both measured with respect to the baseline model.

    Note: This metric can be used in various ways, for example:
    1. To evaluate the potential of a candidate feature to improve a baseline model.
        Choose a candidate feature (a), and take the predictions of the current model (baseline) along with the actual values (b). Then, use the chatterjeexi correlation.
        The chatterjeexi correlation will determine if there is a reasonable relation between the errors of the current model and the error introduced by the candidate feature.
        This computes chatterjeexi(a - baseline, b - baseline), which is 0 if and only if the errors are independent, and 1 if and only if there is a measurable function mapping the errors a - baseline onto the errors b - baseline.
        An additional step you can compute chatterjeexi(a, b - baseline) to see if the candidate feature is directly related to the error of the current model.
    """
    a_err = a - baseline
    b_err = b - baseline
    return metric(a_err, b_err)

def correlation_matrix(A: np.ndarray|pd.DataFrame, B: np.ndarray|pd.DataFrame|None = None, metric=pearsonr) -> CorrelationMatrixOutput:
    if B is None:
        B = A
        symmetric = getattr(metric, '_symmetric', False)
    else:
        symmetric = False
    column_index = getattr(A, 'columns', None)
    row_index = getattr(B, 'columns', None)
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(B, pd.DataFrame):
        B = B.to_numpy()
    n1, m1 = A.shape
    n2, m2 = B.shape
    assert n1 == n2, 'Input arrays must have the same number of rows.'
    C = np.zeros((m1, m2))
    P = np.zeros((m1, m2))
    P[:, :] = np.nan
    name = None
    hypothesis_test = None
    hypothesis_test_description = None
    hypothesis_test_notes = None
    for i in range(m1):
        for j in range(m2):
            if symmetric and j < i:
                C[i, j] = C[j, i]
                P[i, j] = P[j, i]
            else:
                corr_value = metric(A[:, i], B[:, j])
                if isinstance(corr_value, CorrelationOutput):
                    C[i, j] = corr_value.value
                    P[i, j] = corr_value.p_value
                    name = corr_value.name
                    hypothesis_test = corr_value.hypothesis_test
                    hypothesis_test_description = corr_value.hypothesis_test_description
                    hypothesis_test_notes = corr_value.hypothesis_test_notes
                elif isinstance(corr_value, Number):
                    C[i, j] = corr_value
                else:
                    raise ValueError(f'Cannot handle return value {corr_value} of type {type(corr_value)}.')
    return CorrelationMatrixOutput(
        name = name,
        value = C,
        p_value = P,
        hypothesis_test = hypothesis_test,
        hypothesis_test_description = hypothesis_test_description,
        hypothesis_test_notes = hypothesis_test_notes,
        column_index = column_index,
        row_index = row_index,
        symmetric = symmetric,
    )

def auto_correlation_series(a: np.ndarray|pd.Series, lags: int|list[int], metric=pearsonr) -> AutoCorrelationOutput:
    if isinstance(a, pd.Series):
        a = a.to_numpy()
    n = len(a)
    values = []
    p_values = []
    name = None
    hypothesis_test = None
    hypothesis_test_description = None
    hypothesis_test_notes = None
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    lags = sorted(lags)
    for lag in lags:
        a_lagged = a[lag:]
        a_original = a[:n - lag]
        corr_value = metric(a_original, a_lagged)
        if isinstance(corr_value, CorrelationOutput):
            value = corr_value.value
            p_value = corr_value.p_value
            name = corr_value.name
            values.append(value)
            p_values.append(p_value)
            hypothesis_test = corr_value.hypothesis_test
            hypothesis_test_description = corr_value.hypothesis_test_description
            hypothesis_test_notes = corr_value.hypothesis_test_notes
    return AutoCorrelationOutput(
        name = name,
        value = values,
        p_value = p_values,
        lags = lags,
        hypothesis_test=hypothesis_test,
        hypothesis_test_description=hypothesis_test_description,
        hypothesis_test_notes=hypothesis_test_notes
    )

def auto_correlation_matrix(A: np.ndarray|pd.DataFrame, lags: int|list[int], metric=pearsonr) -> AutoCorrelationMatrixOutput:
    column_index = getattr(A, 'columns', None)
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    n, m = A.shape
    results = {}
    for i in range(m):
        series = A[:, i]
        ac_output = auto_correlation_series(series, lags, metric)
        results[i] = ac_output
    C = np.zeros((len(results[0].value),m))
    P = np.zeros((len(results[0].p_value),m))
    for i in range(m):
        C[:, i] = results[i].value
        P[:, i] = results[i].p_value
    return AutoCorrelationMatrixOutput(
        name = results[0].name,
        value = C,
        p_value = P,
        lags = results[0].lags,
        hypothesis_test = results[0].hypothesis_test,
        hypothesis_test_description = results[0].hypothesis_test_description,
        hypothesis_test_notes = results[0].hypothesis_test_notes,
        column_index = column_index,
    )