import scipy.stats as st
import numpy as np

def kendalltau(a, b):
    return st.kendalltau(a, b)[0]

def pearsonr(a, b):
    return st.pearsonr(a, b)[0]

def spearmanr(a, b):
    return st.spearmanr(a, b)[0]

def chatterjeexi(a, b):
    return st.chatterjeexi(a, b)[0]

def somersd(contingency_table):
    return st.somersd(contingency_table)

def chatterjeexi(a, b):
    """Chatterjee's xi correlation coefficient.

    Rank-based correlation coefficient that can detect both linear, monotonic and non-monotonic relationships. See: https://arxiv.org/pdf/1909.10140

    Note: This correlation coefficient has three important properties (when dataset size converges to infinity):
    1. It always lies between 0 and 1.
    2. It is 0 if and only if the variables are independent.
    3. It is 1 if and only if there is a measurable function f such that Y = f(X) almost surely.
    """
    return st.chatterjeexi(a, b)[0]

def stepanovr(a, b):
    """Stepanov's r correlation coefficient.

    Based on Kendall's tau and Spearman's rho. See: https://arxiv.org/pdf/2405.16469
    """
    return (3 * kendalltau(a, b) - spearmanr(a, b)) / 2

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

def correlation_matrix(A, B=None, metric=pearsonr):
    if B is None:
        B = A
        symmetric = True
    else:
        symmetric = False
    n1, m1 = A.shape
    n2, m2 = B.shape
    assert n1 == n2
    C = np.zeros((m1, m2))
    for i in range(m1):
        for j in range(m2):
            if symmetric and j < i:
                C[i, j] = C[j, i]
            else:
                C[i, j] = metric(A[:, i], B[:, j])
    return C