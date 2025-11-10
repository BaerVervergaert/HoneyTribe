import numpy as np
import pandas as pd

from honeytribe.metrics.correlation.standard import pearsonr, correlation_matrix, CorrelationOutput


def test_pearsonr_output_structure():
    rng = np.random.RandomState(0)
    a = rng.randn(100)
    b = a + 0.5 * rng.randn(100)
    out = pearsonr(a, b)
    assert isinstance(out, CorrelationOutput)
    assert out.name == "Pearson-r"
    assert -1 <= out.value <= 1
    assert out.p_value is not None


def test_correlation_matrix_symmetric_and_shape():
    rng = np.random.RandomState(1)
    A = pd.DataFrame(rng.randn(50, 3), columns=['a','b','c'])
    C = correlation_matrix(A, metric=pearsonr)
    assert C.symmetric is True
    assert C.value.shape == (3, 3)
    # symmetric matrix equality
    np.testing.assert_allclose(C.value, C.value.T, rtol=1e-12, atol=1e-12)
