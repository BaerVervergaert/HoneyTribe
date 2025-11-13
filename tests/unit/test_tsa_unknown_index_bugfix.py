"""
Test to verify the bug fix for accessing _index_dimension with unknown index types.

The bug occurred because when _index_dimension property was accessed and returned 'unknown',
and the print statement tried to access self.df.index.dtype, the __getattr__ method would
incorrectly intercept attribute access for underscore-prefixed attributes.
"""

import pandas as pd
import pytest
from honeytribe.tsa import TimeSeriesData


def test_unknown_index_dimension_property_access():
    """Test that _index_dimension property can be safely accessed for unknown index types."""
    # Create a DataFrame with a string index (unknown type)
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])
    
    # This should not raise an error
    ts = TimeSeriesData(df)
    
    # Verify the dimension is correctly identified as unknown
    assert ts._index_dimension == 'unknown'
    
    # Verify we can still access regular DataFrame attributes
    assert ts.shape == (4, 1)
    assert list(ts.columns) == ['value']


def test_unknown_index_initialization_no_recursion():
    """Test that __init__ doesn't cause recursion through __getattr__ for unknown indices."""
    # These should all initialize without errors
    test_cases = [
        pd.DataFrame({'val': [1, 2, 3]}, index=['a', 'b', 'c']),  # String index
        pd.DataFrame({'val': [1, 2, 3]}, index=pd.CategoricalIndex(['x', 'y', 'z'])),  # Categorical
        pd.DataFrame({'val': [1, 2, 3]}, index=pd.MultiIndex.from_tuples([('a', 1), ('b', 2), ('c', 3)])),  # MultiIndex
    ]
    
    for df in test_cases:
        ts = TimeSeriesData(df)
        assert ts._index_dimension == 'unknown'
        assert ts.freq == 'unknown'
        assert len(ts) == 3


def test_private_attribute_access_raises_attribute_error():
    """Test that accessing non-existent private attributes raises AttributeError."""
    df = pd.DataFrame({'val': [1, 2, 3]}, index=['a', 'b', 'c'])
    ts = TimeSeriesData(df)
    
    # Accessing a non-existent private attribute should raise AttributeError
    with pytest.raises(AttributeError, match="'TimeSeriesData' object has no attribute '_nonexistent'"):
        _ = ts._nonexistent


def test_public_attribute_delegates_to_df():
    """Test that public attributes are correctly delegated to the underlying DataFrame."""
    df = pd.DataFrame({'val': [1, 2, 3]}, index=['a', 'b', 'c'])
    ts = TimeSeriesData(df)
    
    # These should delegate to df
    assert ts.shape == df.shape
    assert list(ts.columns) == list(df.columns)
    assert ts.dtypes.equals(df.dtypes)
    
    # Test method delegation
    assert ts.sum()['val'] == df.sum()['val']
    assert ts.mean()['val'] == df.mean()['val']


def test_magic_methods_work_correctly():
    """Test that magic methods like __len__ work correctly."""
    df = pd.DataFrame({'val': [1, 2, 3, 4, 5]}, index=['a', 'b', 'c', 'd', 'e'])
    ts = TimeSeriesData(df)
    
    # __len__ should work
    assert len(ts) == 5
    
    # Should be able to iterate (through delegation)
    count = 0
    for _ in ts.iterrows():
        count += 1
    assert count == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

