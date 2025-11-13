"""Unit tests for time series cleaning and smoothing operations."""

import numpy as np
import pandas as pd
import pytest

from honeytribe.tsa import TimeSeriesData


class TestFillMissing:
    """Tests for fill_missing method."""

    def test_fill_missing_mean(self):
        """Test filling missing values with mean."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='mean')

        expected_mean = (1.0 + 3.0 + 5.0) / 3
        assert ts_filled.df['value'].to_list() == [1.0, expected_mean, 3.0, expected_mean, 5.0]
        assert not ts_filled.df['value'].isna().any()

    def test_fill_missing_median(self):
        """Test filling missing values with median."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='median')

        expected_median = 3.0
        assert ts_filled.df['value'].to_list() == [1.0, expected_median, 3.0, expected_median, 5.0]
        assert not ts_filled.df['value'].isna().any()

    def test_fill_missing_ffill(self):
        """Test forward fill."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, np.nan, 4.0, 5.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='ffill')

        assert ts_filled.df['value'].to_list() == [1.0, 1.0, 1.0, 4.0, 5.0]
        assert not ts_filled.df['value'].isna().any()

    def test_fill_missing_bfill(self):
        """Test backward fill."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, np.nan, 4.0, 5.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='bfill')

        assert ts_filled.df['value'].to_list() == [1.0, 4.0, 4.0, 4.0, 5.0]
        assert not ts_filled.df['value'].isna().any()

    def test_fill_missing_linear(self):
        """Test linear interpolation."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, np.nan, 4.0, 5.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='linear')

        assert ts_filled.df['value'].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert not ts_filled.df['value'].isna().any()

    def test_fill_missing_spline(self):
        """Test spline interpolation."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 5.0, np.nan, 5.0, 3.0, 2.0]
        }, index=pd.date_range('2024-01-01', periods=7))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='spline', order=3)

        # Spline interpolation should fill the gaps smoothly
        assert not ts_filled.df['value'].isna().any()
        # Values should be between neighbors
        assert 1.0 < ts_filled.df['value'].iloc[1] < 5.0
        assert ts_filled.df['value'].iloc[3] > 5.0

    def test_fill_missing_limit(self):
        """Test filling with a limit on consecutive NaNs."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, np.nan, np.nan, 5.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='ffill', limit=2)

        assert ts_filled.df['value'].iloc[1] == 1.0
        assert ts_filled.df['value'].iloc[2] == 1.0
        assert pd.isna(ts_filled.df['value'].iloc[3])  # Should not be filled due to limit

    def test_fill_missing_multiple_columns(self):
        """Test filling missing values in multiple columns."""
        df = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0, np.nan, 5.0],
            'col2': [2.0, np.nan, 4.0, np.nan, 8.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='linear')

        assert not ts_filled.df['col1'].isna().any()
        assert not ts_filled.df['col2'].isna().any()
        assert ts_filled.df['col1'].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert ts_filled.df['col2'].to_list() == [2.0, 3.0, 4.0, 6.0, 8.0]

    def test_fill_missing_invalid_method(self):
        """Test that invalid method raises ValueError."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0]
        }, index=pd.date_range('2024-01-01', periods=3))

        ts = TimeSeriesData(df)
        with pytest.raises(ValueError, match="Unknown filling method"):
            ts.fill_missing(method='invalid')

    def test_fill_missing_returns_new_instance(self):
        """Test that fill_missing returns a new TimeSeriesData instance."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0]
        }, index=pd.date_range('2024-01-01', periods=3))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='mean')

        assert isinstance(ts_filled, TimeSeriesData)
        assert ts is not ts_filled
        # Original should be unchanged
        assert pd.isna(ts.df['value'].iloc[1])


class TestSmooth:
    """Tests for smooth method."""

    def test_smooth_rolling_mean(self):
        """Test rolling mean smoothing."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        ts_smoothed = ts.smooth(method='rolling_mean', window=3, center=True)

        # With center=True and min_periods=1:
        # Index 0: mean([1, 2]) = 1.5 (only 2 values available when centered)
        # Index 1: mean([1, 2, 3]) = 2
        # Index 2: mean([2, 3, 4]) = 3
        # Index 3: mean([3, 4, 5]) = 4
        # Index 4: mean([4, 5]) = 4.5 (only 2 values available when centered)
        assert ts_smoothed.df['value'].iloc[0] == 1.5
        assert ts_smoothed.df['value'].iloc[1] == 2.0
        assert ts_smoothed.df['value'].iloc[2] == 3.0
        assert ts_smoothed.df['value'].iloc[3] == 4.0
        assert ts_smoothed.df['value'].iloc[4] == 4.5

    def test_smooth_rolling_mean_no_center(self):
        """Test rolling mean without centering."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        ts_smoothed = ts.smooth(method='rolling_mean', window=3, center=False)

        # With center=False:
        # Index 0: mean([1]) = 1
        # Index 1: mean([1, 2]) = 1.5
        # Index 2: mean([1, 2, 3]) = 2
        # Index 3: mean([2, 3, 4]) = 3
        # Index 4: mean([3, 4, 5]) = 4
        assert ts_smoothed.df['value'].iloc[0] == 1.0
        assert ts_smoothed.df['value'].iloc[1] == 1.5
        assert ts_smoothed.df['value'].iloc[2] == 2.0
        assert ts_smoothed.df['value'].iloc[3] == 3.0
        assert ts_smoothed.df['value'].iloc[4] == 4.0

    def test_smooth_spline(self):
        """Test spline smoothing."""
        # Create noisy data
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = np.sin(x) + np.random.normal(0, 0.1, 20)

        df = pd.DataFrame({
            'value': y
        }, index=pd.RangeIndex(len(y)))

        ts = TimeSeriesData(df)
        ts_smoothed = ts.smooth(method='spline', s=1.0)

        # Smoothed values should exist
        assert not ts_smoothed.df['value'].isna().any()
        # Smoothing should reduce variance
        assert ts_smoothed.df['value'].var() < ts.df['value'].var()

    def test_smooth_spline_datetime_index(self):
        """Test spline smoothing with datetime index."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=20)
        y = np.sin(np.linspace(0, 10, 20)) + np.random.normal(0, 0.1, 20)

        df = pd.DataFrame({
            'value': y
        }, index=dates)

        ts = TimeSeriesData(df)
        ts_smoothed = ts.smooth(method='spline', s=1.0)

        assert not ts_smoothed.df['value'].isna().any()
        assert ts_smoothed.df['value'].var() < ts.df['value'].var()

    def test_smooth_spline_too_few_points(self):
        """Test spline smoothing with too few points."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0]
        }, index=pd.date_range('2024-01-01', periods=3))

        ts = TimeSeriesData(df)
        # Should print warning but not crash
        ts_smoothed = ts.smooth(method='spline')

        # Values should be unchanged when too few points
        pd.testing.assert_frame_equal(ts.df, ts_smoothed.df)

    def test_smooth_kalman(self):
        """Test Kalman filter smoothing."""
        # Create noisy data
        np.random.seed(42)
        true_value = 5.0
        noise = np.random.normal(0, 0.5, 20)
        measurements = true_value + noise

        df = pd.DataFrame({
            'value': measurements
        }, index=pd.date_range('2024-01-01', periods=20))

        ts = TimeSeriesData(df)
        ts_smoothed = ts.smooth(method='kalman', process_variance=1e-5, measurement_variance=0.25)

        # Smoothed values should be closer to true value
        assert not ts_smoothed.df['value'].isna().any()
        # Kalman filter should reduce variance
        assert ts_smoothed.df['value'].var() < ts.df['value'].var()
        # Mean should be close to true value
        assert abs(ts_smoothed.df['value'].mean() - true_value) < 0.5

    def test_smooth_kalman_with_missing(self):
        """Test Kalman filter with missing values."""
        np.random.seed(42)
        measurements = [5.0, 5.1, np.nan, 4.9, 5.2, np.nan, np.nan, 5.0]

        df = pd.DataFrame({
            'value': measurements
        }, index=pd.date_range('2024-01-01', periods=len(measurements)))

        ts = TimeSeriesData(df)
        ts_smoothed = ts.smooth(method='kalman')

        # All values should be filled
        assert not ts_smoothed.df['value'].isna().any()
        # Values should be reasonable estimates
        assert all(4.5 < v < 5.5 for v in ts_smoothed.df['value'])

    def test_smooth_multiple_columns(self):
        """Test smoothing multiple columns."""
        df = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'col2': [5.0, 4.0, 3.0, 2.0, 1.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        ts_smoothed = ts.smooth(method='rolling_mean', window=3)

        assert not ts_smoothed.df['col1'].isna().any()
        assert not ts_smoothed.df['col2'].isna().any()
        # Both columns should be smoothed
        assert ts_smoothed.df['col1'].iloc[2] == 3.0
        assert ts_smoothed.df['col2'].iloc[2] == 3.0

    def test_smooth_invalid_method(self):
        """Test that invalid method raises ValueError."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0]
        }, index=pd.date_range('2024-01-01', periods=3))

        ts = TimeSeriesData(df)
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            ts.smooth(method='invalid')

    def test_smooth_returns_new_instance(self):
        """Test that smooth returns a new TimeSeriesData instance."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0]
        }, index=pd.date_range('2024-01-01', periods=3))

        ts = TimeSeriesData(df)
        ts_smoothed = ts.smooth(method='rolling_mean', window=2)

        assert isinstance(ts_smoothed, TimeSeriesData)
        assert ts is not ts_smoothed


class TestMethodChaining:
    """Tests for chaining fill_missing and smooth operations."""

    def test_fill_and_smooth(self):
        """Test chaining fill_missing and smooth."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0]
        }, index=pd.date_range('2024-01-01', periods=7))

        ts = TimeSeriesData(df)
        ts_processed = ts.fill_missing(method='linear').smooth(method='rolling_mean', window=3)

        assert not ts_processed.df['value'].isna().any()
        assert isinstance(ts_processed, TimeSeriesData)

    def test_smooth_and_fill(self):
        """Test smoothing first (which might create NaNs at edges) then filling."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        }, index=pd.date_range('2024-01-01', periods=5))

        ts = TimeSeriesData(df)
        # Smooth first, then fill any potential edge effects
        ts_processed = ts.smooth(method='rolling_mean', window=3).fill_missing(method='bfill')

        assert not ts_processed.df['value'].isna().any()
        assert isinstance(ts_processed, TimeSeriesData)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_nan_column(self):
        """Test handling of all-NaN columns."""
        df = pd.DataFrame({
            'value': [np.nan, np.nan, np.nan]
        }, index=pd.date_range('2024-01-01', periods=3))

        ts = TimeSeriesData(df)
        # Mean/median should leave NaNs
        ts_filled = ts.fill_missing(method='mean')
        assert ts_filled.df['value'].isna().all()

    def test_no_missing_values(self):
        """Test fill_missing when no values are missing."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0]
        }, index=pd.date_range('2024-01-01', periods=3))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='linear')

        pd.testing.assert_frame_equal(ts.df, ts_filled.df)

    def test_single_value(self):
        """Test operations with single value."""
        df = pd.DataFrame({
            'value': [1.0]
        }, index=pd.RangeIndex(1))

        ts = TimeSeriesData(df)
        ts_smoothed = ts.smooth(method='rolling_mean', window=3)

        assert ts_smoothed.df['value'].iloc[0] == 1.0

    def test_numeric_index(self):
        """Test operations with numeric index."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, 4.0, 5.0]
        }, index=pd.RangeIndex(5))

        ts = TimeSeriesData(df)
        ts_filled = ts.fill_missing(method='linear')
        ts_smoothed = ts_filled.smooth(method='rolling_mean', window=3)

        assert not ts_smoothed.df['value'].isna().any()
        assert isinstance(ts_smoothed, TimeSeriesData)

