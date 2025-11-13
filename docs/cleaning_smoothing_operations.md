# Time Series Cleaning and Smoothing Operations

## Summary

Added comprehensive cleaning and smoothing operations to the `TimeSeriesData` class in `src/honeytribe/tsa.py`.

## New Methods

### 1. `fill_missing()` - Missing Value Filling

Fills missing (NaN) values in time series data using various methods.

**Parameters:**
- `method`: Filling strategy
  - `'mean'`: Replace with column mean
  - `'median'`: Replace with column median
  - `'ffill'`: Forward fill from previous value
  - `'bfill'`: Backward fill from next value
  - `'linear'`: Linear interpolation
  - `'spline'`: Spline interpolation
- `order`: Spline order (default: 3)
- `limit`: Maximum consecutive NaNs to fill

**Returns:** New `TimeSeriesData` object with filled values

**Example:**
```python
ts_filled = ts.fill_missing(method='linear')
```

### 2. `smooth()` - Time Series Smoothing

Applies smoothing to reduce noise in time series data.

**Parameters:**
- `method`: Smoothing technique
  - `'rolling_mean'`: Centered rolling window average
  - `'spline'`: Univariate spline smoothing
  - `'kalman'`: Kalman filter (1D implementation)
- `window`: Window size for rolling mean (default: 3)
- `center`: Center the rolling window (default: True)
- `s`: Spline smoothing parameter (None = automatic)
- `process_variance`: Kalman filter process noise (default: 1e-5)
- `measurement_variance`: Kalman filter measurement noise (default: 1e-2)

**Returns:** New `TimeSeriesData` object with smoothed values

**Example:**
```python
ts_smoothed = ts.smooth(method='rolling_mean', window=5, center=True)
```

## Features

### Method Chaining
Both methods return new `TimeSeriesData` instances, allowing for easy chaining:

```python
ts_processed = ts.fill_missing(method='linear').smooth(method='kalman')
```

### Multi-Column Support
Both methods work seamlessly with DataFrames containing multiple columns.

### Index Type Support
- DateTime indices
- Numeric indices (RangeIndex, custom numeric)
- Handles irregular time series

### Immutability
Original data is never modified; all operations return new objects.

## Implementation Details

### Filling Methods
- **Mean/Median**: Uses pandas `fillna()` with aggregated statistics
- **Forward/Backward Fill**: Uses pandas `ffill()`/`bfill()`
- **Linear**: Uses pandas `interpolate(method='linear')`
- **Spline**: Uses pandas `interpolate(method='spline')` with configurable order

### Smoothing Methods
- **Rolling Mean**: Uses pandas `rolling().mean()` with configurable window and centering
- **Spline**: Uses scipy's `UnivariateSpline` with automatic datetime conversion
- **Kalman**: Custom 1D implementation with configurable noise parameters

## Testing

Comprehensive test suite in `tests/test_timeseries_cleaning_smoothing.py`:
- 26 unit tests covering all methods and edge cases
- Tests for method chaining
- Tests for multi-column data
- Tests for various index types
- Edge case handling (all NaN, single value, etc.)

All tests pass: ✓ 26/26

## Demo

Run `examples/timeseries_cleaning_smoothing_demo.py` to see:
- Comparison of all filling methods
- Comparison of all smoothing methods
- Method chaining examples
- Visual comparison with true signal
- Statistical comparison (mean, std, variance reduction)

## Use Cases

1. **Data Preparation**: Fill gaps before analysis
2. **Noise Reduction**: Smooth noisy sensor data
3. **Trend Detection**: Remove high-frequency noise to identify trends
4. **Forecasting**: Prepare clean data for prediction models
5. **Visualization**: Create cleaner plots for presentations

## Performance

- Efficient for large datasets (uses vectorized pandas/numpy operations)
- Kalman filter is O(n) complexity
- Spline fitting is O(n log n) complexity
- Rolling operations are O(n * window_size)

