"""
Example demonstrating cleaning and smoothing operations on time series data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from honeytribe.tsa import TimeSeriesData

# Set random seed for reproducibility
np.random.seed(42)

# Create sample time series with missing values and noise
dates = pd.date_range('2024-01-01', periods=100, freq='D')
true_signal = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.linspace(0, 2, 100)
noise = np.random.normal(0, 0.2, 100)
noisy_signal = true_signal + noise

# Introduce some missing values
missing_indices = np.random.choice(100, 15, replace=False)
noisy_signal_with_gaps = noisy_signal.copy()
noisy_signal_with_gaps[missing_indices] = np.nan

# Create TimeSeriesData
df = pd.DataFrame({'value': noisy_signal_with_gaps}, index=dates)
ts = TimeSeriesData(df)

print("Original data:")
print(f"  - Total points: {len(ts.df)}")
print(f"  - Missing values: {ts.df['value'].isna().sum()}")
print(f"  - Mean: {ts.df['value'].mean():.4f}")
print(f"  - Std: {ts.df['value'].std():.4f}")

# Demonstrate different filling methods
print("\n" + "="*60)
print("FILLING METHODS")
print("="*60)

methods = ['mean', 'ffill', 'bfill', 'linear', 'spline']
filled_series = {}

for method in methods:
    ts_filled = ts.fill_missing(method=method)
    filled_series[method] = ts_filled
    print(f"\n{method.upper()}:")
    print(f"  - Missing values after: {ts_filled.df['value'].isna().sum()}")
    print(f"  - Mean: {ts_filled.df['value'].mean():.4f}")
    print(f"  - Std: {ts_filled.df['value'].std():.4f}")

# Demonstrate different smoothing methods
print("\n" + "="*60)
print("SMOOTHING METHODS")
print("="*60)

# First fill with linear interpolation
ts_filled = ts.fill_missing(method='linear')

smoothing_methods = [
    ('rolling_mean', {'window': 5, 'center': True}),
    ('rolling_mean', {'window': 10, 'center': True}),
    ('spline', {'s': 10.0}),
    ('kalman', {'process_variance': 1e-5, 'measurement_variance': 0.04})
]

smoothed_series = {}

for method, kwargs in smoothing_methods:
    method_name = f"{method}_{list(kwargs.values())[0]}" if method == 'rolling_mean' else method
    ts_smoothed = ts_filled.smooth(method=method, **kwargs)
    smoothed_series[method_name] = ts_smoothed
    print(f"\n{method_name.upper()}:")
    print(f"  - Mean: {ts_smoothed.df['value'].mean():.4f}")
    print(f"  - Std: {ts_smoothed.df['value'].std():.4f}")
    print(f"  - Reduction in std: {(1 - ts_smoothed.df['value'].std() / ts_filled.df['value'].std()) * 100:.1f}%")

# Demonstrate method chaining
print("\n" + "="*60)
print("METHOD CHAINING")
print("="*60)

ts_processed = ts.fill_missing(method='linear').smooth(method='rolling_mean', window=7, center=True)
print(f"\nChained (fill → smooth):")
print(f"  - Missing values: {ts_processed.df['value'].isna().sum()}")
print(f"  - Mean: {ts_processed.df['value'].mean():.4f}")
print(f"  - Std: {ts_processed.df['value'].std():.4f}")

# Create visualization
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Time Series Cleaning and Smoothing Operations', fontsize=16)

# Plot 1: Original with missing values
ax = axes[0, 0]
ax.plot(dates, noisy_signal_with_gaps, 'o-', alpha=0.5, label='With gaps', markersize=3)
ax.plot(dates, true_signal, 'r--', alpha=0.7, label='True signal', linewidth=2)
ax.set_title('Original Data (with missing values)')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Different filling methods
ax = axes[0, 1]
ax.plot(dates, ts.df['value'], 'ko', alpha=0.3, markersize=2, label='Original')
for method in ['linear', 'spline', 'mean']:
    ax.plot(dates, filled_series[method].df['value'], alpha=0.7, label=method, linewidth=1.5)
ax.set_title('Comparison of Filling Methods')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Rolling mean with different windows
ax = axes[1, 0]
ax.plot(dates, ts_filled.df['value'], 'gray', alpha=0.3, label='Filled (unsmoothed)', linewidth=1)
ax.plot(dates, smoothed_series['rolling_mean_5'].df['value'], label='Window=5', linewidth=2)
ax.plot(dates, smoothed_series['rolling_mean_10'].df['value'], label='Window=10', linewidth=2)
ax.plot(dates, true_signal, 'r--', alpha=0.5, label='True signal', linewidth=1)
ax.set_title('Rolling Mean (different windows)')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Spline smoothing
ax = axes[1, 1]
ax.plot(dates, ts_filled.df['value'], 'gray', alpha=0.3, label='Filled (unsmoothed)', linewidth=1)
ax.plot(dates, smoothed_series['spline'].df['value'], 'g', label='Spline (s=10)', linewidth=2)
ax.plot(dates, true_signal, 'r--', alpha=0.5, label='True signal', linewidth=1)
ax.set_title('Spline Smoothing')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Kalman filter
ax = axes[2, 0]
ax.plot(dates, ts_filled.df['value'], 'gray', alpha=0.3, label='Filled (unsmoothed)', linewidth=1)
ax.plot(dates, smoothed_series['kalman'].df['value'], 'purple', label='Kalman filter', linewidth=2)
ax.plot(dates, true_signal, 'r--', alpha=0.5, label='True signal', linewidth=1)
ax.set_title('Kalman Filter Smoothing')
ax.set_ylabel('Value')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Comparison of all smoothing methods
ax = axes[2, 1]
ax.plot(dates, ts_filled.df['value'], 'gray', alpha=0.2, label='Unsmoothed', linewidth=1)
ax.plot(dates, smoothed_series['rolling_mean_5'].df['value'], label='Rolling (w=5)', linewidth=1.5)
ax.plot(dates, smoothed_series['spline'].df['value'], label='Spline', linewidth=1.5)
ax.plot(dates, smoothed_series['kalman'].df['value'], label='Kalman', linewidth=1.5)
ax.plot(dates, true_signal, 'r--', alpha=0.5, label='True signal', linewidth=1)
ax.set_title('Comparison of Smoothing Methods')
ax.set_ylabel('Value')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('timeseries_cleaning_smoothing_demo.png', dpi=150, bbox_inches='tight')
print(f"\n\nVisualization saved to 'timeseries_cleaning_smoothing_demo.png'")
print("\nDemo complete!")

