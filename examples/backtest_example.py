#!/usr/bin/env python3
"""
Backtesting demo: Set up a BacktestGame with synthetic EOD data and run a simple algorithm.

This example shows:
- Creating a time-indexed DataFrame of prices
- Building a BacktestGame with a rolling window and weekly cadence
- Using different prediction horizons (1-day ahead vs 5-day ahead)
- Implementing a minimal OnlineAlgorithm that predicts the next close as last close (persistence)
- Running Backtester to collect predictions and metrics
"""
import os
import sys
import numpy as np
import pandas as pd

# Ensure 'src' is on the Python path for local execution
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from honeytribe.backtesting import BacktestGame, Backtester
from honeytribe.online_convex_optimization.base import OnlineAlgorithm


def make_synthetic_prices(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range('2020-01-01', periods=n_days, freq='D')
    # simple random walk for prices
    returns = rng.normal(loc=0.0, scale=1.0, size=n_days)
    prices = 100 + np.cumsum(returns)
    df = pd.DataFrame({'close': prices}, index=idx)
    return df


class NaivePersistence(OnlineAlgorithm):
    """Predict next close as last observed close in the window."""

    def __init__(self):
        super().__init__()
        self.history_states = []
        self.history_y_trues = []

    def predict_step(self, state):
        # state is a DataFrame window
        try:
            return float(state['close'].iloc[-1])
        except Exception:
            return 0.0

    def update(self, state, prediction, loss, y_true=None):
        # no-op update for demo; track a counter
        self.t += 1
        self.history_states.append(state)
        self.history_y_trues.append(y_true)

    def update_regret(self, loss):
        # simple regret accumulation
        try:
            self.regret += float(loss)
        except Exception:
            pass


def main():
    df = make_synthetic_prices(n_days=150)

    print("="*80)
    print("Backtesting with Different Prediction Horizons")
    print("="*80)

    # Test with different horizons
    horizons = [
        ('1D', 'calendar', '1-day ahead'),
        ('5D', 'calendar', '5-day ahead'),
        ('3D', 'calendar', '3-day ahead'),
    ]

    results_dict = {}

    for horizon, horizon_unit, label in horizons:
        print(f"\n{label}:")
        print("-"*40)

        game = BacktestGame(
            data=df,
            window_length=20,
            cadence='7D',  # weekly cadence
            window_overlap=True,
            target_col='close',
            horizon=horizon,
        )

        algo = NaivePersistence()
        backtester = Backtester(game, price_col='close')

        result = backtester.run(algo, end='2020-05-31')
        results_dict[label] = (result, algo)

        print(f"Number of predictions: {len(result.predictions)}")
        print(f"Mean loss: {result.metrics['mean_loss']:.4f}")
        print("Last 3 predictions:")
        print(result.predictions.tail(3))

    # Plot prices and predictions if matplotlib is available
    if HAS_MPL:
        fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 4*len(horizons)), sharex=True)
        if len(horizons) == 1:
            axes = [axes]

        for ax, (label, (result, algo)) in zip(axes, results_dict.items()):
            # Plot close prices
            df['close'].plot(ax=ax, label='Close', color='C0', alpha=0.7)

            # Plot predictions
            for i, (idx, pred) in enumerate(result.predictions['prediction'].items()):
                ax.plot(idx, pred, 'ko', alpha=0.5, markersize=4, label='Prediction' if i == 0 else "")
                ax.text(idx, pred, f'{i}', fontsize=6, alpha=0.5, color='black')

            # Mark the window endpoints and targets
            for i, state, y_true in zip(range(len(algo.history_states)), algo.history_states, algo.history_y_trues):
                # Window end (where we make prediction from)
                window_end = state.index[-1]
                ax.axvline(window_end, color='black', linewidth=0.5)
                ax.plot(window_end, state['close'].iloc[-1], 'ro',
                       alpha=0.3, markersize=3, label='Window end' if 'Window end' not in ax.get_legend_handles_labels()[1] else "")
                ax.text(window_end, state['close'].iloc[-1], f'{i}', fontsize=6, alpha=0.5, color='red')

                # Target (what we're trying to predict)
                if y_true is not None:
                    # Find the target date by looking ahead
                    # Should be the first index where close == y_true after the window end
                    try:
                        target_idx = df.index[(df.index >= window_end) & (df['close'] == y_true)].min()
                        ax.plot(target_idx, y_true, 'go',
                               alpha=0.5, markersize=5, label='Target' if 'Target' not in ax.get_legend_handles_labels()[1] else "")
                        ax.text(target_idx, y_true, f'{i}', fontsize=6, alpha=0.5, color='green')
                    except Exception:
                        pass

            ax.set_title(f'Backtest: {label} (Weekly cadence, window=20, loss={result.metrics["mean_loss"]:.2f})')
            ax.set_ylabel('Price')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        out_path = os.path.join(THIS_DIR, 'backtest_example_horizons.png')
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved plot to {out_path}")
        plt.close()
    else:
        print("\nmatplotlib not installed; skipping plot. Install with: pip install 'honeytribe[visual]' or 'matplotlib'.")


if __name__ == '__main__':
    main()
