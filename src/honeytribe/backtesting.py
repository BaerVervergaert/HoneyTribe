"""
Backtesting utilities re-using OnlineGame / DatasetGame building blocks.

Provides:
- BacktestGame: an OnlineGame presenting historical windows as state and stepping through data at a configurable cadence
- Backtester: orchestrator that runs strategies/algorithms over a BacktestGame and records predictions/trades/pnl
- BacktestResult: simple dataclass holding results

This implementation aims to be minimal and safe as a first pass; it's easy to extend with more sophisticated fill/padding/trade models later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable, Literal

import pandas as pd
import numpy as np

from .online_convex_optimization.base import OnlineGame, OnlineAlgorithm


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    trades: pd.DataFrame
    pnl: pd.Series
    metrics: Dict[str, float]
    meta: Dict[str, Any]


class BacktestGame(OnlineGame):
    """OnlineGame that yields a historical window (DataFrame) as the environment state.

    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed price/features table (index must be a DatetimeIndex)
    window_length : int
        Number of observations in each historical window (default: 1)
    cadence : Union[int, str, pd.Timedelta]
        How to move the window between steps. If int: number of index steps to advance.
        If str or Timedelta: calendar offset applied to the current step's timestamp.
        Default '1D' (daily) which will advance to the next available timestamp >= current + offset.
    window_overlap : bool
        If True, windows may overlap (the pointer moves by cadence but window length remains).
        If False and cadence is integer >= window_length, windows won't overlap.
    warmup_periods : int
        Number of initial steps counted as warm-up (info['is_warmup'] True)
    target_col : Optional[str]
        Column name to use for y_true (for compute_loss/get_y_true). If None, get_y_true returns None.
    horizon : Union[str, pd.Timedelta]
        Prediction horizon - how far ahead to predict the target. If int: number of days ahead.
        If str or Timedelta: time offset from current step (e.g., '5D' for 5 days ahead).
        Default "1D" (next day).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        window_length: int = 1,
        cadence: Union[int, str, pd.Timedelta] = "1D",
        window_overlap: bool = True,
        warmup_periods: int = 0,
        target_col: Optional[str] = None,
        window_unit: str = "observations",
        horizon: Union[str, pd.Timedelta] = "1D",
        line_up_cadence_with: Literal["front", "back"] = "front",
    ) -> None:
        # Validate and normalize data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception:
                raise ValueError("Data index must be a DatetimeIndex or convertible to datetime")

        # Sort index ascending
        data = data.sort_index()

        # Store
        self.data = data
        self.window_length = int(window_length)
        self.cadence = cadence
        self.window_overlap = bool(window_overlap)
        self.warmup_periods = int(warmup_periods)
        self.target_col = target_col
        self.window_unit = window_unit
        self.horizon = horizon
        self.line_up_cadence_with = line_up_cadence_with

        # Internal pointer - will be set on reset()
        self._step_dates: List[pd.Timestamp] = []
        self._step_pos = 0

        # Compute step dates immediately for simplicity
        self._compute_step_dates()

        # call OnlineGame ctor with n_steps
        super().__init__(n_steps=len(self._step_dates))

    def _compute_step_dates(self) -> None:
        """Precompute the list of timestamp "end points" for each step according to cadence.

        Behavior:
        - If cadence is int: treat as index-offset (advance pointer by that many observations)
        - If cadence is string or Timedelta: treat as calendar offset and choose the nearest index >= current + offset
        - If line_up_cadence_with is 'front', align the first step date with the start of the data
        - If line_up_cadence_with is 'back', align the last step date with the end of the data
        """
        # Handle empty data case
        idx = self.data.index
        if len(idx) == 0:
            self._step_dates = []
            return

        iter_forward = self.line_up_cadence_with == "front"
        if self.line_up_cadence_with == "front":
            start = 0
        else:
            start = len(idx) - 1

        step_dates = []

        current_pos = start
        current_date = idx[start]
        prev_ts = None
        date_min = min(idx)
        date_max = max(idx)
        while True:
            # Check if current_date can form a valid window
            skip_add = False
            if self.window_unit == "observations":
                if current_pos < self.window_length - 1:
                    skip_add = True
            else:
                # TODO: Implement calendar-based window check
                print("Skip window check for calendar-based windows (not implemented)")
                pass

            if not skip_add:
                step_dates.append(current_date)

            # Advance to next step
            if isinstance(self.cadence, int):
                if iter_forward:
                    current_pos += max(1, int(self.cadence))
                else:
                    current_pos -= max(1, int(self.cadence))

                if current_pos < 0 or current_pos >= len(idx):
                    break

                current_date = idx[current_pos]
            else:
                if isinstance(self.cadence, str):
                    offset = pd.tseries.frequencies.to_offset(self.cadence)
                else:
                    offset = pd.Timedelta(self.cadence)
                if iter_forward:
                    next_date = current_date + offset
                else:
                    next_date = current_date - offset

                if next_date < date_min or next_date > date_max:
                    break

                ts = self._find_first_index_at_or_after(next_date)

                if ts == prev_ts or ts is None:
                    raise ValueError("Help! Stuck in an infinite loop computing step dates.")

                current_date = ts
                current_pos = idx.get_loc(current_date)

                prev_ts = ts

        self._step_dates = sorted(step_dates)

    def reset(self) -> None:
        """Reset internal pointer."""
        self._step_pos = 0
        # Reset OnlineGame counters
        super().reset()

    def available_steps(self) -> int:
        """Return the total number of available steps.

        Note: this is NOT the number of remaining steps from the current position.
        Use remaining_steps() for that.
        """
        return len(self._step_dates)

    def remaining_steps(self) -> int:
        """Return the number of remaining steps from the current position."""
        return len(self._step_dates) - self._step_pos

    def step_date(self, idx: int) -> Optional[pd.Timestamp]:
        """Return the step date at the given index, or None if out of bounds."""
        if 0 <= idx < len(self._step_dates):
            return self._step_dates[idx]
        return None

    @property
    def current_date(self) -> Optional[pd.Timestamp]:
        """Return the current step date, or None if out of bounds."""
        return self.step_date(self._step_pos)

    def steps_generator(self) -> Iterable[int]:
        for i in range(0, len(self._step_dates)):
            self._step_pos = i
            yield i

    def _find_first_index_at_or_before(self, ts: pd.Timestamp|int) -> Optional[pd.Timestamp|int]:
        """Helper to find the first index at or before the given timestamp or integer position."""
        if isinstance(ts, int):
            if ts < 0 or ts >= len(self.data):
                return None
            return ts
        else:
            idx = self.data.index
            pos = idx.get_indexer([ts], method="pad")[0]
            if pos == -1 or pos >= len(idx):
                return None
            return idx[pos]

    def get_window(self, at_index_or_timestamp: Union[int, pd.Timestamp]) -> pd.DataFrame:
        """Return the historical window ending at the given integer index or timestamp.

        If an int is provided it is interpreted as an exact index in the data used with data.iloc.
        """
        end_pos = self._find_first_index_at_or_before(at_index_or_timestamp)

        # Build window
        if self.window_unit == "observations":
            if isinstance(end_pos, int):
                start_pos = max(0, end_pos - (self.window_length - 1))
                window = self.data.iloc[start_pos : end_pos + 1]
            else:
                # find position of end_pos in index
                pos = self.data.index.get_loc(end_pos)
                start_pos = max(0, pos - (self.window_length - 1))
                window = self.data.iloc[start_pos : pos + 1]
        else:
            # calendar window: include rows within [end_ts - window_length, end_ts]
            delta = pd.Timedelta(self.window_length)
            start_pos = end_pos - delta
            window = self.data.loc[start_pos:end_pos]

        return window.copy()

    def get_state(self) -> pd.DataFrame:
        """Return the window for the current step pointer."""
        if self._step_pos >= len(self._step_dates):
            raise StopIteration("No more steps available")
        if self.current_date is None:
            raise ValueError("Current date is None; cannot get state")
        return self.get_window(self.current_date)

    def _add_horizon(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Helper to add horizon to a timestamp or integer position."""
        if isinstance(self.horizon, str):
            horizon_delta = pd.tseries.frequencies.to_offset(self.horizon)
        else:
            horizon_delta = pd.Timedelta(self.horizon)
        return ts + horizon_delta

    def _find_first_index_at_or_after(self, ts: pd.Timestamp|int) -> Optional[pd.Timestamp|int]:
        """Helper to find the first index at or after the given timestamp or integer position."""
        if isinstance(ts, int):
            if ts < 0 or ts >= len(self.data):
                return None
            return ts
        else:
            idx = self.data.index
            pos = idx.get_indexer([ts], method="backfill")[0]
            if pos == -1 or pos >= len(idx):
                return None
            return idx[pos]

    def get_y_true(self) -> Any:
        """Return the 'true' target for the current step, if target_col configured.

        The target is computed at the horizon ahead from the current step date.
        """
        if self.target_col is None:
            return None

        # find the current step date
        current_date = self.current_date
        if current_date is None:
            return None

        horizon_ts = self._add_horizon(current_date)
        target_pos = self._find_first_index_at_or_after(horizon_ts)

        if target_pos is None:
            return None

        # retrieve target value
        if isinstance(target_pos, int):
            if target_pos < 0 or target_pos >= len(self.data):
                return None
            row = self.data.iloc[target_pos]
        else:
            if target_pos not in self.data.index:
                return None
            row = self.data.loc[target_pos]
        return row.get(self.target_col, None)

    def step(self, algorithm: 'OnlineAlgorithm') -> Tuple[Any, float]:
        """Perform a single step: get state, predict, compute loss, update algorithm."""
        prediction, loss = super().step(algorithm)

        # find the current step date
        if self._step_pos >= len(self._step_dates):
            return prediction, loss # Exit early if no more steps

        current_date = self.current_date
        if current_date is None:
            return prediction, loss

        # Find the target timestamp based on horizon
        pred_pos = self._add_horizon(current_date)

        self.prediction_index = pred_pos
        return prediction, loss


    def compute_loss(self, prediction: Any, state: Any = None) -> Union[float, np.ndarray]:
        """Compute loss using target_col if present. Default: squared loss on target_col.

        If no target configured, returns 0.0.
        """
        if self.target_col is None:
            return 0.0

        y_true = self.get_y_true()
        if y_true is None:
            return 0.0

        try:
            return 0.5 * (y_true - prediction) ** 2
        except Exception:
            return 0.0


class Backtester:
    """Simple orchestration for running algorithms on a BacktestGame.

    This is intentionally lightweight: it runs one algorithm at a time, records predictions
    (one value per step) and returns a BacktestResult. Trade simulation is minimal and
    configurable later via pluggable models.
    """

    def __init__(self, game: BacktestGame, price_col: Optional[str] = None):
        self.game = game
        self.price_col = price_col

    def run(self, algorithm: OnlineAlgorithm, end: Optional[pd.Timestamp] = None) -> BacktestResult:
        """Run a single algorithm over the game's steps and collect predictions.

        Returns BacktestResult with predictions DataFrame indexed by step timestamps and a placeholder trades/pnl.
        """
        # Reset game (position to start if provided)
        self.game.reset()

        predictions: List[Tuple[pd.Timestamp, Any]] = []
        losses: List[float] = []

        # We won't manipulate the algorithm's internal fit state beyond calling step-loop
        for _ in self.game.steps_generator():
            pred, loss = self.game.step(algorithm)
            ts = self.game.prediction_index
            predictions.append((ts, pred))

            losses.append(float(loss) if isinstance(loss, (int, float, np.floating, np.integer)) else 0.0)

            # optionally break if end provided
            if end is not None and ts >= pd.to_datetime(end):
                break

        # construct outputs
        pred_index = pd.DatetimeIndex([t for t, _ in predictions])
        pred_values = [p for _, p in predictions]
        predictions_df = pd.DataFrame({"prediction": pred_values}, index=pred_index)

        trades_df = pd.DataFrame(columns=["timestamp", "side", "size", "price"])  # placeholder
        pnl_series = pd.Series(index=pred_index, data=np.zeros(len(pred_index)))

        mean_loss = float(np.nanmean(losses)) if losses else 0.0
        if np.isnan(mean_loss):
            mean_loss = 0.0
        metrics: Dict[str, float] = {"mean_loss": mean_loss}
        meta: Dict[str, Any] = {"n_steps": len(predictions_df), "price_col": self.price_col}

        return BacktestResult(predictions=predictions_df, trades=trades_df, pnl=pnl_series, metrics=metrics, meta=meta)


# End of module
