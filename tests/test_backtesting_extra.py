import numpy as np
import pandas as pd
import pytest

from honeytribe.backtesting import BacktestGame


def make_daily_df(n=30):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({"close": np.arange(n, dtype=float), "feat": np.arange(n, dtype=float) * 0.1}, index=idx)
    return df

def test_calendar_cadence_7d_has_weekly_steps():
    df = make_daily_df(30)
    game = BacktestGame(df, window_length=1, cadence="7D")
    # ensure at least a couple steps
    step_dates = game._step_dates
    assert len(step_dates) >= 3
    # Check that consecutive step dates are roughly a week apart (>=6 days)
    diffs = [step_dates[i + 1] - step_dates[i] for i in range(len(step_dates) - 1)]
    for d in diffs:
        assert d >= pd.Timedelta("6D")


def test_target_col_last_step_no_target_and_loss_zero():
    df = make_daily_df(10)
    df = df.copy()
    df["target"] = df["close"].shift(-1)

    game = BacktestGame(df, window_length=2, cadence=1, target_col="target")
    # iterate to final step
    last_idx = len(game._step_dates) - 1
    # move pointer to last step and check y_true
    y_true_last = game.get_window(last_idx)
    # get_y_true should be None for last step because target is NaN
    # Use get_y_true via setting internal pointer and calling the method
    game._step_pos = last_idx
    assert game.get_y_true() is None
    assert game.compute_loss(1.23) == 0.0


def test_overlapping_windows_share_rows():
    df = make_daily_df(12)
    # cadence smaller than window_length to force overlap
    game = BacktestGame(df, window_length=5, cadence=2, window_overlap=True)
    assert len(game._step_dates) >= 3
    w0 = game.get_window(0)
    w1 = game.get_window(1)
    # windows should overlap: intersection of indices > 0
    inter = w0.index.intersection(w1.index)
    assert len(inter) > 0

