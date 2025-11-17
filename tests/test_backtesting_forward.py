import pandas as pd
import numpy as np
from honeytribe.backtesting import BacktestGame


def make_daily_df(n=40):
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame({"price": 100 + np.arange(n)}, index=idx)


def test_forward_only_integer_cadence():
    df = make_daily_df(30)
    game = BacktestGame(df, window_length=5, cadence=3)  # advance by index steps
    previous_date = None
    for step_idx in game.steps_generator():
        current_date = game.step_date(step_idx)
        if previous_date is not None:
            assert current_date > previous_date, "Step dates must strictly increase"
        previous_date = current_date
        w = game.get_window(current_date)
        assert w.index[-1] == current_date
        assert all(w.index <= current_date)


def test_forward_only_calendar_cadence():
    df = make_daily_df(60)
    game = BacktestGame(df, window_length=10, cadence="7D")  # weekly calendar cadence
    previous_date = None
    for step_idx in game.steps_generator():
        current_date = game.step_date(step_idx)
        if previous_date is not None:
            assert current_date > previous_date, "Step dates must strictly increase"
        previous_date = current_date
        w = game.get_window(current_date)
        assert w.index[-1] == current_date
        assert all(w.index <= current_date)

