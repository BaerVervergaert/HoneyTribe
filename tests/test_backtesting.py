import numpy as np
import pandas as pd

from honeytribe.backtesting import BacktestGame, Backtester


class DummyAlgorithm:
    """Very small algorithm that provides predict_step and update methods.

    predict_step(state): returns the last value of the first column (or mean of state) as prediction.
    update(...) is a no-op but included to ensure Backtester calls it safely.
    """

    def predict_step(self, state):
        if state is None:
            return None
        if isinstance(state, pd.DataFrame) and len(state) > 0:
            # return last value of first column
            return float(state.iloc[-1, 0])
        # if flattened array
        arr = np.asarray(state)
        if arr.size == 0:
            return None
        return float(arr.flat[-1])

    def update(self, state, pred, loss, y_true=None):
        # no-op update
        self.last_update = dict(loss=loss, y_true=y_true)


def make_daily_df(n=30):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({"close": np.arange(n, dtype=float), "feat": np.arange(n, dtype=float) * 0.1}, index=idx)
    return df


def test_backtestgame_basic_step():
    df = make_daily_df(30)
    game = BacktestGame(df, window_length=5, cadence=7, horizon='1D')
    # There should be precomputed step dates
    assert game.available_steps() > 0

    # iterate steps and ensure windows have at most window_length rows
    game.reset()
    steps_seen = 0
    for step_idx in game.steps_generator():
        window = game.get_window(game.step_date(step_idx))
        assert isinstance(window, pd.DataFrame)
        assert 1 <= len(window) <= 5
        # the last timestamp of the window equals the step date
        step_ts = game._step_dates[step_idx]
        print(window.index)
        assert window.index[-1] == step_ts
        steps_seen += 1

    assert steps_seen == game.available_steps()


def test_backtester_run_basic():
    df = make_daily_df(20)
    # include a target_col so compute_loss can run
    df = df.copy()
    df['target'] = df['close'].shift(-1)

    game = BacktestGame(df, window_length=2, cadence=1, target_col='target')
    alg = DummyAlgorithm()
    bt = Backtester(game, price_col='close')

    result = bt.run(alg)

    # predictions length should equal number of steps performed
    assert len(result.predictions) == result.meta['n_steps']
    # mean_loss must be present and non-negative
    assert 'mean_loss' in result.metrics
    assert result.metrics['mean_loss'] >= 0.0
    # trades and pnl must be dataframes/series with expected types
    assert hasattr(result, 'trades')
    assert hasattr(result, 'pnl')


def test_backtester_y_true_advances():
    """Test that y_true changes at each step (not stuck on same value)."""
    df = make_daily_df(30)
    df = df.copy()
    df['target'] = df['close'].shift(-1)

    game = BacktestGame(df, window_length=5, cadence=3, target_col='target')

    class YTrueCollector:
        def __init__(self):
            self.y_trues = []

        def predict_step(self, state):
            return 0.0

        def update(self, state, pred, loss, y_true=None):
            self.y_trues.append(y_true)

    alg = YTrueCollector()
    bt = Backtester(game, price_col='close')
    result = bt.run(alg)

    # All y_true values should be different (unique at each step)
    # Filter out None values
    y_trues = [y for y in alg.y_trues if y is not None]
    assert len(y_trues) > 3, "Should have collected multiple y_true values"
    assert len(set(y_trues)) == len(y_trues), "All y_true values should be unique (advancing)"


def test_horizon_observations_basic():
    """Test that horizon in observations mode returns the correct future value."""
    df = make_daily_df(30)

    # Test with horizon=1 (default, next observation)
    game1 = BacktestGame(df, window_length=5, cadence=1, target_col='close', horizon='1D')
    game1.reset()

    # At step 0 (day 4 since window_length=5), horizon=1 should give us day 5
    y_true1 = game1.get_y_true()
    assert y_true1 == 5.0, f"Expected 5.0, got {y_true1}"

    # Test with horizon=3 (3 observations ahead)
    game3 = BacktestGame(df, window_length=5, cadence=1, target_col='close', horizon='3D')
    game3.reset()

    # At step 0 (day 4), horizon=3 should give us day 7
    y_true3 = game3.get_y_true()
    assert y_true3 == 7.0, f"Expected 7.0, got {y_true3}"


def test_horizon_observations_multiple_steps():
    """Test that horizon works consistently across multiple steps."""
    df = make_daily_df(30)
    horizon = '5D'

    game = BacktestGame(df, window_length=3, cadence=1, target_col='close', horizon=horizon)
    game.reset()

    # Collect targets for first few steps
    targets = []
    for i, step_idx in enumerate(game.steps_generator()):
        if i >= 5:  # test first 5 steps
            break
        game._step_pos = step_idx
        y_true = game.get_y_true()
        if y_true is not None:
            targets.append(y_true)

    # Each target should be horizon steps ahead of the window end
    # Step 0: window ends at day 2, horizon=5 -> target is day 7 (value 7.0)
    # Step 1: window ends at day 3, horizon=5 -> target is day 8 (value 8.0)
    # etc.
    assert len(targets) >= 3
    assert targets[0] == 7.0
    assert targets[1] == 8.0
    assert targets[2] == 9.0


def test_horizon_calendar_basic():
    """Test that horizon in calendar mode returns the correct future value."""
    df = make_daily_df(30)

    # Test with horizon='2D' (2 days ahead)
    game = BacktestGame(df, window_length=5, cadence='1D', target_col='close', horizon='2D')
    game.reset()
    game._step_pos = 0

    # At step 0 (day 4), horizon='2D' should give us day 6
    y_true = game.get_y_true()
    assert y_true == 6.0, f"Expected 6.0, got {y_true}"


def test_horizon_calendar_basic_line_up_with_end():
    """Test that horizon in calendar mode returns the correct future value."""
    df = make_daily_df(30)
    print('df', df)

    # Test with horizon='2D' (2 days ahead)
    game = BacktestGame(df, window_length=5, cadence='1D', target_col='close', horizon='2D', line_up_cadence_with="back")
    game.reset()
    game._step_pos = 0

    # At step 0 (day 4), horizon='2D' should give us day 6
    y_true = game.get_y_true()
    assert y_true == 6.0, f"Expected 6.0, got {y_true}"


def test_horizon_calendar_multiple_steps():
    """Test that calendar-based horizon works consistently across multiple steps."""
    df = make_daily_df(30)
    horizon = '3D'

    game = BacktestGame(df, window_length=3, cadence='1D', target_col='close', horizon=horizon)
    game.reset()

    # Collect targets for first few steps
    targets = []
    for i, step_idx in enumerate(game.steps_generator()):
        if i >= 5:  # test first 5 steps
            break
        game._step_pos = step_idx
        y_true = game.get_y_true()
        if y_true is not None:
            targets.append(y_true)

    # Each target should be 3 days ahead of the step date
    # Step 0: ends at day 2 (2020-01-03), +3D -> day 5 (2020-01-06, value 5.0)
    # Step 1: ends at day 3 (2020-01-04), +3D -> day 6 (2020-01-07, value 6.0)
    assert len(targets) >= 3
    assert targets[0] == 5.0
    assert targets[1] == 6.0
    assert targets[2] == 7.0


def test_horizon_none_at_end():
    """Test that horizon returns None when the target would be beyond available data."""
    df = make_daily_df(20)
    horizon = 10

    game = BacktestGame(df, window_length=5, cadence=1, target_col='close', horizon=horizon)
    game.reset()

    # Near the end of the data, y_true should become None
    targets_with_none = []
    for step_idx in game.steps_generator():
        game._step_pos = step_idx
        y_true = game.get_y_true()
        targets_with_none.append(y_true)

    # Should have some None values at the end
    assert None in targets_with_none, "Expected some None values when horizon exceeds data"
    # Should have some non-None values at the beginning
    assert any(t is not None for t in targets_with_none), "Expected some non-None values"


def test_horizon_with_backtester():
    """Test that Backtester properly uses horizon in loss computation."""
    df = make_daily_df(30)
    horizon = '5D'

    game = BacktestGame(df, window_length=3, cadence=1, target_col='close', horizon=horizon)

    class HorizonTestAlgorithm:
        def __init__(self):
            self.predictions = []
            self.y_trues = []

        def predict_step(self, state):
            # Predict the last observed value
            return float(state.iloc[-1, 0])

        def update(self, state, pred, loss, y_true=None):
            self.predictions.append(pred)
            self.y_trues.append(y_true)

    alg = HorizonTestAlgorithm()
    bt = Backtester(game, price_col='close')
    result = bt.run(alg)

    # Check that predictions and y_trues are offset by horizon
    # Filter out None values
    valid_pairs = [(p, y) for p, y in zip(alg.predictions, alg.y_trues) if y is not None]
    assert len(valid_pairs) > 5

    # First prediction is based on window ending at day 2, predicts value 2.0
    # First y_true should be day 7 (horizon=5 ahead), value 7.0
    assert valid_pairs[0][0] == 2.0
    assert valid_pairs[0][1] == 7.0


def test_horizon_timedelta():
    """Test that horizon can be specified as a pd.Timedelta."""
    df = make_daily_df(30)
    horizon = pd.Timedelta(days=4)

    game = BacktestGame(df, window_length=5, cadence='1D', target_col='close', horizon=horizon)
    game.reset()
    game._step_pos = 0

    # At step 0 (day 4), horizon=4 days should give us day 8
    y_true = game.get_y_true()
    assert y_true == 8.0, f"Expected 8.0, got {y_true}"


def test_get_y_true_correctness():
    """Test that get_y_true retrieves the correct value at the expected index.

    This test verifies that when we request y_true with a specific horizon,
    we get the value at exactly that horizon offset, not one position after.
    """
    # Create a simple dataframe where values equal their index position
    # This makes it easy to verify we're getting the right value
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "value": np.arange(n, dtype=float),  # value[i] = i
        "index_position": np.arange(n, dtype=float)  # explicit position tracker
    }, index=idx)

    # Test with horizon='1D' (1 day ahead)
    game = BacktestGame(df, window_length=5, cadence='1D', target_col='value', horizon='1D')
    game.reset()

    # Manually verify each step
    for step_idx in range(min(10, game.available_steps())):
        game._step_pos = step_idx
        current_date = game.current_date

        # Calculate expected position
        # With window_length=5, first step is at index 4 (day 2020-01-05)
        # With horizon='1D', we should get value at index (4 + step_idx) + 1
        expected_value = float(4 + step_idx + 1)

        y_true = game.get_y_true()

        # Verify we got the correct value
        assert y_true == expected_value, \
            f"Step {step_idx}: Expected value {expected_value} at current_date {current_date}, got {y_true}"

    # Test with horizon='3D' (3 days ahead)
    game3 = BacktestGame(df, window_length=5, cadence='1D', target_col='value', horizon='3D')
    game3.reset()

    for step_idx in range(min(5, game3.available_steps())):
        game3._step_pos = step_idx
        current_date = game3.current_date

        # Expected: window ends at index (4 + step_idx), horizon adds 3 days
        expected_value = float(4 + step_idx + 3)

        y_true = game3.get_y_true()

        # Verify we got the correct value
        assert y_true == expected_value, \
            f"Step {step_idx} with 3D horizon: Expected value {expected_value} at current_date {current_date}, got {y_true}"

    # Test with horizon='7D' (1 week ahead)
    game7 = BacktestGame(df, window_length=3, cadence='1D', target_col='value', horizon='7D')
    game7.reset()

    for step_idx in range(min(5, game7.available_steps())):
        game7._step_pos = step_idx
        current_date = game7.current_date

        # Expected: window ends at index (2 + step_idx), horizon adds 7 days
        expected_value = float(2 + step_idx + 7)

        y_true = game7.get_y_true()

        # Verify we got the correct value
        assert y_true == expected_value, \
            f"Step {step_idx} with 7D horizon: Expected value {expected_value} at current_date {current_date}, got {y_true}"

    # Test edge case: verify the actual date being accessed
    game_check = BacktestGame(df, window_length=5, cadence='1D', target_col='value', horizon='1D')
    game_check.reset()
    game_check._step_pos = 0

    current_date = game_check.current_date
    horizon_date = current_date + pd.Timedelta(days=1)
    expected_iloc_position = df.index.get_loc(horizon_date)
    expected_value_from_df = df.iloc[expected_iloc_position]['value']

    y_true = game_check.get_y_true()

    assert y_true == expected_value_from_df, \
        f"Direct check: Expected {expected_value_from_df} from position {expected_iloc_position}, got {y_true}"


def test_get_y_true_exact_index_matching():
    """Test that get_y_true returns values from the exact expected dates, not off-by-one.

    This test creates explicit date-value mappings and verifies each retrieval.
    """
    # Create data where the value encodes both the date and position
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    values = [100.0 + i for i in range(20)]  # 100.0, 101.0, 102.0, ...
    df = pd.DataFrame({"target": values}, index=dates)

    # With window_length=1 and cadence='1D', each step corresponds to consecutive dates
    game = BacktestGame(df, window_length=1, cadence='1D', target_col='target', horizon='1D')
    game.reset()

    # Verify first few steps explicitly
    test_cases = [
        (0, "2020-01-01", "2020-01-02", 101.0),  # Step 0: current=2020-01-01, target=2020-01-02 -> value 101.0
        (1, "2020-01-02", "2020-01-03", 102.0),  # Step 1: current=2020-01-02, target=2020-01-03 -> value 102.0
        (2, "2020-01-03", "2020-01-04", 103.0),  # Step 2: current=2020-01-03, target=2020-01-04 -> value 103.0
    ]

    for step_idx, expected_current, expected_horizon, expected_value in test_cases:
        game._step_pos = step_idx
        current_date = game.current_date
        y_true = game.get_y_true()

        assert current_date == pd.Timestamp(expected_current), \
            f"Step {step_idx}: Expected current date {expected_current}, got {current_date}"

        assert y_true == expected_value, \
            f"Step {step_idx}: Current date {current_date}, expected y_true={expected_value}, got {y_true}. " \
            f"Should retrieve value from {expected_horizon}"


