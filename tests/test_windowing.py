"""Tests for the windowing module (Nodo 2 of the RUL pipeline).

Uses motor 1 from the C-MAPSS FD001 clean dataset as the primary test
fixture — a real train motor with 192 cycles and evento=1 at the last
cycle. Synthetic DataFrames are used for edge case and error path tests
where real data is unnecessary.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.windowing import MotorData, MotorWindows, build_windows, flatten_windows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def motor_1_df() -> pd.DataFrame:
    """Loads motor 1 from the clean dataset and adds unit_number column.

    Motor 1 is a train motor with 192 cycles and evento=1 only at the
    last cycle (after the dataset correction).
    """
    df = pd.read_csv('data/clean/data_motor_1.csv')
    df.insert(0, 'unit_number', 1)
    return df


@pytest.fixture(scope='module')
def window_size() -> int:
    """Default window size used across all tests."""
    return 30


@pytest.fixture(scope='module')
def clipping_threshold() -> int:
    """Default clipping threshold used across all tests."""
    return 125


@pytest.fixture(scope='module')
def motor_1_windows(
    motor_1_df: pd.DataFrame,
    window_size: int,
    clipping_threshold: int
) -> MotorWindows:
    """Builds windows for motor 1 — used by most tests."""
    return build_windows(motor_1_df, window_size, clipping_threshold)


def _make_synthetic_df(
    n_motors: int = 2,
    n_cycles: int = 50,
    n_sensors: int = 3,
    evento_motors: list[int] | None = None,
) -> pd.DataFrame:
    """Builds a minimal synthetic DataFrame with the expected column structure.

    Args:
        n_motors: Number of motors in the DataFrame.
        n_cycles: Number of cycles per motor.
        n_sensors: Number of sensor columns.
        evento_motors: List of motor IDs that have evento=1. All others
            are censored (evento=0).
    """
    if evento_motors is None:
        evento_motors = [1]

    rng = np.random.default_rng(42)
    rows = []
    for motor_id in range(1, n_motors + 1):
        for cycle in range(1, n_cycles + 1):
            is_last = (cycle == n_cycles)
            is_train = motor_id in evento_motors
            row: dict[str, int | float] = {
                'unit_number': motor_id,
                'time_in_cycles': cycle,
                'RUL': max(n_cycles - cycle, 0),
                'evento': int(is_train and is_last),
            }
            for s in range(n_sensors):
                row[f'sensor_{s+1}'] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestBuildWindows — Shape
# ---------------------------------------------------------------------------

class TestShape:
    """Tests for the shape of X_windows and output structure."""

    def test_x_windows_is_3d(
        self, motor_1_windows: MotorWindows, window_size: int
    ):
        """X_windows must be a 3D array (n_windows, window_size, n_features)."""
        data = motor_1_windows[1]
        assert data['X_windows'].ndim == 3

    def test_x_windows_window_size_axis(
        self, motor_1_windows: MotorWindows, window_size: int
    ):
        """Second axis of X_windows must equal window_size."""
        data = motor_1_windows[1]
        assert data['X_windows'].shape[1] == window_size

    def test_n_windows_correct(
        self, motor_1_df: pd.DataFrame, motor_1_windows: MotorWindows, window_size: int
    ):
        """Number of windows must equal n_cycles - window_size + 1."""
        n_cycles = len(motor_1_df)
        expected = n_cycles - window_size + 1
        assert motor_1_windows[1]['X_windows'].shape[0] == expected

    def test_n_features_matches_sensor_columns(
        self, motor_1_df: pd.DataFrame, motor_1_windows: MotorWindows
    ):
        """Third axis of X_windows must equal the number of feature columns."""
        meta_cols = {'unit_number', 'time_in_cycles', 'RUL', 'evento'}
        expected_n_features = len([c for c in motor_1_df.columns if c not in meta_cols])
        assert motor_1_windows[1]['X_windows'].shape[2] == expected_n_features

    def test_all_motors_present_in_output(self):
        """All motor IDs in input must appear as keys in output."""
        df = _make_synthetic_df(n_motors=3, n_cycles=50)
        result = build_windows(df, window_size=10, clipping_threshold=125)
        assert set(result.keys()) == {1, 2, 3}

    def test_output_arrays_same_length(self, motor_1_windows: MotorWindows):
        """All arrays in MotorData must have the same first dimension."""
        data = motor_1_windows[1]
        n = data['X_windows'].shape[0]
        assert len(data['t_start']) == n
        assert len(data['t_stop']) == n
        assert len(data['evento']) == n
        assert len(data['y_rul']) == n


# ---------------------------------------------------------------------------
# TestBuildWindows — Survival Targets
# ---------------------------------------------------------------------------

class TestSurvivalTargets:
    """Tests for the counting process interval construction."""

    def test_t_start_equals_first_cycle_minus_one(
        self, motor_1_df: pd.DataFrame, motor_1_windows: MotorWindows, window_size: int
    ):
        """t_start must equal the time_in_cycles of the first row of each window minus 1."""
        data = motor_1_windows[1]
        times = motor_1_df['time_in_cycles'].to_numpy()

        # First window: rows 0..window_size-1, t_start = times[0] - 1
        assert data['t_start'][0] == pytest.approx(times[0] - 1.0)
        # Second window: rows 1..window_size, t_start = times[1] - 1
        assert data['t_start'][1] == pytest.approx(times[1] - 1.0)

    def test_t_stop_equals_last_cycle_of_window(
        self, motor_1_df: pd.DataFrame, motor_1_windows: MotorWindows, window_size: int
    ):
        """t_stop must equal the time_in_cycles of the last row of each window."""
        data = motor_1_windows[1]
        times = motor_1_df['time_in_cycles'].to_numpy()

        # First window ends at row window_size - 1
        assert data['t_stop'][0] == pytest.approx(times[window_size - 1])
        # Last window ends at last row
        assert data['t_stop'][-1] == pytest.approx(times[-1])

    def test_t_start_strictly_less_than_t_stop(self, motor_1_windows: MotorWindows):
        """t_start must be strictly less than t_stop for all windows."""
        data = motor_1_windows[1]
        assert (data['t_start'] < data['t_stop']).all()

    def test_t_stop_non_decreasing(self, motor_1_windows: MotorWindows):
        """t_stop must be non-decreasing across windows."""
        data = motor_1_windows[1]
        assert (np.diff(data['t_stop']) >= 0).all()

    def test_evento_at_last_window_only_for_train_motor(
        self, motor_1_windows: MotorWindows
    ):
        """Train motor must have evento=1 only at the last window."""
        data = motor_1_windows[1]
        assert data['evento'].sum() == 1
        assert data['evento'][-1] == 1

    def test_censored_motor_evento_all_zero(self):
        """Censored motor (evento=0) must have evento=0 in all windows."""
        df = _make_synthetic_df(n_motors=1, n_cycles=50, evento_motors=[])
        result = build_windows(df, window_size=10, clipping_threshold=125)
        assert result[1]['evento'].sum() == 0

    def test_evento_zero_for_all_non_last_windows(self, motor_1_windows: MotorWindows):
        """All windows except the last must have evento=0 for a train motor."""
        data = motor_1_windows[1]
        assert (data['evento'][:-1] == 0).all()


# ---------------------------------------------------------------------------
# TestBuildWindows — RUL
# ---------------------------------------------------------------------------

class TestRUL:
    """Tests for the y_rul target construction."""

    def test_y_rul_clipped_to_threshold(self, motor_1_windows: MotorWindows, clipping_threshold: int):
        """All y_rul values must be <= clipping_threshold."""
        data = motor_1_windows[1]
        assert (data['y_rul'] <= clipping_threshold).all()

    def test_y_rul_non_negative(self, motor_1_windows: MotorWindows):
        """All y_rul values must be non-negative."""
        data = motor_1_windows[1]
        assert (data['y_rul'] >= 0).all()

    def test_y_rul_last_window_is_zero(self, motor_1_windows: MotorWindows):
        """RUL at the last window of a train motor must be 0."""
        data = motor_1_windows[1]
        assert data['y_rul'][-1] == pytest.approx(0.0)

    def test_y_rul_non_increasing(self, motor_1_windows: MotorWindows, clipping_threshold: int):
        """y_rul must be non-increasing (or flat in clipped region)."""
        data = motor_1_windows[1]
        # After unclipping we expect non-increasing — with clipping the
        # first windows are flat at clipping_threshold
        diffs = np.diff(data['y_rul'])
        assert (diffs <= 0).all()

    def test_y_rul_matches_rul_column_at_t_stop(
        self, motor_1_df: pd.DataFrame, motor_1_windows: MotorWindows,
        window_size: int, clipping_threshold: int
    ):
        """y_rul must equal min(RUL[t_stop_index], clipping_threshold)."""
        data = motor_1_windows[1]
        rul_vals = motor_1_df['RUL'].to_numpy()
        # Last window: t_stop corresponds to the last row
        expected = float(min(rul_vals[-1], clipping_threshold))
        assert data['y_rul'][-1] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestBuildWindows — Feature Names
# ---------------------------------------------------------------------------

class TestFeatureNames:
    """Tests for the feature_names field."""

    def test_feature_names_excludes_metadata(
        self, motor_1_df: pd.DataFrame, motor_1_windows: MotorWindows
    ):
        """feature_names must not contain metadata columns."""
        feature_names = motor_1_windows[1]['feature_names']
        for col in ['unit_number', 'time_in_cycles', 'RUL', 'evento']:
            assert col not in feature_names

    def test_feature_names_matches_x_windows_third_axis(
        self, motor_1_windows: MotorWindows
    ):
        """Length of feature_names must match third axis of X_windows."""
        data = motor_1_windows[1]
        assert len(data['feature_names']) == data['X_windows'].shape[2]

    def test_feature_names_consistent_across_motors(self):
        """All motors must have identical feature_names."""
        df = _make_synthetic_df(n_motors=3, n_cycles=50)
        result = build_windows(df, window_size=10, clipping_threshold=125)
        names = [result[m]['feature_names'] for m in result]
        assert all(n == names[0] for n in names)


# ---------------------------------------------------------------------------
# TestBuildWindows — Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for error paths and boundary conditions."""

    def test_raises_on_window_size_less_than_one(self):
        """window_size < 1 must raise ValueError."""
        df = _make_synthetic_df(n_motors=1, n_cycles=50)
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            build_windows(df, window_size=0, clipping_threshold=125)

    def test_raises_on_motor_shorter_than_window(self):
        """Motor with fewer cycles than window_size must raise ValueError."""
        df = _make_synthetic_df(n_motors=1, n_cycles=5)
        with pytest.raises(ValueError, match="window_size"):
            build_windows(df, window_size=10, clipping_threshold=125)

    def test_raises_on_negative_first_cycle(self):
        """Motor whose first cycle is < 1 must raise ValueError."""
        df = _make_synthetic_df(n_motors=1, n_cycles=50)
        df['time_in_cycles'] = df['time_in_cycles'] - 5  # starts at -4
        with pytest.raises(ValueError, match="time_in_cycles starting at"):
            build_windows(df, window_size=10, clipping_threshold=125)

    def test_window_size_equals_n_cycles(self):
        """window_size == n_cycles must produce exactly one window."""
        df = _make_synthetic_df(n_motors=1, n_cycles=20)
        result = build_windows(df, window_size=20, clipping_threshold=125)
        assert result[1]['X_windows'].shape[0] == 1

    def test_missing_unit_number_treated_as_single_motor(self):
        """DataFrame without unit_number must produce output keyed at 0."""
        df = _make_synthetic_df(n_motors=1, n_cycles=50).drop(columns=['unit_number'])
        result = build_windows(df, window_size=10, clipping_threshold=125)
        assert 0 in result

    def test_missing_rul_produces_nan(self):
        """DataFrame without RUL column must produce NaN in y_rul."""
        df = _make_synthetic_df(n_motors=1, n_cycles=50).drop(columns=['RUL'])
        result = build_windows(df, window_size=10, clipping_threshold=125)
        assert np.all(np.isnan(result[1]['y_rul']))

    def test_missing_evento_produces_all_zero(self):
        """DataFrame without evento column must produce 0 in evento."""
        df = _make_synthetic_df(n_motors=1, n_cycles=50).drop(columns=['evento'])
        result = build_windows(df, window_size=10, clipping_threshold=125)
        assert (result[1]['evento'] == 0).all()


# ---------------------------------------------------------------------------
# TestFlattenWindows
# ---------------------------------------------------------------------------

class TestFlattenWindows:
    """Tests for the flatten_windows utility function."""

    def test_flatten_x_windows_shape(self):
        """Flattened X_windows must have shape (total_windows, window_size, n_features)."""
        df = _make_synthetic_df(n_motors=3, n_cycles=50)
        result = build_windows(df, window_size=10, clipping_threshold=125)
        X, *_ = flatten_windows(result)
        total = sum(d['X_windows'].shape[0] for d in result.values())
        assert X.shape[0] == total
        assert X.shape[1] == 10

    def test_flatten_groups_match_motor_ids(self):
        """Groups array must map each window to its motor_id."""
        df = _make_synthetic_df(n_motors=2, n_cycles=50)
        result = build_windows(df, window_size=10, clipping_threshold=125)
        _, _, _, _, _, groups = flatten_windows(result)
        for motor_id, data in result.items():
            n = data['X_windows'].shape[0]
            motor_groups = groups[groups == motor_id]
            assert len(motor_groups) == n

    def test_flatten_returns_six_arrays(self):
        """flatten_windows must return exactly 6 arrays."""
        df = _make_synthetic_df(n_motors=2, n_cycles=50)
        result = build_windows(df, window_size=10, clipping_threshold=125)
        output = flatten_windows(result)
        assert len(output) == 6

    def test_flatten_t_stop_non_decreasing_per_motor(self):
        """t_stop must be non-decreasing within each motor's windows."""
        df = _make_synthetic_df(n_motors=2, n_cycles=50)
        result = build_windows(df, window_size=10, clipping_threshold=125)
        _, _, t_stop, _, _, groups = flatten_windows(result)
        for motor_id in np.unique(groups):
            mask = groups == motor_id
            assert (np.diff(t_stop[mask]) >= 0).all()

    def test_flatten_all_arrays_same_length(self):
        """All arrays returned by flatten_windows must have the same length."""
        df = _make_synthetic_df(n_motors=2, n_cycles=50)
        result = build_windows(df, window_size=10, clipping_threshold=125)
        X, t_start, t_stop, evento, y_rul, groups = flatten_windows(result)
        n = X.shape[0]
        assert len(t_start) == n
        assert len(t_stop) == n
        assert len(evento) == n
        assert len(y_rul) == n
        assert len(groups) == n