"""Tests for the feature_extraction module (Nodo 3 of the RUL pipeline).

Uses motor 1 from the C-MAPSS FD001 clean dataset as the primary fixture,
processed through build_windows first (Nodo 2 output is Nodo 3 input).
A small window_size is used to keep extraction fast during testing.

The timing fixture logs elapsed time to give visibility into extraction
speed — one of the key concerns identified during pipeline design.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.pipeline.windowing import MotorData, MotorWindows, build_windows
from src.pipeline.feature_extraction import (
    ALL_FEATURES,
    STATISTICAL_FEATURES,
    TREND_FEATURES,
    _build_feature_names,
    _compute_features,
    extract_window_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def window_size() -> int:
    """Small window size to keep extraction fast during testing."""
    return 20


@pytest.fixture(scope='module')
def clipping_threshold() -> int:
    return 125


@pytest.fixture(scope='module')
def motor_1_windows(window_size: int, clipping_threshold: int) -> MotorWindows:
    """Builds Nodo 2 output for motor 1 — input to Nodo 3."""
    df = pd.read_csv('data/clean/data_motor_1.csv')
    df.insert(0, 'unit_number', 1)
    return build_windows(df, window_size=window_size, clipping_threshold=clipping_threshold)


@pytest.fixture(scope='module')
def motor_1_features(motor_1_windows: MotorWindows) -> MotorWindows:
    """Runs Nodo 3 on motor 1 and logs elapsed time."""
    start = time.perf_counter()
    result = extract_window_features(motor_1_windows)
    elapsed = time.perf_counter() - start
    print(f"\n[Nodo 3 timing] Motor 1 extraction: {elapsed:.2f}s")
    return result


# ---------------------------------------------------------------------------
# TestComputeFeatures — unit tests for _compute_features
# ---------------------------------------------------------------------------

class TestComputeFeatures:
    """Unit tests for the internal _compute_features function."""

    @pytest.fixture
    def simple_X(self) -> np.ndarray:
        """Simple 3D array: 2 windows, 10 cycles, 2 sensors."""
        rng = np.random.default_rng(42)
        return rng.normal(size=(2, 10, 2))

    def test_output_shape_statistical(self, simple_X: np.ndarray):
        """Statistical features must produce (n_windows, n_features * n_sensors)."""
        result = _compute_features(simple_X, STATISTICAL_FEATURES)
        n_windows, _, n_sensors = simple_X.shape
        assert result.shape == (n_windows, len(STATISTICAL_FEATURES) * n_sensors)

    def test_output_shape_trend(self, simple_X: np.ndarray):
        """Trend features must produce (n_windows, n_features * n_sensors)."""
        result = _compute_features(simple_X, TREND_FEATURES)
        n_windows, _, n_sensors = simple_X.shape
        assert result.shape == (n_windows, len(TREND_FEATURES) * n_sensors)

    def test_output_shape_all(self, simple_X: np.ndarray):
        """All features must produce (n_windows, n_all_features * n_sensors)."""
        result = _compute_features(simple_X, ALL_FEATURES)
        n_windows, _, n_sensors = simple_X.shape
        assert result.shape == (n_windows, len(ALL_FEATURES) * n_sensors)

    def test_no_nan_in_output(self, simple_X: np.ndarray):
        """Output must contain no NaN values."""
        result = _compute_features(simple_X, ALL_FEATURES)
        assert not np.isnan(result).any()

    def test_no_inf_in_output(self, simple_X: np.ndarray):
        """Output must contain no infinite values."""
        result = _compute_features(simple_X, ALL_FEATURES)
        assert np.isfinite(result).all()

    def test_median_correct(self):
        """Median feature must match np.median over window axis."""
        X = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)
        result = _compute_features(X, ['median'])
        expected = np.median(X, axis=1)  # (1, 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_abs_energy_correct(self):
        """abs_energy must equal sum of squares over window axis."""
        X = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        result = _compute_features(X, ['abs_energy'])
        expected = np.sum(X ** 2, axis=1)  # (1, 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_q25_less_than_q75(self, simple_X: np.ndarray):
        """Q25 must be <= Q75 for all windows and sensors."""
        q25 = _compute_features(simple_X, ['q25'])
        q75 = _compute_features(simple_X, ['q75'])
        assert (q25 <= q75).all()

    def test_rvalue_in_valid_range(self, simple_X: np.ndarray):
        """Pearson r must be in [-1, 1]."""
        result = _compute_features(simple_X, ['rvalue'])
        assert (result >= -1.0 - 1e-10).all()
        assert (result <= 1.0 + 1e-10).all()

    def test_constant_window_rvalue_is_zero(self):
        """Constant sensor series must produce rvalue=0 (not NaN or inf)."""
        X = np.ones((2, 10, 2))  # constant windows
        result = _compute_features(X, ['rvalue'])
        assert not np.isnan(result).any()
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_constant_window_autocorr_is_zero(self):
        """Constant sensor series must produce autocorr=0 (not NaN or inf)."""
        X = np.ones((2, 10, 2))
        result = _compute_features(X, ['autocorr_lag_1'])
        assert not np.isnan(result).any()
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_constant_window_partial_autocorr_is_zero(self):
        """Constant sensor series must produce partial_autocorr=0."""
        X = np.ones((2, 10, 2))
        result = _compute_features(X, ['partial_autocorr_lag_1'])
        assert not np.isnan(result).any()
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_raises_on_unknown_feature(self, simple_X: np.ndarray):
        """Unknown feature name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown features"):
            _compute_features(simple_X, ['nonexistent_feature'])

    def test_increasing_series_positive_slope(self):
        """Strictly increasing series must have positive slope."""
        X = np.arange(10, dtype=float).reshape(1, 10, 1)
        result = _compute_features(X, ['slope'])
        assert result[0, 0] > 0

    def test_decreasing_series_negative_slope(self):
        """Strictly decreasing series must have negative slope."""
        X = np.arange(9, -1, -1, dtype=float).reshape(1, 10, 1)
        result = _compute_features(X, ['slope'])
        assert result[0, 0] < 0


# ---------------------------------------------------------------------------
# TestBuildFeatureNames
# ---------------------------------------------------------------------------

class TestBuildFeatureNames:
    """Tests for the _build_feature_names utility."""

    def test_length_correct(self):
        """Length must equal len(features) * len(sensor_names)."""
        names = _build_feature_names(['median', 'slope'], ['T24', 'T30'])
        assert len(names) == 4

    def test_format_uses_double_underscore(self):
        """Names must use double-underscore separator."""
        names = _build_feature_names(['median'], ['T24'])
        assert names[0] == 'T24__median'

    def test_order_feature_then_sensor(self):
        """Column order must be: feature_1_sensor_1, feature_1_sensor_2, ..."""
        names = _build_feature_names(['median', 'slope'], ['T24', 'T30'])
        assert names == ['T24__median', 'T30__median', 'T24__slope', 'T30__slope']


# ---------------------------------------------------------------------------
# TestExtractWindowFeatures — integration tests with real data
# ---------------------------------------------------------------------------

class TestShape:
    """Tests for the shape of X_windows after feature extraction."""

    def test_output_x_windows_is_2d(self, motor_1_features: MotorWindows):
        """X_windows must be 2D after feature extraction."""
        assert motor_1_features[1]['X_windows'].ndim == 2

    def test_n_windows_preserved(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        """Number of windows must be identical before and after extraction."""
        n_before = motor_1_windows[1]['X_windows'].shape[0]
        n_after = motor_1_features[1]['X_windows'].shape[0]
        assert n_before == n_after

    def test_n_features_matches_feature_names(self, motor_1_features: MotorWindows):
        """Number of columns must match length of feature_names."""
        data = motor_1_features[1]
        assert data['X_windows'].shape[1] == len(data['feature_names'])


class TestPassthrough:
    """Tests that survival targets and RUL pass through unchanged."""

    def test_t_start_unchanged(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        np.testing.assert_array_equal(
            motor_1_windows[1]['t_start'], motor_1_features[1]['t_start']
        )

    def test_t_stop_unchanged(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        np.testing.assert_array_equal(
            motor_1_windows[1]['t_stop'], motor_1_features[1]['t_stop']
        )

    def test_evento_unchanged(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        np.testing.assert_array_equal(
            motor_1_windows[1]['evento'], motor_1_features[1]['evento']
        )

    def test_y_rul_unchanged(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        np.testing.assert_array_almost_equal(
            motor_1_windows[1]['y_rul'], motor_1_features[1]['y_rul']
        )


class TestNaN:
    """Tests for NaN and inf handling."""

    def test_no_nan_in_x_windows(self, motor_1_features: MotorWindows):
        assert not np.isnan(motor_1_features[1]['X_windows']).any()

    def test_no_inf_in_x_windows(self, motor_1_features: MotorWindows):
        assert np.isfinite(motor_1_features[1]['X_windows']).all()


class TestEdgeCases:
    """Tests for error paths and configuration variants."""

    def test_raises_on_non_3d_input(self, motor_1_windows: MotorWindows):
        """Must raise ValueError if X_windows is not 3D."""
        bad_windows: MotorWindows = {
            1: MotorData(
                X_windows=np.zeros((10, 5)),
                t_start=motor_1_windows[1]['t_start'][:10],
                t_stop=motor_1_windows[1]['t_stop'][:10],
                evento=motor_1_windows[1]['evento'][:10],
                y_rul=motor_1_windows[1]['y_rul'][:10],
                feature_names=motor_1_windows[1]['feature_names'],
            )
        }
        with pytest.raises(ValueError, match="3D"):
            extract_window_features(bad_windows)

    def test_raises_on_unordered_windows(self, motor_1_windows: MotorWindows):
        """Must raise ValueError if t_stop is not non-decreasing."""
        data = motor_1_windows[1]
        t_stop_bad = data['t_stop'].copy()
        t_stop_bad[5] = t_stop_bad[0]
        bad_windows: MotorWindows = {
            1: MotorData(
                X_windows=data['X_windows'],
                t_start=data['t_start'],
                t_stop=t_stop_bad,
                evento=data['evento'],
                y_rul=data['y_rul'],
                feature_names=data['feature_names'],
            )
        }
        with pytest.raises(ValueError, match="chronological order"):
            extract_window_features(bad_windows)

    def test_statistical_features_only(self, motor_1_windows: MotorWindows):
        """Extraction with only statistical features must succeed."""
        result = extract_window_features(motor_1_windows, features=STATISTICAL_FEATURES)
        assert result[1]['X_windows'].ndim == 2

    def test_trend_features_only(self, motor_1_windows: MotorWindows):
        """Extraction with only trend features must succeed."""
        result = extract_window_features(motor_1_windows, features=TREND_FEATURES)
        assert result[1]['X_windows'].ndim == 2

    def test_all_features_more_columns_than_statistical(
        self, motor_1_windows: MotorWindows
    ):
        """ALL_FEATURES must produce more columns than STATISTICAL_FEATURES alone."""
        result_stat = extract_window_features(
            motor_1_windows, features=STATISTICAL_FEATURES
        )
        result_all = extract_window_features(
            motor_1_windows, features=ALL_FEATURES
        )
        assert result_all[1]['X_windows'].shape[1] > result_stat[1]['X_windows'].shape[1]