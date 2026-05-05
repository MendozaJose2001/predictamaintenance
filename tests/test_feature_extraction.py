"""Tests for the feature_extraction module (Nodo 3 of the RUL pipeline).

Uses motor 1 from the C-MAPSS FD001 clean dataset as the primary fixture,
processed through build_windows first (Nodo 2 output is Nodo 3 input).
A small window_size is used to keep tsfresh extraction fast during testing.

The timing fixture logs elapsed time to give visibility into extraction
speed — one of the key concerns identified during pipeline design.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.pipeline.windowing import build_windows, MotorWindows
from src.pipeline.feature_extraction import (
    STATISTICAL_FEATURES,
    TSFRESH_FEATURES,
    TREND_FEATURES,
    extract_window_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def window_size() -> int:
    """Small window size to keep tsfresh fast during testing."""
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
    """Runs Nodo 3 on motor 1 and logs elapsed time.

    Module scope ensures tsfresh runs only once for all tests that
    depend on this fixture — extraction is expensive.
    """
    start = time.perf_counter()
    result = extract_window_features(motor_1_windows)
    elapsed = time.perf_counter() - start
    print(f"\n[Nodo 3 timing] Motor 1 extraction: {elapsed:.2f}s")
    return result


# ---------------------------------------------------------------------------
# TestShape
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

    def test_n_features_positive(self, motor_1_features: MotorWindows):
        """Output must have at least one feature column."""
        assert motor_1_features[1]['X_windows'].shape[1] > 0

    def test_n_features_matches_feature_names(self, motor_1_features: MotorWindows):
        """Number of columns must match length of feature_names."""
        data = motor_1_features[1]
        assert data['X_windows'].shape[1] == len(data['feature_names'])


# ---------------------------------------------------------------------------
# TestPassthrough
# ---------------------------------------------------------------------------

class TestPassthrough:
    """Tests that survival targets and RUL pass through unchanged."""

    def test_t_start_unchanged(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        """t_start must be identical before and after extraction."""
        np.testing.assert_array_equal(
            motor_1_windows[1]['t_start'],
            motor_1_features[1]['t_start']
        )

    def test_t_stop_unchanged(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        """t_stop must be identical before and after extraction."""
        np.testing.assert_array_equal(
            motor_1_windows[1]['t_stop'],
            motor_1_features[1]['t_stop']
        )

    def test_evento_unchanged(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        """evento must be identical before and after extraction."""
        np.testing.assert_array_equal(
            motor_1_windows[1]['evento'],
            motor_1_features[1]['evento']
        )

    def test_y_rul_unchanged(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        """y_rul must be identical before and after extraction."""
        np.testing.assert_array_almost_equal(
            motor_1_windows[1]['y_rul'],
            motor_1_features[1]['y_rul']
        )

    def test_motor_ids_preserved(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        """All motor IDs from input must be present in output."""
        assert set(motor_1_windows.keys()) == set(motor_1_features.keys())


# ---------------------------------------------------------------------------
# TestFeatureNames
# ---------------------------------------------------------------------------

class TestFeatureNames:
    """Tests for the feature_names field after extraction."""

    def test_feature_names_updated(
        self, motor_1_windows: MotorWindows, motor_1_features: MotorWindows
    ):
        """feature_names must change after extraction — tsfresh names expected."""
        names_before = motor_1_windows[1]['feature_names']
        names_after = motor_1_features[1]['feature_names']
        assert names_before != names_after

    def test_feature_names_contain_sensor_references(
        self, motor_1_features: MotorWindows
    ):
        """tsfresh feature names must reference the original sensor names."""
        feature_names = motor_1_features[1]['feature_names']
        # tsfresh names format: sensor_name__calculator_name
        assert any('__' in name for name in feature_names)

    def test_feature_names_are_strings(self, motor_1_features: MotorWindows):
        """All feature names must be strings."""
        for name in motor_1_features[1]['feature_names']:
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# TestNaN
# ---------------------------------------------------------------------------

class TestNaN:
    """Tests for NaN handling in the output."""

    def test_no_nan_in_x_windows(self, motor_1_features: MotorWindows):
        """X_windows must contain no NaN values after extraction."""
        assert not np.isnan(motor_1_features[1]['X_windows']).any()

    def test_no_inf_in_x_windows(self, motor_1_features: MotorWindows):
        """X_windows must contain no infinite values after extraction."""
        assert np.isfinite(motor_1_features[1]['X_windows']).all()


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for error paths and configuration variants."""

    def test_raises_on_non_3d_input(self, motor_1_windows: MotorWindows):
        """Must raise ValueError if X_windows is not 3D."""
        from src.pipeline.windowing import MotorData
        import copy

        bad_windows: MotorWindows = {
            1: MotorData(
                X_windows=np.zeros((10, 5)),  # 2D — should be 3D
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
        from src.pipeline.windowing import MotorData

        data = motor_1_windows[1]
        t_stop_bad = data['t_stop'].copy()
        t_stop_bad[5] = t_stop_bad[0]  # introduce disorder

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
        result = extract_window_features(
            motor_1_windows, feature_config=STATISTICAL_FEATURES
        )
        assert result[1]['X_windows'].ndim == 2
        assert result[1]['X_windows'].shape[0] == motor_1_windows[1]['X_windows'].shape[0]

    def test_trend_features_only(self, motor_1_windows: MotorWindows):
        """Extraction with only trend features must succeed."""
        result = extract_window_features(
            motor_1_windows, feature_config=TREND_FEATURES
        )
        assert result[1]['X_windows'].ndim == 2

    def test_full_config_has_more_features_than_partial(
        self, motor_1_windows: MotorWindows
    ):
        """Full TSFRESH_FEATURES must produce more columns than either subset."""
        result_stat = extract_window_features(
            motor_1_windows, feature_config=STATISTICAL_FEATURES
        )
        result_full = extract_window_features(
            motor_1_windows, feature_config=TSFRESH_FEATURES
        )
        n_stat = result_stat[1]['X_windows'].shape[1]
        n_full = result_full[1]['X_windows'].shape[1]
        assert n_full > n_stat