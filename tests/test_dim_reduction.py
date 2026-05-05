"""Tests for the dim_reduction module (Nodo 4 of the RUL pipeline).

Uses motor 1 from C-MAPSS processed through Nodos 1-3 as the primary
fixture. A small window_size and n_components are used to keep tests fast.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.pipeline.scaling import FeatureScaler
from src.pipeline.windowing import build_windows, MotorWindows, MotorData
from src.pipeline.feature_extraction import extract_window_features
from src.pipeline.dim_reduction import DimReducer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def motor_1_features() -> MotorWindows:
    """Builds Nodo 3 output for motor 1 — input to Nodo 4.

    Runs the full Nodo 1 → 2 → 3 chain on motor 1.
    """
    df = pd.read_csv('data/clean/data_motor_1.csv')
    df.insert(0, 'unit_number', 1)

    # Nodo 1 — scaling (fit and transform on same motor for test purposes)
    X_df = df.drop(columns=['RUL', 'evento'])
    scaler = FeatureScaler()
    scaler.fit(X_df)
    X_scaled: pd.DataFrame = scaler.transform(X_df)

    # Nodo 2 — windowing (no RUL/evento in X_scaled)
    motor_windows = build_windows(X_scaled, window_size=20, clipping_threshold=125)

    # Nodo 3 — feature extraction
    return extract_window_features(motor_windows)


@pytest.fixture(scope='module')
def n_components() -> int:
    """Default number of components for tests."""
    return 5


@pytest.fixture(scope='module')
def fitted_reducer(
    motor_1_features: MotorWindows,
    n_components: int,
) -> DimReducer:
    """Returns a fitted DimReducer on motor 1 features."""
    reducer = DimReducer(n_components=n_components)
    reducer.fit(motor_1_features)
    return reducer


@pytest.fixture(scope='module')
def reduced_windows(
    fitted_reducer: DimReducer,
    motor_1_features: MotorWindows,
) -> MotorWindows:
    """Returns the transformed MotorWindows after PCA."""
    return fitted_reducer.transform(motor_1_features)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for DimReducer initialization."""

    def test_default_n_components(self):
        """Default n_components must be 10."""
        reducer = DimReducer()
        assert reducer.n_components == 10

    def test_custom_n_components(self):
        """Custom n_components must be stored as provided."""
        reducer = DimReducer(n_components=15)
        assert reducer.n_components == 15

    def test_pca_not_set_at_init(self):
        """pca_ must not exist before fit() is called."""
        reducer = DimReducer()
        assert not hasattr(reducer, 'pca_')

    def test_pc_names_not_set_at_init(self):
        """pc_names_ must not exist before fit() is called."""
        reducer = DimReducer()
        assert not hasattr(reducer, 'pc_names_')


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:
    """Tests for the fit() method."""

    def test_fit_returns_self(self, motor_1_features: MotorWindows):
        """fit() must return self."""
        reducer = DimReducer(n_components=5)
        result = reducer.fit(motor_1_features)
        assert result is reducer

    def test_fit_sets_pca_(self, motor_1_features: MotorWindows):
        """fit() must set the pca_ attribute."""
        reducer = DimReducer(n_components=5)
        reducer.fit(motor_1_features)
        assert hasattr(reducer, 'pca_')

    def test_fit_sets_pc_names_(self, motor_1_features: MotorWindows):
        """fit() must set the pc_names_ attribute."""
        reducer = DimReducer(n_components=5)
        reducer.fit(motor_1_features)
        assert hasattr(reducer, 'pc_names_')

    def test_pc_names_length_matches_n_components(
        self, motor_1_features: MotorWindows
    ):
        """pc_names_ length must equal n_components."""
        n = 5
        reducer = DimReducer(n_components=n)
        reducer.fit(motor_1_features)
        assert len(reducer.pc_names_) == n

    def test_pc_names_format(self, motor_1_features: MotorWindows):
        """pc_names_ must follow the format PC_1, PC_2, ..."""
        reducer = DimReducer(n_components=3)
        reducer.fit(motor_1_features)
        assert reducer.pc_names_ == ['PC_1', 'PC_2', 'PC_3']

    def test_raises_on_non_2d_input(self, motor_1_features: MotorWindows):
        """Must raise ValueError if X_windows is not 2D."""
        bad_windows: MotorWindows = {
            1: MotorData(
                X_windows=np.zeros((10, 5, 3)),  # 3D — should be 2D
                t_start=motor_1_features[1]['t_start'][:10],
                t_stop=motor_1_features[1]['t_stop'][:10],
                evento=motor_1_features[1]['evento'][:10],
                y_rul=motor_1_features[1]['y_rul'][:10],
                feature_names=motor_1_features[1]['feature_names'],
            )
        }
        with pytest.raises(ValueError, match="2D"):
            DimReducer(n_components=3).fit(bad_windows)

    def test_raises_if_n_components_exceeds_features(
        self, motor_1_features: MotorWindows
    ):
        """Must raise ValueError if n_components > n_features."""
        n_features = motor_1_features[1]['X_windows'].shape[1]
        reducer = DimReducer(n_components=n_features + 1)
        with pytest.raises(ValueError, match="n_components"):
            reducer.fit(motor_1_features)


# ---------------------------------------------------------------------------
# TestTransform
# ---------------------------------------------------------------------------

class TestTransform:
    """Tests for the transform() method."""

    def test_raises_if_not_fitted(self, motor_1_features: MotorWindows):
        """Must raise NotFittedError if transform() called before fit()."""
        reducer = DimReducer(n_components=5)
        with pytest.raises(NotFittedError):
            reducer.transform(motor_1_features)

    def test_output_x_windows_shape(
        self,
        reduced_windows: MotorWindows,
        motor_1_features: MotorWindows,
        n_components: int,
    ):
        """X_windows must have shape (n_windows, n_components)."""
        n_windows = motor_1_features[1]['X_windows'].shape[0]
        assert reduced_windows[1]['X_windows'].shape == (n_windows, n_components)

    def test_n_windows_preserved(
        self,
        motor_1_features: MotorWindows,
        reduced_windows: MotorWindows,
    ):
        """Number of windows must be identical before and after PCA."""
        n_before = motor_1_features[1]['X_windows'].shape[0]
        n_after = reduced_windows[1]['X_windows'].shape[0]
        assert n_before == n_after

    def test_feature_names_updated_to_pc_names(
        self,
        reduced_windows: MotorWindows,
        n_components: int,
    ):
        """feature_names must be updated to PC_1, PC_2, ..."""
        feature_names = reduced_windows[1]['feature_names']
        assert feature_names == [f'PC_{i+1}' for i in range(n_components)]

    def test_t_start_unchanged(
        self,
        motor_1_features: MotorWindows,
        reduced_windows: MotorWindows,
    ):
        """t_start must be identical before and after PCA."""
        np.testing.assert_array_equal(
            motor_1_features[1]['t_start'],
            reduced_windows[1]['t_start']
        )

    def test_t_stop_unchanged(
        self,
        motor_1_features: MotorWindows,
        reduced_windows: MotorWindows,
    ):
        """t_stop must be identical before and after PCA."""
        np.testing.assert_array_equal(
            motor_1_features[1]['t_stop'],
            reduced_windows[1]['t_stop']
        )

    def test_evento_unchanged(
        self,
        motor_1_features: MotorWindows,
        reduced_windows: MotorWindows,
    ):
        """evento must be identical before and after PCA."""
        np.testing.assert_array_equal(
            motor_1_features[1]['evento'],
            reduced_windows[1]['evento']
        )

    def test_y_rul_unchanged(
        self,
        motor_1_features: MotorWindows,
        reduced_windows: MotorWindows,
    ):
        """y_rul must be identical before and after PCA."""
        np.testing.assert_array_almost_equal(
            motor_1_features[1]['y_rul'],
            reduced_windows[1]['y_rul']
        )

    def test_no_nan_in_output(self, reduced_windows: MotorWindows):
        """Output X_windows must contain no NaN values."""
        assert not np.isnan(reduced_windows[1]['X_windows']).any()

    def test_no_inf_in_output(self, reduced_windows: MotorWindows):
        """Output X_windows must contain no infinite values."""
        assert np.isfinite(reduced_windows[1]['X_windows']).all()

    def test_raises_on_non_2d_input(
        self,
        fitted_reducer: DimReducer,
        motor_1_features: MotorWindows,
    ):
        """Must raise ValueError if X_windows is not 2D."""
        bad_windows: MotorWindows = {
            1: MotorData(
                X_windows=np.zeros((10, 5, 3)),
                t_start=motor_1_features[1]['t_start'][:10],
                t_stop=motor_1_features[1]['t_stop'][:10],
                evento=motor_1_features[1]['evento'][:10],
                y_rul=motor_1_features[1]['y_rul'][:10],
                feature_names=motor_1_features[1]['feature_names'],
            )
        }
        with pytest.raises(ValueError, match="2D"):
            fitted_reducer.transform(bad_windows)

    def test_motor_ids_preserved(
        self,
        motor_1_features: MotorWindows,
        reduced_windows: MotorWindows,
    ):
        """All motor IDs from input must be present in output."""
        assert set(motor_1_features.keys()) == set(reduced_windows.keys())


# ---------------------------------------------------------------------------
# TestExplainedVariance
# ---------------------------------------------------------------------------

class TestExplainedVariance:
    """Tests for the explained_variance_ratio() method."""

    def test_raises_if_not_fitted(self):
        """Must raise NotFittedError if called before fit()."""
        reducer = DimReducer(n_components=5)
        with pytest.raises(NotFittedError):
            reducer.explained_variance_ratio()

    def test_returns_array_of_correct_length(
        self,
        fitted_reducer: DimReducer,
        n_components: int,
    ):
        """Must return array of length n_components."""
        evr = fitted_reducer.explained_variance_ratio()
        assert len(evr) == n_components

    def test_values_in_unit_interval(self, fitted_reducer: DimReducer):
        """All values must be in [0, 1]."""
        evr = fitted_reducer.explained_variance_ratio()
        assert (evr >= 0.0).all()
        assert (evr <= 1.0).all()

    def test_values_sum_to_at_most_one(self, fitted_reducer: DimReducer):
        """Cumulative explained variance must not exceed 1.0."""
        evr = fitted_reducer.explained_variance_ratio()
        assert evr.sum() <= 1.0 + 1e-10

    def test_values_non_increasing(self, fitted_reducer: DimReducer):
        """Explained variance must be non-increasing (PCA sorts by variance)."""
        evr = fitted_reducer.explained_variance_ratio()
        assert (np.diff(evr) <= 1e-10).all()