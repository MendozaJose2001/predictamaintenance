"""Tests for the scaling module (Nodo 1 of the RUL pipeline).

Uses synthetic DataFrames that mimic the C-MAPSS clean dataset structure.
No real data is needed — the scaler logic is independent of the dataset.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.pipeline.scaling import FeatureScaler, _PROTECTED_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n_rows: int = 20,
    n_sensors: int = 3,
    include_unit: bool = True,
    include_time: bool = True,
    include_evento: bool = True,
    constant_sensors: bool = False,
) -> pd.DataFrame:
    """Builds a minimal DataFrame mimicking the pipeline input structure.

    Args:
        n_rows: Number of rows.
        n_sensors: Number of sensor columns.
        include_unit: Whether to include unit_number column.
        include_time: Whether to include time_in_cycles column.
        include_evento: Whether to include evento column.
        constant_sensors: If True all sensor values are 5.0 (edge case).

    Returns:
        DataFrame with metadata and sensor columns.
    """
    rng = np.random.default_rng(42)
    data: dict[str, object] = {}

    if include_unit:
        data['unit_number'] = np.repeat(np.arange(1, 3), n_rows // 2)
    if include_time:
        data['time_in_cycles'] = np.tile(np.arange(1, n_rows // 2 + 1), 2)
    if include_evento:
        # evento=1 only at last row of each motor — mimics real dataset
        evento = np.zeros(n_rows, dtype=int)
        evento[n_rows // 2 - 1] = 1
        evento[n_rows - 1] = 1
        data['evento'] = evento

    for i in range(n_sensors):
        data[f'sensor_{i+1}'] = (
            np.full(n_rows, 5.0) if constant_sensors
            else rng.normal(loc=i * 10, scale=2.0, size=n_rows)
        )

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for FeatureScaler initialization."""

    def test_default_quantile_range(self):
        """Default quantile range must be (25.0, 75.0)."""
        scaler = FeatureScaler()
        assert scaler.quantile_range == (25.0, 75.0)

    def test_custom_quantile_range(self):
        """Custom quantile range must be stored as provided."""
        scaler = FeatureScaler(quantile_range=(10.0, 90.0))
        assert scaler.quantile_range == (10.0, 90.0)

    def test_not_fitted_at_init(self):
        """scaler_ must not exist before fit() is called."""
        scaler = FeatureScaler()
        assert not hasattr(scaler, 'scaler_')

    def test_feature_cols_not_set_at_init(self):
        """feature_cols_ must not exist before fit() is called."""
        scaler = FeatureScaler()
        assert not hasattr(scaler, 'feature_cols_')


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:
    """Tests for the fit() method."""

    def test_fit_returns_self(self):
        """fit() must return self for sklearn pipeline compatibility."""
        df = _make_df()
        scaler = FeatureScaler()
        result = scaler.fit(df)
        assert result is scaler

    def test_fit_sets_scaler_(self):
        """fit() must set the scaler_ attribute."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        assert hasattr(scaler, 'scaler_')

    def test_fit_sets_feature_cols_(self):
        """fit() must set the feature_cols_ attribute."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        assert hasattr(scaler, 'feature_cols_')

    def test_feature_cols_excludes_protected(self):
        """feature_cols_ must not contain unit_number or time_in_cycles."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        for col in _PROTECTED_COLS:
            assert col not in scaler.feature_cols_

    def test_feature_cols_includes_sensors(self):
        """feature_cols_ must include all sensor columns."""
        df = _make_df(n_sensors=3)
        scaler = FeatureScaler()
        scaler.fit(df)
        for col in ['sensor_1', 'sensor_2', 'sensor_3']:
            assert col in scaler.feature_cols_

    def test_raises_when_no_feature_cols(self):
        """Must raise ValueError if all columns are protected metadata."""
        df = pd.DataFrame({
            'unit_number': [1, 2, 3],
            'time_in_cycles': [1, 2, 3],
            'evento': [0, 0, 1],
        })
        scaler = FeatureScaler()
        with pytest.raises(ValueError, match="No feature columns found"):
            scaler.fit(df)

    def test_fit_without_unit_number(self):
        """fit() must work even when unit_number is absent."""
        df = _make_df(include_unit=False)
        scaler = FeatureScaler()
        scaler.fit(df)
        assert 'unit_number' not in scaler.feature_cols_

    def test_fit_without_time_in_cycles(self):
        """fit() must work even when time_in_cycles is absent."""
        df = _make_df(include_time=False)
        scaler = FeatureScaler()
        scaler.fit(df)
        assert 'time_in_cycles' not in scaler.feature_cols_


# ---------------------------------------------------------------------------
# TestTransform
# ---------------------------------------------------------------------------

class TestTransform:
    """Tests for the transform() method."""

    def test_raises_if_not_fitted(self):
        """Must raise NotFittedError if transform() called before fit()."""
        df = _make_df()
        scaler = FeatureScaler()
        with pytest.raises(NotFittedError):
            scaler.transform(df)

    def test_output_is_dataframe(self):
        """transform() must return a DataFrame."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        result = scaler.transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_match_input(self):
        """Output must have the same columns as input in the same order."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        result = scaler.transform(df)
        assert list(result.columns) == list(df.columns)

    def test_output_shape_matches_input(self):
        """Output shape must match input shape."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        result = scaler.transform(df)
        assert result.shape == df.shape

    def test_unit_number_unchanged(self):
        """unit_number must be identical before and after transform."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        result = scaler.transform(df)
        pd.testing.assert_series_equal(result['unit_number'], df['unit_number'])

    def test_time_in_cycles_unchanged(self):
        """time_in_cycles must be identical before and after transform."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        result = scaler.transform(df)
        pd.testing.assert_series_equal(result['time_in_cycles'], df['time_in_cycles'])

    def test_evento_unchanged(self):
        """evento must be identical before and after transform."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        result = scaler.transform(df)
        pd.testing.assert_series_equal(result['evento'], df['evento'])

    def test_feature_cols_excludes_evento(self):
        """feature_cols_ must not contain evento."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        assert 'evento' not in scaler.feature_cols_

    def test_sensor_columns_are_scaled(self):
        """Sensor columns must differ from input after scaling."""
        df = _make_df(n_sensors=3)
        scaler = FeatureScaler()
        scaler.fit(df)
        result = scaler.transform(df)
        for col in ['sensor_1', 'sensor_2', 'sensor_3']:
            assert not np.allclose(result[col].to_numpy(), df[col].to_numpy())

    def test_scaled_median_near_zero(self):
        """After scaling, median of each feature column must be near zero."""
        df = _make_df(n_sensors=3)
        scaler = FeatureScaler()
        scaler.fit(df)
        result = scaler.transform(df)
        for col in scaler.feature_cols_:
            assert abs(result[col].median()) < 1e-6

    def test_raises_on_missing_column(self):
        """Must raise ValueError if a column seen at fit is missing at transform."""
        df_train = _make_df(n_sensors=3)
        df_val = _make_df(n_sensors=2)  # missing sensor_3
        scaler = FeatureScaler()
        scaler.fit(df_train)
        with pytest.raises(ValueError, match="missing in transform"):
            scaler.transform(df_val)

    def test_transform_on_different_data(self):
        """Scaler fitted on train must apply same parameters to val data."""
        rng = np.random.default_rng(0)
        df_train = pd.DataFrame({
            'unit_number': [1] * 20,
            'time_in_cycles': range(1, 21),
            'sensor_1': rng.normal(10, 2, 20),
        })
        df_val = pd.DataFrame({
            'unit_number': [2] * 10,
            'time_in_cycles': range(1, 11),
            'sensor_1': rng.normal(10, 2, 10),
        })
        scaler = FeatureScaler()
        scaler.fit(df_train)
        result = scaler.transform(df_val)

        # Median of train is used — val median won't be exactly 0
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df_val.shape

    def test_no_nan_in_output(self):
        """Output must contain no NaN values for well-behaved input."""
        df = _make_df()
        scaler = FeatureScaler()
        scaler.fit(df)
        result = scaler.transform(df)
        assert not result.isnull().any().any()

    def test_constant_sensor_scaled_to_zero(self):
        """Constant sensor column must be scaled to all zeros."""
        df = _make_df(constant_sensors=True)
        scaler = FeatureScaler()
        scaler.fit(df)
        result = scaler.transform(df)
        for col in scaler.feature_cols_:
            np.testing.assert_array_almost_equal(result[col].to_numpy(), 0.0)


# ---------------------------------------------------------------------------
# TestFitTransform
# ---------------------------------------------------------------------------

class TestFitTransform:
    """Tests for the fit_transform() inherited method."""

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        """fit_transform() must produce the same result as fit() then transform()."""
        df = _make_df()
        scaler1 = FeatureScaler()
        result1 = scaler1.fit_transform(df)

        scaler2 = FeatureScaler()
        scaler2.fit(df)
        result2 = scaler2.transform(df)

        pd.testing.assert_frame_equal(
            pd.DataFrame(result1),
            result2
        )