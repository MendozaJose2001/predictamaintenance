"""Tests for the CoxFrailty model.

These tests require a working R installation with the survival package.
CoxFrailty delegates fitting and prediction to the r_repository layer,
which uses survival::coxph with frailty() for shared frailty estimation.
Integration tests use real C-MAPSS data to guarantee convergence.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.models.cox_frailty import CoxFrailty


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def synthetic_survival_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]:
    """Loads real C-MAPSS data for CoxFrailty integration tests.

    Uses only train motors (evento=1) to guarantee coxph convergence —
    sufficient observed events are required to estimate the frailty term.
    Returns the model instance so that _column_names is available for
    fit() calls in TestFit.

    Returns:
        Tuple of (X, y_surv, groups, model_instance).
    """
    from src.dataset_manager import DatasetManager

    m_train, _ = DatasetManager.split_dataset()

    # Filter only motors with observed failure (evento=1)
    train_motors = [
        idx for idx in m_train
        if pd.read_csv(f'data/clean/data_motor_{idx}.csv')['evento'].iloc[0] == 1
    ]

    model_instance = CoxFrailty()
    X, y_surv, _, groups = model_instance.prepare_training_data(
        np.array(train_motors[:10])
    )

    return X, y_surv, groups, model_instance


@pytest.fixture(scope='module')
def fitted_cox_frailty(
    synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
) -> CoxFrailty:
    """Returns a fitted CoxFrailty model using the real pipeline infrastructure."""
    from sklearn.pipeline import Pipeline
    from src.mad_scaler import MADScaler

    X, y_surv, groups, _ = synthetic_survival_data
    model = CoxFrailty(distribution='gamma', clipping_threshold=80)
    pipeline = Pipeline([
        ('scaler', MADScaler()),
        ('model', model)
    ])
    pipeline.set_output(transform='pandas')
    pipeline.fit(X, y_surv, model__groups=groups)
    return model


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for CoxFrailty initialization."""

    def test_default_parameters(self):
        """Default parameters must match the documented defaults."""
        model = CoxFrailty()
        assert model.distribution == 'gamma'
        assert model.maxit == 300
        assert model.confidence_threshold == 0.5
        assert model.clipping_threshold == 125

    def test_custom_parameters(self):
        """Custom parameters must be stored as provided."""
        model = CoxFrailty(
            distribution='gaussian',
            maxit=500,
            confidence_threshold=0.9,
            clipping_threshold=110
        )
        assert model.distribution == 'gaussian'
        assert model.maxit == 500
        assert model.confidence_threshold == 0.9
        assert model.clipping_threshold == 110

    def test_is_fitted_false_at_init(self):
        """Model must not be marked as fitted before calling fit()."""
        model = CoxFrailty()
        assert model.is_fitted_ is False

    def test_model_name_none_at_init(self):
        """_model_name must be None before calling fit()."""
        model = CoxFrailty()
        assert model._model_name is None


# ---------------------------------------------------------------------------
# prepare_training_data
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:
    """Tests for survival data preparation in counting process format."""

    def _make_motor_df(
        self,
        unit: int,
        cycles: int,
        evento: int,
    ) -> pd.DataFrame:
        """Builds a minimal motor CSV DataFrame."""
        rul_values = list(range(cycles - 1, -1, -1)) if evento == 1 \
            else list(range(10, 10 + cycles))
        return pd.DataFrame({
            'time_in_cycles': list(range(1, cycles + 1)),
            'RUL': rul_values,
            'evento': [evento] * cycles,
            'sensor_1': np.random.default_rng(unit).normal(size=cycles),
            'sensor_2': np.random.default_rng(unit + 1).normal(size=cycles),
        })

    def test_y_fit_has_correct_dtype(self):
        """y_fit must have structured dtype with evento, t_start, t_stop fields."""
        motor_df = self._make_motor_df(1, 5, evento=1)
        model = CoxFrailty()
        with patch('src.models.cox_frailty.pd.read_csv', return_value=motor_df):
            _, y_fit, _, _ = model.prepare_training_data(np.array([1]))
        assert y_fit.dtype.names == ('evento', 't_start', 't_stop')

    def test_t_start_equals_time_minus_one(self):
        """t_start in y_fit must equal time_in_cycles - 1."""
        cycles = 5
        motor_df = self._make_motor_df(1, cycles, evento=1)
        model = CoxFrailty()
        with patch('src.models.cox_frailty.pd.read_csv', return_value=motor_df):
            _, y_fit, _, _ = model.prepare_training_data(np.array([1]))
        expected = motor_df['time_in_cycles'].to_numpy() - 1.0
        np.testing.assert_array_almost_equal(y_fit['t_start'], expected)

    def test_t_stop_equals_time_in_cycles(self):
        """t_stop in y_fit must equal time_in_cycles."""
        cycles = 5
        motor_df = self._make_motor_df(1, cycles, evento=1)
        model = CoxFrailty()
        with patch('src.models.cox_frailty.pd.read_csv', return_value=motor_df):
            _, y_fit, _, _ = model.prepare_training_data(np.array([1]))
        expected = motor_df['time_in_cycles'].to_numpy().astype(float)
        np.testing.assert_array_almost_equal(y_fit['t_stop'], expected)

    def test_t_start_less_than_t_stop(self):
        """t_start must always be strictly less than t_stop."""
        motor_df = self._make_motor_df(1, 5, evento=1)
        model = CoxFrailty()
        with patch('src.models.cox_frailty.pd.read_csv', return_value=motor_df):
            _, y_fit, _, _ = model.prepare_training_data(np.array([1]))
        assert (y_fit['t_start'] < y_fit['t_stop']).all()

    def test_evento_true_only_at_last_cycle_for_train(self):
        """Train motors must have evento=True only at the last cycle."""
        cycles = 5
        motor_df = self._make_motor_df(1, cycles, evento=1)
        model = CoxFrailty()
        with patch('src.models.cox_frailty.pd.read_csv', return_value=motor_df):
            _, y_fit, _, _ = model.prepare_training_data(np.array([1]))
        assert y_fit['evento'].sum() == 1
        assert y_fit['evento'][-1] is np.bool_(True)

    def test_censored_motor_evento_all_false(self):
        """Censored motors must have evento=False for all cycles."""
        motor_df = self._make_motor_df(1, 5, evento=0)
        model = CoxFrailty()
        with patch('src.models.cox_frailty.pd.read_csv', return_value=motor_df):
            _, y_fit, _, _ = model.prepare_training_data(np.array([1]))
        assert not y_fit['evento'].any()

    def test_y_metrics_equals_rul_per_row(self):
        """y_metrics must equal the scalar RUL at each individual cycle."""
        cycles = 5
        motor_df = self._make_motor_df(1, cycles, evento=1)
        model = CoxFrailty()
        with patch('src.models.cox_frailty.pd.read_csv', return_value=motor_df):
            _, _, y_metrics, _ = model.prepare_training_data(np.array([1]))
        expected_rul = motor_df['RUL'].to_numpy()
        np.testing.assert_array_almost_equal(y_metrics, expected_rul)

    def test_x_includes_time_in_cycles(self):
        """X must include time_in_cycles as a feature column."""
        motor_df = self._make_motor_df(1, 5, evento=1)
        model = CoxFrailty()
        with patch('src.models.cox_frailty.pd.read_csv', return_value=motor_df):
            X, _, _, _ = model.prepare_training_data(np.array([1]))
        assert 'time_in_cycles' in X.columns

    def test_x_excludes_rul_and_evento(self):
        """X must not contain RUL or evento columns."""
        motor_df = self._make_motor_df(1, 5, evento=1)
        model = CoxFrailty()
        with patch('src.models.cox_frailty.pd.read_csv', return_value=motor_df):
            X, _, _, _ = model.prepare_training_data(np.array([1]))
        assert 'RUL' not in X.columns
        assert 'evento' not in X.columns

    def test_groups_match_motor_ids(self):
        """Groups array must map each row to its motor unit identifier."""
        motor_df = self._make_motor_df(1, 5, evento=1)
        model = CoxFrailty()
        with patch('src.models.cox_frailty.pd.read_csv', return_value=motor_df):
            _, _, _, groups = model.prepare_training_data(np.array([1]))
        assert (groups == 1).all()
        assert len(groups) == 5


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class TestFit:
    """Tests for the fit method."""

    def _fit_with_pipeline(
        self,
        X: pd.DataFrame,
        y_surv: np.ndarray,
        groups: np.ndarray,
        distribution: str = 'gamma',
        clipping_threshold: int = 125,
    ) -> CoxFrailty:
        """Fits CoxFrailty through the real pipeline infrastructure."""
        from sklearn.pipeline import Pipeline
        from src.mad_scaler import MADScaler

        model = CoxFrailty(
            distribution=distribution,
            clipping_threshold=clipping_threshold
        )
        pipeline = Pipeline([
            ('scaler', MADScaler()),
            ('model', model)
        ])
        pipeline.set_output(transform='pandas')
        pipeline.fit(X, y_surv, model__groups=groups)
        return model

    def test_fit_marks_model_as_fitted(
        self,
        synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
    ):
        """After successful fit, is_fitted_ must be True."""
        X, y_surv, groups, _ = synthetic_survival_data
        model = self._fit_with_pipeline(X, y_surv, groups)
        assert model.is_fitted_ is True

    def test_fit_sets_model_name(
        self,
        synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
    ):
        """After successful fit, _model_name must be a non-empty string."""
        X, y_surv, groups, _ = synthetic_survival_data
        model = self._fit_with_pipeline(X, y_surv, groups)
        assert isinstance(model._model_name, str)
        assert len(model._model_name) > 0

    def test_fit_returns_self(
        self,
        synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
    ):
        """fit() must return self for sklearn pipeline compatibility."""
        from sklearn.pipeline import Pipeline
        from src.mad_scaler import MADScaler

        X, y_surv, groups, _ = synthetic_survival_data
        model = CoxFrailty(distribution='gamma')
        pipeline = Pipeline([('scaler', MADScaler()), ('model', model)])
        pipeline.set_output(transform='pandas')
        pipeline.fit(X, y_surv, model__groups=groups)
        assert pipeline.named_steps['model'] is model

    def test_fit_accepts_groups_via_kwargs(
        self,
        synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
    ):
        """fit() must accept groups as a keyword argument."""
        X, y_surv, groups, _ = synthetic_survival_data
        model = self._fit_with_pipeline(X, y_surv, groups)
        assert model.is_fitted_ is True

    def test_fit_failure_marks_model_as_not_fitted(
        self,
        synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
    ):
        """On fitting failure, is_fitted_ must remain False."""
        from sklearn.pipeline import Pipeline
        from src.mad_scaler import MADScaler

        X, y_surv, groups, _ = synthetic_survival_data
        model = CoxFrailty(distribution='gamma')
        pipeline = Pipeline([('scaler', MADScaler()), ('model', model)])
        pipeline.set_output(transform='pandas')
        with patch(
            'src.models.cox_frailty.fit_cox_frailty',
            side_effect=RuntimeError("mocked failure")
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X, y_surv, model__groups=groups)
        assert model.is_fitted_ is False

    def test_fit_failure_emits_runtime_warning(
        self,
        synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
    ):
        """On fitting failure, a RuntimeWarning must be emitted."""
        from sklearn.pipeline import Pipeline
        from src.mad_scaler import MADScaler

        X, y_surv, groups, _ = synthetic_survival_data
        model = CoxFrailty(distribution='gamma')
        pipeline = Pipeline([('scaler', MADScaler()), ('model', model)])
        pipeline.set_output(transform='pandas')
        with patch(
            'src.models.cox_frailty.fit_cox_frailty',
            side_effect=RuntimeError("mocked failure")
        ):
            with pytest.warns(RuntimeWarning, match="Fit failed"):
                pipeline.fit(X, y_surv, model__groups=groups)


# ---------------------------------------------------------------------------
# _survival_to_rul
# ---------------------------------------------------------------------------

class TestSurvivalToRul:
    """Tests for the survival function to RUL conversion."""

    def _make_model(
        self,
        confidence_threshold: float = 0.5,
        clipping_threshold: int = 200
    ) -> CoxFrailty:
        """Returns an unfitted CoxFrailty with specified thresholds."""
        return CoxFrailty(
            confidence_threshold=confidence_threshold,
            clipping_threshold=clipping_threshold
        )

    def test_threshold_detected_correctly(self):
        """RUL must equal t_failure - t_current when threshold is reached."""
        times = np.array([10.0, 20.0, 30.0])
        probs = np.array([1.0, 0.8, 0.4])
        t_current = np.array([10.0])
        model = self._make_model(confidence_threshold=0.5)
        rul = model._survival_to_rul([(times, probs)], t_current)
        assert rul[0] == pytest.approx(20.0)

    def test_fallback_when_threshold_never_reached(self):
        """When threshold never reached, uses max time - t_current."""
        times = np.array([10.0, 20.0, 30.0])
        probs = np.array([1.0, 0.9, 0.8])
        t_current = np.array([5.0])
        model = self._make_model(confidence_threshold=0.9)
        rul = model._survival_to_rul([(times, probs)], t_current)
        assert rul[0] == pytest.approx(25.0)

    def test_rul_never_negative(self):
        """RUL must never be negative even if t_current > t_failure."""
        times = np.array([1.0, 2.0])
        probs = np.array([0.3, 0.1])
        t_current = np.array([100.0])
        model = self._make_model(confidence_threshold=0.5)
        rul = model._survival_to_rul([(times, probs)], t_current)
        assert rul[0] >= 0.0

    def test_clip_applied(self):
        """RUL must be clipped to clipping_threshold."""
        times = np.array([1.0, 500.0])
        probs = np.array([1.0, 0.9])
        t_current = np.array([0.0])
        model = self._make_model(confidence_threshold=0.5, clipping_threshold=80)
        rul = model._survival_to_rul([(times, probs)], t_current)
        assert rul[0] <= 80.0

    def test_processes_multiple_functions(self):
        """Must return one RUL per input survival function."""
        times = np.array([1.0, 2.0, 3.0])
        sfs = [
            (times, np.array([1.0, 0.8, 0.3])),
            (times, np.array([1.0, 0.6, 0.2])),
            (times, np.array([1.0, 0.9, 0.7])),
        ]
        t_current = np.array([0.0, 0.0, 0.0])
        model = self._make_model(confidence_threshold=0.5)
        rul = model._survival_to_rul(sfs, t_current)
        assert rul.shape == (3,)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    """Tests for the predict method."""

    def test_predict_before_fit_returns_nan(self):
        """Calling predict before fit must return an array of NaN values."""
        X = np.random.default_rng(0).normal(size=(10, 3))
        model = CoxFrailty()
        preds = model.predict(X)
        assert np.all(np.isnan(preds))

    def test_predict_before_fit_returns_correct_shape(self):
        """NaN array before fit must match the number of input samples."""
        X = np.random.default_rng(0).normal(size=(15, 3))
        model = CoxFrailty()
        preds = model.predict(X)
        assert preds.shape == (15,)

    def test_predict_returns_correct_shape(
        self,
        fitted_cox_frailty: CoxFrailty,
        synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
    ):
        """Predictions must have the same number of rows as the input."""
        X, _, _, _ = synthetic_survival_data
        preds = fitted_cox_frailty.predict(X.to_numpy())
        assert preds.shape == (len(X),)

    def test_predict_clips_to_threshold(
        self,
        fitted_cox_frailty: CoxFrailty,
        synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
    ):
        """All predictions must be at or below clipping_threshold."""
        X, _, _, _ = synthetic_survival_data
        preds = fitted_cox_frailty.predict(X.to_numpy())
        assert (preds <= fitted_cox_frailty.clipping_threshold).all()

    def test_predict_returns_non_negative_values(
        self,
        fitted_cox_frailty: CoxFrailty,
        synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
    ):
        """All predictions must be non-negative."""
        X, _, _, _ = synthetic_survival_data
        preds = fitted_cox_frailty.predict(X.to_numpy())
        assert (preds >= 0.0).all()

    def test_predict_returns_float_array(
        self,
        fitted_cox_frailty: CoxFrailty,
        synthetic_survival_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, CoxFrailty]
    ):
        """Predictions must be returned as a floating point numpy array."""
        X, _, _, _ = synthetic_survival_data
        preds = fitted_cox_frailty.predict(X.to_numpy())
        assert preds.dtype.kind == 'f'