import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from sksurv.functions import StepFunction

from src.models.cox import CoxPiecewise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Returns a seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_survival_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generates a minimal synthetic dataset compatible with Cox survival models.

    Produces a feature matrix and a structured survival array with realistic
    event times and a mix of observed failures and censored observations.

    Returns:
        Tuple of (X, y_surv) where X has shape (100, 5) and y_surv is a
        structured array with dtype [('evento', bool), ('tiempo', float)].
    """
    X = rng.normal(size=(100, 5))
    eventos = rng.choice([True, False], size=100, p=[0.7, 0.3])
    tiempos = rng.uniform(low=10.0, high=200.0, size=100)
    y_surv = np.array(
        list(zip(eventos, tiempos)),
        dtype=[('evento', bool), ('tiempo', float)]
    )
    return X, y_surv


@pytest.fixture
def fitted_model(
    synthetic_survival_data: tuple[np.ndarray, np.ndarray]
) -> CoxPiecewise:
    """Returns a fitted CoxPiecewise model with default parameters."""
    X, y_surv = synthetic_survival_data
    model = CoxPiecewise(alphas=0.1, clipping_threshold=80)
    model.fit(X, y_surv)
    return model


def _make_step_function(times: np.ndarray, probs: np.ndarray) -> StepFunction:
    """Builds a StepFunction mimicking sksurv survival function output.

    Args:
        times: Array of time points.
        probs: Array of survival probabilities at each time point.

    Returns:
        StepFunction instance evaluable at arbitrary time points.
    """
    return StepFunction(x=times, y=probs)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for CoxPiecewise initialization."""

    def test_default_parameters(self):
        """Default parameters must match the documented defaults."""
        model = CoxPiecewise()
        assert model.alphas is None
        assert model.l1_ratio == 0.5
        assert model.confidence_threshold == 0.5
        assert model.clipping_threshold == 125

    def test_custom_parameters(self):
        """Custom parameters must be stored as provided."""
        model = CoxPiecewise(
            alphas=0.01,
            l1_ratio=1.0,
            confidence_threshold=0.9,
            clipping_threshold=110
        )
        assert model.alphas == 0.01
        assert model.l1_ratio == 1.0
        assert model.confidence_threshold == 0.9
        assert model.clipping_threshold == 110

    def test_is_fitted_false_at_init(self):
        """Model must not be marked as fitted before calling fit()."""
        model = CoxPiecewise()
        assert model.is_fitted_ is False

    def test_model_none_at_init(self):
        """model_ must be None before calling fit()."""
        model = CoxPiecewise()
        assert model.model_ is None


# ---------------------------------------------------------------------------
# prepare_training_data
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:
    """Tests for survival data preparation."""

    def _make_motor_df(
        self,
        unit: int,
        cycles: int,
        evento: int,
        rul_last: float
    ) -> pd.DataFrame:
        """Builds a minimal motor CSV DataFrame.

        Args:
            unit: Motor unit identifier.
            cycles: Number of cycles.
            evento: Event indicator (1=failure, 0=censored).
            rul_last: RUL value at the last cycle.

        Returns:
            DataFrame mimicking a per-motor clean CSV file.
        """
        return pd.DataFrame({
            'time_in_cycles': list(range(1, cycles + 1)),
            'RUL': list(range(cycles - 1, -1, -1)) if evento == 1
                   else [rul_last] * cycles,
            'evento': [evento] * cycles,
            'sensor_1': np.random.default_rng(unit).normal(size=cycles),
            'sensor_2': np.random.default_rng(unit + 1).normal(size=cycles),
        })

    def test_y_surv_has_correct_dtype(self):
        """y_surv must have structured dtype with evento and tiempo fields."""
        motor_df = self._make_motor_df(1, 5, evento=1, rul_last=0)
        model = CoxPiecewise()

        with patch('src.models.cox.pd.read_csv', return_value=motor_df):
            _, y_surv, _ = model.prepare_training_data([1])

        assert y_surv.dtype.names == ('evento', 'tiempo')

    def test_train_motor_has_evento_true(self):
        """Train motors (evento=1) must have evento=True in y_surv."""
        motor_df = self._make_motor_df(1, 5, evento=1, rul_last=0)
        model = CoxPiecewise()

        with patch('src.models.cox.pd.read_csv', return_value=motor_df):
            _, y_surv, _ = model.prepare_training_data([1])

        assert y_surv['evento'].all()

    def test_censored_motor_has_evento_false(self):
        """Censored motors (evento=0) must have evento=False in y_surv."""
        motor_df = self._make_motor_df(1, 4, evento=0, rul_last=10)
        model = CoxPiecewise()

        with patch('src.models.cox.pd.read_csv', return_value=motor_df):
            _, y_surv, _ = model.prepare_training_data([1])

        assert not y_surv['evento'].any()

    def test_train_motor_tiempo_equals_total_lifetime(self):
        """Train motor tiempo must equal last_cycle + RUL_at_last_cycle."""
        cycles = 5
        motor_df = self._make_motor_df(1, cycles, evento=1, rul_last=0)
        model = CoxPiecewise()

        with patch('src.models.cox.pd.read_csv', return_value=motor_df):
            _, y_surv, _ = model.prepare_training_data([1])

        # last cycle = cycles, RUL at last = 0 → tiempo = cycles + 0 = cycles
        expected_tiempo = float(cycles)
        assert (y_surv['tiempo'] == expected_tiempo).all()

    def test_censored_motor_tiempo_equals_last_observed_cycle(self):
        """Censored motor tiempo must equal the last observed cycle."""
        cycles = 4
        motor_df = self._make_motor_df(1, cycles, evento=0, rul_last=10)
        model = CoxPiecewise()

        with patch('src.models.cox.pd.read_csv', return_value=motor_df):
            _, y_surv, _ = model.prepare_training_data([1])

        expected_tiempo = float(cycles)
        assert (y_surv['tiempo'] == expected_tiempo).all()

    def test_all_rows_of_same_motor_share_tiempo(self):
        """All rows of the same motor must have identical tiempo values."""
        motor_df = self._make_motor_df(1, 5, evento=1, rul_last=0)
        model = CoxPiecewise()

        with patch('src.models.cox.pd.read_csv', return_value=motor_df):
            _, y_surv, _ = model.prepare_training_data([1])

        assert len(np.unique(y_surv['tiempo'])) == 1

    def test_groups_match_motor_ids(self):
        """Groups array must contain one entry per row mapped to motor id."""
        motor_df = self._make_motor_df(1, 5, evento=1, rul_last=0)
        model = CoxPiecewise()

        with patch('src.models.cox.pd.read_csv', return_value=motor_df):
            _, _, groups = model.prepare_training_data([1])

        assert (groups == 1).all()
        assert len(groups) == 5

    def test_x_excludes_non_feature_columns(self):
        """X must not contain time_in_cycles, RUL, or evento columns."""
        motor_df = self._make_motor_df(1, 5, evento=1, rul_last=0)
        model = CoxPiecewise()

        with patch('src.models.cox.pd.read_csv', return_value=motor_df):
            X, _, _ = model.prepare_training_data([1])

        assert 'time_in_cycles' not in X.columns
        assert 'RUL' not in X.columns
        assert 'evento' not in X.columns


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class TestFit:
    """Tests for the fit method."""

    def test_fit_marks_model_as_fitted(self, synthetic_survival_data):
        """After successful fit, is_fitted_ must be True."""
        X, y_surv = synthetic_survival_data
        model = CoxPiecewise(alphas=0.1)
        model.fit(X, y_surv)
        assert model.is_fitted_ is True

    def test_fit_sets_model_(self, synthetic_survival_data):
        """After successful fit, model_ must not be None."""
        X, y_surv = synthetic_survival_data
        model = CoxPiecewise(alphas=0.1)
        model.fit(X, y_surv)
        assert model.model_ is not None

    def test_fit_returns_self(self, synthetic_survival_data):
        """fit() must return self for sklearn pipeline compatibility."""
        X, y_surv = synthetic_survival_data
        model = CoxPiecewise(alphas=0.1)
        result = model.fit(X, y_surv)
        assert result is model

    def test_fit_failure_marks_model_as_not_fitted(self, synthetic_survival_data):
        """On fitting failure, is_fitted_ must remain False."""
        X, y_surv = synthetic_survival_data
        model = CoxPiecewise(alphas=0.1)
        with patch(
            'src.models.cox.CoxnetSurvivalAnalysis.fit',
            side_effect=RuntimeError("mocked failure")
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y_surv)
        assert model.is_fitted_ is False

    def test_fit_failure_sets_model_to_none(self, synthetic_survival_data):
        """On fitting failure, model_ must be None."""
        X, y_surv = synthetic_survival_data
        model = CoxPiecewise(alphas=0.1)
        with patch(
            'src.models.cox.CoxnetSurvivalAnalysis.fit',
            side_effect=RuntimeError("mocked failure")
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y_surv)
        assert model.model_ is None

    def test_fit_failure_emits_runtime_warning(self, synthetic_survival_data):
        """On fitting failure, a RuntimeWarning must be emitted."""
        X, y_surv = synthetic_survival_data
        model = CoxPiecewise(alphas=0.1)
        with patch(
            'src.models.cox.CoxnetSurvivalAnalysis.fit',
            side_effect=RuntimeError("mocked failure")
        ):
            with pytest.warns(RuntimeWarning, match="Fit failed"):
                model.fit(X, y_surv)


# ---------------------------------------------------------------------------
# _survival_to_rul
# ---------------------------------------------------------------------------

class TestSurvivalToRul:
    """Tests for the survival function to RUL conversion."""

    def _make_model(
        self,
        confidence_threshold: float = 0.5,
        clipping_threshold: int = 200
    ) -> CoxPiecewise:
        """Returns an unfitted CoxPiecewise with specified thresholds."""
        return CoxPiecewise(
            confidence_threshold=confidence_threshold,
            clipping_threshold=clipping_threshold
        )

    def test_threshold_detected_correctly(self):
        """RUL must equal t_failure - 1 when threshold is reached.

        Setup: S(t) = [1.0, 0.8, 0.4, 0.2] at t = [1, 2, 3, 4]
            F(t) = [0.0, 0.2, 0.6, 0.8]
            confidence_threshold = 0.5 → first F(t) >= 0.5 at t=3
            RUL = 3 - 1 = 2
        """
        times = np.array([1.0, 2.0, 3.0, 4.0])
        probs = np.array([1.0, 0.8, 0.4, 0.2])
        sf = _make_step_function(times, probs)

        model = self._make_model(confidence_threshold=0.5)
        rul = model._survival_to_rul(np.array([sf]))

        assert rul[0] == pytest.approx(2.0)

    def test_fallback_when_threshold_never_reached(self):
        """When F(t) never reaches threshold, RUL must use max time - 1.

        Setup: S(t) = [1.0, 0.9, 0.8] at t = [1, 2, 3]
            F(t) = [0.0, 0.1, 0.2] — never reaches 0.9
            RUL = 3 - 1 = 2
        """
        times = np.array([1.0, 2.0, 3.0])
        probs = np.array([1.0, 0.9, 0.8])
        sf = _make_step_function(times, probs)

        model = self._make_model(confidence_threshold=0.9)
        rul = model._survival_to_rul(np.array([sf]))

        assert rul[0] == pytest.approx(2.0)

    def test_rul_never_negative(self):
        """RUL must never be negative even when t_failure = 1."""
        times = np.array([1.0, 2.0])
        probs = np.array([0.3, 0.1])  # F(1) = 0.7 >= 0.5 → t_failure = 1
        sf = _make_step_function(times, probs)

        model = self._make_model(confidence_threshold=0.5)
        rul = model._survival_to_rul(np.array([sf]))

        assert rul[0] >= 0.0

    def test_clip_applied_to_rul(self):
        """RUL must be clipped to clipping_threshold."""
        times = np.array([1.0, 500.0])
        probs = np.array([1.0, 0.9])  # F never reaches 0.5 → fallback = 500 - 1 = 499
        sf = _make_step_function(times, probs)

        model = self._make_model(confidence_threshold=0.5, clipping_threshold=80)
        rul = model._survival_to_rul(np.array([sf]))

        assert rul[0] <= 80.0

    def test_processes_multiple_survival_functions(self):
        """Must return one RUL value per input survival function."""
        times = np.array([1.0, 2.0, 3.0])
        sf1 = _make_step_function(times, np.array([1.0, 0.8, 0.3]))
        sf2 = _make_step_function(times, np.array([1.0, 0.6, 0.2]))
        sf3 = _make_step_function(times, np.array([1.0, 0.9, 0.7]))

        model = self._make_model(confidence_threshold=0.5)
        rul = model._survival_to_rul(np.array([sf1, sf2, sf3]))

        assert rul.shape == (3,)

    def test_higher_confidence_threshold_gives_higher_rul(self):
        """A stricter confidence threshold must yield a higher or equal RUL.

        A higher threshold means we wait for higher failure probability,
        which occurs later in time, yielding more remaining life estimated.
        """
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        probs = np.array([1.0, 0.85, 0.60, 0.35, 0.10])
        sf = _make_step_function(times, probs)

        model_lenient = self._make_model(confidence_threshold=0.5)
        model_strict = self._make_model(confidence_threshold=0.9)

        rul_lenient = model_lenient._survival_to_rul(np.array([sf]))[0]
        rul_strict = model_strict._survival_to_rul(np.array([sf]))[0]

        assert rul_strict >= rul_lenient


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    """Tests for the predict method."""

    def test_predict_before_fit_returns_nan(self):
        """Calling predict before fit must return an array of NaN values."""
        X = np.random.default_rng(0).normal(size=(10, 5))
        model = CoxPiecewise()
        preds = model.predict(X)
        assert np.all(np.isnan(preds))

    def test_predict_before_fit_returns_correct_shape(self):
        """NaN array before fit must match the number of input samples."""
        X = np.random.default_rng(0).normal(size=(15, 5))
        model = CoxPiecewise()
        preds = model.predict(X)
        assert preds.shape == (15,)

    def test_predict_returns_correct_shape(self, fitted_model, synthetic_survival_data):
        """Predictions must have the same number of rows as the input."""
        X, _ = synthetic_survival_data
        preds = fitted_model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_clips_to_threshold(self, fitted_model, synthetic_survival_data):
        """All predictions must be at or below clipping_threshold."""
        X, _ = synthetic_survival_data
        preds = fitted_model.predict(X)
        assert (preds <= fitted_model.clipping_threshold).all()

    def test_predict_returns_non_negative_values(self, fitted_model, synthetic_survival_data):
        """All predictions must be non-negative."""
        X, _ = synthetic_survival_data
        preds = fitted_model.predict(X)
        assert (preds >= 0.0).all()

    def test_predict_returns_float_array(self, fitted_model, synthetic_survival_data):
        """Predictions must be returned as a floating point numpy array."""
        X, _ = synthetic_survival_data
        preds = fitted_model.predict(X)
        assert preds.dtype.kind == 'f'