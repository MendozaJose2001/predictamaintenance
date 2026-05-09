"""Tests for WeibullAFTModel.

Uses a mix of:
    - Synthetic data for unit tests (shape, NaN, warnings) — fast, no pipeline
    - motor_1_pipeline_output from conftest for integration tests

Design philosophy:
    WeibullAFTModel may or may not converge depending on the event rate
    and data structure. Tests verify:
        1. The model handles non-convergence gracefully (no exceptions,
           is_fitted_=False, NaN predictions with appropriate warnings)
        2. When convergence IS achieved (with penalizer and sufficient data),
           the model produces valid predictions via the death curve F(t)

    Synthetic fixtures use penalizer=0.5 and 10 motors (5% event rate)
    to encourage convergence. Integration tests use motor_1_pipeline_output
    which may not converge — expected and documented as a negative result.

RUL prediction via death curve (Enfoque B):
    1. Fit WeibullAFT with t_stop (duration) and evento (event indicator)
    2. Compute S(t) — survival function
    3. Derive F(t) = 1 - S(t) — death curve
    4. Find t* where F(t*) = confidence_threshold
    5. RUL = t* - t_stop_current
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.models.weibull_aft import WeibullAFTModel


# ---------------------------------------------------------------------------
# Synthetic fixtures — 10 motors × 20 cycles = 200 windows, 10 events (5%)
# ---------------------------------------------------------------------------

@pytest.fixture
def n_windows() -> int:
    return 200


@pytest.fixture
def n_components() -> int:
    return 5


@pytest.fixture
def X_synth(n_windows: int, n_components: int) -> np.ndarray:
    """Synthetic feature matrix — standardized like PCA output."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(n_windows, n_components))


@pytest.fixture
def t_stop_synth(n_windows: int) -> np.ndarray:
    """Synthetic t_stop — cycles 1..20 repeated for 10 motors."""
    return np.tile(np.arange(1, 21, dtype=float), 10)


@pytest.fixture
def evento_synth(n_windows: int) -> np.ndarray:
    """Synthetic evento — last window of each of 10 motors has event=1."""
    evento = np.zeros(n_windows, dtype=int)
    for i in range(10):
        evento[(i + 1) * 20 - 1] = 1
    return evento


@pytest.fixture
def y_rul_synth(n_windows: int) -> np.ndarray:
    """Synthetic y_rul — decreasing within each motor."""
    return np.tile(np.linspace(19, 0, 20), 10)


@pytest.fixture
def fitted_model_synth(
    X_synth: np.ndarray,
    y_rul_synth: np.ndarray,
    t_stop_synth: np.ndarray,
    evento_synth: np.ndarray,
) -> WeibullAFTModel:
    """WeibullAFTModel fitted on synthetic data with penalizer=0.5."""
    model = WeibullAFTModel(penalizer=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_synth, y_rul_synth,
                  t_stop=t_stop_synth, evento=evento_synth)
    return model


# ---------------------------------------------------------------------------
# Integration fixtures — real pipeline data (motor 1)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pipeline_output(motor_1_pipeline_output: dict) -> dict:
    return motor_1_pipeline_output


@pytest.fixture(scope='module')
def fitted_model_real(pipeline_output: dict) -> WeibullAFTModel:
    """WeibullAFTModel fitted on motor 1 — may NOT converge (0.36% events).

    This is the documented negative result for the paper.
    """
    model = WeibullAFTModel(
        clipping_threshold=125,
        confidence_threshold=0.8,
        penalizer=0.1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(
            pipeline_output['X'],
            pipeline_output['y_rul'],
            t_stop=pipeline_output['t_stop'],
            evento=pipeline_output['evento'],
        )
    return model


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_params(self):
        model = WeibullAFTModel()
        assert model.clipping_threshold == 125
        assert model.confidence_threshold == 0.8
        assert model.penalizer == 0.0
        assert model.l1_ratio == 0.0
        assert model.fit_intercept is True

    def test_custom_params(self):
        model = WeibullAFTModel(
            clipping_threshold=110,
            confidence_threshold=0.9,
            penalizer=0.1,
            l1_ratio=0.5,
            fit_intercept=False,
        )
        assert model.clipping_threshold == 110
        assert model.confidence_threshold == 0.9
        assert model.penalizer == 0.1
        assert model.l1_ratio == 0.5
        assert model.fit_intercept is False

    def test_not_fitted_at_init(self):
        assert WeibullAFTModel().is_fitted_ is False

    def test_model_none_at_init(self):
        assert WeibullAFTModel().model_ is None

    def test_feature_names_empty_at_init(self):
        assert WeibullAFTModel().feature_names_ == []


# ---------------------------------------------------------------------------
# TestPrepareTrainingData
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="sliding window pipeline"):
            WeibullAFTModel().prepare_training_data(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth,
    ):
        model = WeibullAFTModel(penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.fit(X_synth, y_rul_synth,
                               t_stop=t_stop_synth, evento=evento_synth)
        assert result is model

    def test_fit_is_fitted_is_bool(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth,
    ):
        """is_fitted_ must be bool — never exception regardless of convergence."""
        model = WeibullAFTModel(penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_synth, y_rul_synth,
                      t_stop=t_stop_synth, evento=evento_synth)
        assert isinstance(model.is_fitted_, bool)

    def test_fit_sets_feature_names(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth, n_components,
    ):
        """feature_names_ set regardless of convergence."""
        model = WeibullAFTModel(penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_synth, y_rul_synth,
                      t_stop=t_stop_synth, evento=evento_synth)
        assert len(model.feature_names_) == n_components
        assert model.feature_names_[0] == 'PC_1'

    def test_fit_without_t_stop_fails_gracefully(self, X_synth, y_rul_synth):
        model = WeibullAFTModel()
        with pytest.warns(RuntimeWarning, match="t_stop is required"):
            model.fit(X_synth, y_rul_synth, t_stop=None)
        assert model.is_fitted_ is False

    def test_fit_without_evento_no_exception(
        self, X_synth, y_rul_synth, t_stop_synth,
    ):
        model = WeibullAFTModel(penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_synth, y_rul_synth, t_stop=t_stop_synth, evento=None)
        assert isinstance(model.is_fitted_, bool)

    def test_fit_ignores_groups(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth, n_windows,
    ):
        model = WeibullAFTModel(penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_synth, y_rul_synth, t_stop=t_stop_synth,
                      evento=evento_synth, groups=np.ones(n_windows, dtype=int))
        assert isinstance(model.is_fitted_, bool)

    def test_fit_removes_nonpositive_duration_warns(
        self, X_synth, y_rul_synth, evento_synth,
    ):
        t_stop_bad = np.tile(np.arange(1, 21, dtype=float), 10)
        t_stop_bad[0] = 0.0
        model = WeibullAFTModel(penalizer=0.5)
        with pytest.warns(RuntimeWarning, match="duration <= 0"):
            model.fit(X_synth, y_rul_synth, t_stop=t_stop_bad,
                      evento=evento_synth)

    def test_fit_low_event_rate_no_exception(self, pipeline_output):
        """Low event rate (motor 1 only) must not raise — documented negative result."""
        model = WeibullAFTModel(penalizer=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                pipeline_output['X'],
                pipeline_output['y_rul'],
                t_stop=pipeline_output['t_stop'],
                evento=pipeline_output['evento'],
            )
        assert isinstance(model.is_fitted_, bool)


# ---------------------------------------------------------------------------
# TestPredict — always returns NaN + warning
# ---------------------------------------------------------------------------

class TestPredict:

    def test_returns_nan_always(self, X_synth, n_windows):
        with pytest.warns(RuntimeWarning, match="predict_with_time"):
            result = WeibullAFTModel().predict(X_synth)
        assert result.shape == (n_windows,)
        assert np.isnan(result).all()

    def test_returns_nan_even_if_fitted(
        self, fitted_model_synth, X_synth, n_windows,
    ):
        with pytest.warns(RuntimeWarning, match="predict_with_time"):
            result = fitted_model_synth.predict(X_synth)
        assert result.shape == (n_windows,)
        assert np.isnan(result).all()


# ---------------------------------------------------------------------------
# TestPredictSurvivalFunction
# ---------------------------------------------------------------------------

class TestPredictSurvivalFunction:

    def test_returns_none_if_not_fitted(self, X_synth):
        assert WeibullAFTModel().predict_survival_function(X_synth) is None

    def test_returns_dataframe_if_fitted(self, fitted_model_synth, X_synth):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth)
        assert isinstance(sf, pd.DataFrame)

    def test_n_columns_matches_n_windows(
        self, fitted_model_synth, X_synth, n_windows,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth)
        assert sf is not None
        assert sf.shape[1] == n_windows

    def test_values_in_unit_interval(self, fitted_model_synth, X_synth):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth)
        assert sf is not None
        assert (sf.values >= 0.0).all()
        assert (sf.values <= 1.0 + 1e-6).all()

    def test_non_increasing_over_time(self, fitted_model_synth, X_synth):
        """S(t) must be non-increasing over time."""
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth)
        assert sf is not None
        # Check first 5 windows
        for col in sf.columns[:5]:
            diffs = np.diff(sf[col].values)
            assert (diffs <= 1e-6).all()

    def test_real_data_no_exception(self, fitted_model_real, pipeline_output):
        """predict_survival_function must not raise even if not fitted."""
        result = fitted_model_real.predict_survival_function(
            pipeline_output['X']
        )
        assert result is None or isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# TestPredictDeathCurve
# ---------------------------------------------------------------------------

class TestPredictDeathCurve:

    def test_returns_none_if_not_fitted(self, X_synth):
        assert WeibullAFTModel().predict_death_curve(X_synth) is None

    def test_returns_dataframe_if_fitted(self, fitted_model_synth, X_synth):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        ft = fitted_model_synth.predict_death_curve(X_synth)
        assert isinstance(ft, pd.DataFrame)

    def test_equals_one_minus_survival(self, fitted_model_synth, X_synth):
        """F(t) = 1 - S(t) exactly."""
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth)
        ft = fitted_model_synth.predict_death_curve(X_synth)
        assert sf is not None and ft is not None
        np.testing.assert_array_almost_equal(ft.values, 1.0 - sf.values)

    def test_values_in_unit_interval(self, fitted_model_synth, X_synth):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        ft = fitted_model_synth.predict_death_curve(X_synth)
        assert ft is not None
        assert (ft.values >= 0.0).all()
        assert (ft.values <= 1.0 + 1e-6).all()

    def test_non_decreasing_over_time(self, fitted_model_synth, X_synth):
        """F(t) must be non-decreasing over time."""
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        ft = fitted_model_synth.predict_death_curve(X_synth)
        assert ft is not None
        for col in ft.columns[:5]:
            diffs = np.diff(ft[col].values)
            assert (diffs >= -1e-6).all()


# ---------------------------------------------------------------------------
# TestPredictWithTime
# ---------------------------------------------------------------------------

class TestPredictWithTime:

    def test_returns_nan_if_not_fitted(self, X_synth, t_stop_synth, n_windows):
        result = WeibullAFTModel().predict_with_time(X_synth, t_stop_synth)
        assert result.shape == (n_windows,)
        assert np.isnan(result).all()

    def test_output_shape(self, fitted_model_synth, X_synth, t_stop_synth, n_windows):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        result = fitted_model_synth.predict_with_time(X_synth, t_stop_synth)
        assert result.shape == (n_windows,)

    def test_output_non_negative(self, fitted_model_synth, X_synth, t_stop_synth):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        result = fitted_model_synth.predict_with_time(X_synth, t_stop_synth)
        finite = np.isfinite(result)
        assert (result[finite] >= 0.0).all()

    def test_output_clipped(self, fitted_model_synth, X_synth, t_stop_synth):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        result = fitted_model_synth.predict_with_time(X_synth, t_stop_synth)
        finite = np.isfinite(result)
        assert (result[finite] <= fitted_model_synth.clipping_threshold).all()

    def test_output_is_float(self, X_synth, t_stop_synth):
        result = WeibullAFTModel().predict_with_time(X_synth, t_stop_synth)
        assert np.issubdtype(result.dtype, np.floating)

    def test_lower_confidence_gives_lower_rul(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth,
    ):
        """Lower confidence_threshold → earlier t* on F(t) → lower RUL.

        F(t*) = confidence_threshold → t* is where that fraction has died.
        Lower threshold → t* earlier → RUL = t* - t_stop smaller.

        Maintenance semantics:
            Low threshold = conservative = alert sooner (few motors dead)
            High threshold = aggressive = alert later (most motors dead)
        """
        m_low  = WeibullAFTModel(confidence_threshold=0.3, penalizer=0.5)
        m_high = WeibullAFTModel(confidence_threshold=0.8, penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m_low.fit(X_synth, y_rul_synth,
                      t_stop=t_stop_synth, evento=evento_synth)
            m_high.fit(X_synth, y_rul_synth,
                       t_stop=t_stop_synth, evento=evento_synth)
        if m_low.is_fitted_ and m_high.is_fitted_:
            pred_low  = m_low.predict_with_time(X_synth, t_stop_synth)
            pred_high = m_high.predict_with_time(X_synth, t_stop_synth)
            finite = np.isfinite(pred_low) & np.isfinite(pred_high)
            if finite.any():
                # lower confidence → lower t* on F(t) → lower RUL
                assert pred_low[finite].mean() <= pred_high[finite].mean()

    def test_real_data_no_exception(self, fitted_model_real, pipeline_output):
        """predict_with_time on real data must not raise — even if not fitted."""
        result = fitted_model_real.predict_with_time(
            pipeline_output['X'], pipeline_output['t_stop']
        )
        assert result.shape == (pipeline_output['X'].shape[0],)


# ---------------------------------------------------------------------------
# TestPrintSummary
# ---------------------------------------------------------------------------

class TestPrintSummary:

    def test_no_error_if_fitted(self, fitted_model_synth, capsys):
        fitted_model_synth.print_summary()

    def test_no_error_if_not_fitted(self, capsys):
        WeibullAFTModel().print_summary()
        captured = capsys.readouterr()
        assert "not fitted" in captured.out