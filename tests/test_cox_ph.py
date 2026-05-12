"""Tests for CoxPHModel (src/models/cox_ph.py).

Uses a mix of:
    - Synthetic data for unit tests (shape, NaN, warnings) — fast, no pipeline
    - motor_1_pipeline_output from conftest for integration tests

Design philosophy:
    CoxPHModel may or may not converge depending on the event rate,
    data structure, and baseline_estimation_method. Tests verify:
        1. The model handles non-convergence gracefully (no exceptions,
           is_fitted_=False, NaN predictions with appropriate warnings)
        2. When convergence IS achieved (breslow + penalizer + sufficient data),
           the model produces valid predictions via the death curve F(t)
        3. The spline baseline may fail on synthetic high-dimensional data —
           this is expected and documented. Tests verify the silent failure
           contract (RuntimeWarning + is_fitted_=False) rather than convergence.

    Synthetic fixtures use breslow + penalizer=0.5 and 10 motors (5% event
    rate) to encourage convergence. Integration tests use motor_1_pipeline_output
    which may not converge — expected and documented as a negative result
    consistent with WeibullAFTModel on C-MAPSS FD001.

RUL prediction via death curve:
    1. Fit CoxPH with t_stop (duration) and evento (event indicator)
    2. Compute S(t|X) = S₀(t)^exp(β'X) — survival function
    3. Derive F(t|X) = 1 - S(t|X) — death curve
    4. Find t* where F(t*) = confidence_threshold
    5. RUL = max(t* - t_stop_current, 0), clipped to clipping_threshold

baseline_estimation_method note:
    'spline' baseline requires convergence of a parametric optimizer.
    With synthetic high-dimensional PCA data and few events, convergence
    is not guaranteed even with penalizer. The model emits RuntimeWarning
    and sets is_fitted_=False — consistent with the SVRModel convention
    where kernel-specific parameters are always passed but may not help.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.models.cox_ph import CoxPHModel


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
) -> CoxPHModel:
    """CoxPHModel fitted on synthetic data with breslow + penalizer=0.5."""
    model = CoxPHModel(
        baseline_estimation_method='breslow',
        penalizer=0.5,
    )
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
def fitted_model_real(pipeline_output: dict) -> CoxPHModel:
    """CoxPHModel fitted on motor 1 — may NOT converge (0.36% events).

    This is the documented negative result for the paper, consistent
    with WeibullAFTModel behavior on C-MAPSS FD001.
    """
    model = CoxPHModel(
        baseline_estimation_method='breslow',
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
        model = CoxPHModel()
        assert model.clipping_threshold == 125
        assert model.confidence_threshold == 0.8
        assert model.baseline_estimation_method == 'breslow'
        assert model.n_baseline_knots == 3
        assert model.penalizer == 0.0
        assert model.l1_ratio == 0.0

    def test_custom_params(self):
        model = CoxPHModel(
            clipping_threshold=115,
            confidence_threshold=0.5,
            baseline_estimation_method='spline',
            n_baseline_knots=5,
            penalizer=0.1,
            l1_ratio=0.5,
        )
        assert model.clipping_threshold == 115
        assert model.confidence_threshold == 0.5
        assert model.baseline_estimation_method == 'spline'
        assert model.n_baseline_knots == 5
        assert model.penalizer == 0.1
        assert model.l1_ratio == 0.5

    def test_not_fitted_at_init(self):
        assert CoxPHModel().is_fitted_ is False

    def test_model_none_at_init(self):
        assert CoxPHModel().model_ is None

    def test_feature_names_empty_at_init(self):
        assert CoxPHModel().feature_names_ == []

    def test_sklearn_get_params(self):
        params = CoxPHModel().get_params()
        assert set(params.keys()) == {
            'clipping_threshold', 'confidence_threshold',
            'baseline_estimation_method', 'n_baseline_knots',
            'penalizer', 'l1_ratio',
        }


# ---------------------------------------------------------------------------
# TestPrepareTrainingData
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="sliding window pipeline"):
            CoxPHModel().prepare_training_data(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth,
    ):
        model = CoxPHModel(penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.fit(X_synth, y_rul_synth,
                               t_stop=t_stop_synth, evento=evento_synth)
        assert result is model

    def test_fit_is_fitted_is_bool(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth,
    ):
        """is_fitted_ must be bool — never raises regardless of convergence."""
        model = CoxPHModel(penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_synth, y_rul_synth,
                      t_stop=t_stop_synth, evento=evento_synth)
        assert isinstance(model.is_fitted_, bool)

    def test_fit_breslow_converges(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth,
    ):
        """breslow + penalizer=0.5 must converge on synthetic 5% event data."""
        model = CoxPHModel(baseline_estimation_method='breslow', penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_synth, y_rul_synth,
                      t_stop=t_stop_synth, evento=evento_synth)
        assert model.is_fitted_ is True
        assert model.model_ is not None

    def test_fit_sets_feature_names(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth, n_components,
    ):
        """feature_names_ must be set after successful fit."""
        model = CoxPHModel(penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_synth, y_rul_synth,
                      t_stop=t_stop_synth, evento=evento_synth)
        if model.is_fitted_:
            assert len(model.feature_names_) == n_components
            assert model.feature_names_[0] == 'PC_1'
            assert model.feature_names_[-1] == f'PC_{n_components}'

    def test_fit_without_t_stop_fails_gracefully(self, X_synth, y_rul_synth):
        """Missing t_stop must emit RuntimeWarning and set is_fitted_=False."""
        model = CoxPHModel()
        with pytest.warns(RuntimeWarning, match="t_stop"):
            model.fit(X_synth, y_rul_synth, t_stop=None)
        assert model.is_fitted_ is False

    def test_fit_without_t_stop_feature_names_empty(
        self, X_synth, y_rul_synth,
    ):
        """feature_names_ must remain empty when t_stop=None — assigned after guard."""
        model = CoxPHModel()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_synth, y_rul_synth, t_stop=None)
        assert model.feature_names_ == []

    def test_fit_without_evento_no_exception(
        self, X_synth, y_rul_synth, t_stop_synth,
    ):
        """Absent evento must not raise — defaults to all-ones internally."""
        model = CoxPHModel(penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_synth, y_rul_synth, t_stop=t_stop_synth, evento=None)
        assert isinstance(model.is_fitted_, bool)

    def test_fit_ignores_groups(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth, n_windows,
    ):
        """groups kwarg must be accepted and silently ignored."""
        model = CoxPHModel(penalizer=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_synth, y_rul_synth, t_stop=t_stop_synth,
                      evento=evento_synth,
                      groups=np.ones(n_windows, dtype=int))
        assert isinstance(model.is_fitted_, bool)

    def test_fit_removes_nonpositive_duration_warns(
        self, X_synth, y_rul_synth, evento_synth,
    ):
        """Rows with duration <= 0 must be removed with RuntimeWarning."""
        t_stop_bad = np.tile(np.arange(1, 21, dtype=float), 10)
        t_stop_bad[0] = 0.0
        model = CoxPHModel(penalizer=0.5)
        with pytest.warns(RuntimeWarning, match="duration <= 0"):
            model.fit(X_synth, y_rul_synth,
                      t_stop=t_stop_bad, evento=evento_synth)

    def test_fit_spline_silent_failure_contract(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth,
    ):
        """spline without penalizer may fail — must emit RuntimeWarning,
        never raise, and leave is_fitted_=False on failure."""
        model = CoxPHModel(baseline_estimation_method='spline',
                           n_baseline_knots=3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            model.fit(X_synth, y_rul_synth,
                      t_stop=t_stop_synth, evento=evento_synth)
        assert isinstance(model.is_fitted_, bool)
        if not model.is_fitted_:
            assert any(issubclass(x.category, RuntimeWarning) for x in w)

    def test_fit_low_event_rate_no_exception(self, pipeline_output):
        """Low event rate (motor 1 only) must not raise — documented negative result."""
        model = CoxPHModel(penalizer=0.0)
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
            result = CoxPHModel().predict(X_synth)
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
        assert CoxPHModel().predict_survival_function(X_synth) is None

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
        """S(t|X) must be non-increasing over time for each window."""
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth)
        assert sf is not None
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
        assert CoxPHModel().predict_death_curve(X_synth) is None

    def test_returns_dataframe_if_fitted(self, fitted_model_synth, X_synth):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        ft = fitted_model_synth.predict_death_curve(X_synth)
        assert isinstance(ft, pd.DataFrame)

    def test_equals_one_minus_survival(self, fitted_model_synth, X_synth):
        """F(t|X) = 1 - S(t|X) exactly."""
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
        """F(t|X) must be non-decreasing over time for each window."""
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
        result = CoxPHModel().predict_with_time(X_synth, t_stop_synth)
        assert result.shape == (n_windows,)
        assert np.isnan(result).all()

    def test_output_shape(
        self, fitted_model_synth, X_synth, t_stop_synth, n_windows,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        result = fitted_model_synth.predict_with_time(X_synth, t_stop_synth)
        assert result.shape == (n_windows,)

    def test_output_non_negative(
        self, fitted_model_synth, X_synth, t_stop_synth,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        result = fitted_model_synth.predict_with_time(X_synth, t_stop_synth)
        finite = np.isfinite(result)
        assert (result[finite] >= 0.0).all()

    def test_output_clipped(
        self, fitted_model_synth, X_synth, t_stop_synth,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        result = fitted_model_synth.predict_with_time(X_synth, t_stop_synth)
        finite = np.isfinite(result)
        assert (result[finite] <= fitted_model_synth.clipping_threshold).all()

    def test_output_is_float(self, X_synth, t_stop_synth):
        result = CoxPHModel().predict_with_time(X_synth, t_stop_synth)
        assert np.issubdtype(result.dtype, np.floating)

    def test_lower_confidence_gives_lower_rul(
        self, X_synth, y_rul_synth, t_stop_synth, evento_synth,
    ):
        """Lower confidence_threshold → earlier t* on F(t) → lower mean RUL.

        F(t*) = confidence_threshold → t* is where that fraction has died.
        Lower threshold → t* earlier → RUL = t* - t_stop smaller.

        Maintenance semantics:
            Low threshold = conservative = alert sooner (few motors dead)
            High threshold = aggressive  = alert later  (most motors dead)
        """
        m_low  = CoxPHModel(confidence_threshold=0.3, penalizer=0.5)
        m_high = CoxPHModel(confidence_threshold=0.8, penalizer=0.5)
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
        """print_summary must not raise even if concordance index fails."""
        fitted_model_synth.print_summary()

    def test_no_error_if_not_fitted(self, capsys):
        CoxPHModel().print_summary()
        captured = capsys.readouterr()
        assert "not fitted" in captured.out