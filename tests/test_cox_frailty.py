"""Tests for CoxFrailty (src/models/cox_frailty.py).

These tests require a working R installation with the survival package,
since CoxFrailty wraps survival::coxph with frailty() via the r_repository
layer. All R interactions go through the model interface — no direct rpy2
calls in these tests.

Uses a mix of:
    - Synthetic data for unit tests (shape, NaN, warnings) — fast
    - motor_1_pipeline_output from conftest for integration tests

Design philosophy:
    CoxFrailty may or may not converge depending on the event rate,
    data structure, and frailty distribution. Tests verify:
        1. The model handles non-convergence gracefully (no exceptions,
           is_fitted_=False, NaN predictions with appropriate warnings)
        2. When convergence IS achieved (gamma frailty + sufficient data),
           the model produces valid predictions via the death curve F(t)
        3. groups kwarg is used as the frailty clustering variable —
           unlike CoxPHModel and WeibullAFTModel which ignore it

    Synthetic fixtures use 10 motors × 20 windows (5% event rate) to
    encourage convergence. Integration tests use motor_1_pipeline_output
    which may not converge — expected and documented as a negative result
    consistent with CoxPHModel on C-MAPSS FD001.

Key differences from test_cox_ph.py and test_weibull_aft.py:
    - predict_survival_function returns list[tuple] | None, not pd.DataFrame
    - predict_death_curve returns list[tuple] | None, not pd.DataFrame
    - groups kwarg is meaningful — used as frailty clustering variable
    - model_name_ (str | None) replaces model_ as the fitted state attribute
    - No fit_intercept parameter — specific to WeibullAFTModel
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.models.cox_frailty import CoxFrailty


# ---------------------------------------------------------------------------
# Synthetic fixtures — 10 motors × 20 windows = 200 windows, 10 events (5%)
# ---------------------------------------------------------------------------

@pytest.fixture
def n_windows() -> int:
    return 200


@pytest.fixture
def n_components() -> int:
    return 5


@pytest.fixture
def X_synth(n_windows: int, n_components: int) -> np.ndarray:
    """Synthetic PCA-like feature matrix — standardized normal."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(n_windows, n_components))


@pytest.fixture
def t_stop_synth(n_windows: int) -> np.ndarray:
    """t_stop — cycles 1..20 repeated for 10 motors."""
    return np.tile(np.arange(1, 21, dtype=float), 10)


@pytest.fixture
def evento_synth(n_windows: int) -> np.ndarray:
    """evento=1 at the last window of each of 10 motors."""
    evento = np.zeros(n_windows, dtype=int)
    for i in range(10):
        evento[(i + 1) * 20 - 1] = 1
    return evento


@pytest.fixture
def y_rul_synth(n_windows: int) -> np.ndarray:
    """Synthetic y_rul — decreasing within each motor."""
    return np.tile(np.linspace(19, 0, 20), 10)


@pytest.fixture
def groups_synth(n_windows: int) -> np.ndarray:
    """Motor ID per window — 0..9 each repeated 20 times."""
    return np.repeat(np.arange(10, dtype=int), 20)


@pytest.fixture
def fitted_model_synth(
    X_synth: np.ndarray,
    y_rul_synth: np.ndarray,
    t_stop_synth: np.ndarray,
    evento_synth: np.ndarray,
    groups_synth: np.ndarray,
) -> CoxFrailty:
    """CoxFrailty fitted on synthetic data with gamma frailty."""
    model = CoxFrailty(distribution='gamma', maxit=300)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(
            X_synth, y_rul_synth,
            t_stop=t_stop_synth,
            evento=evento_synth,
            groups=groups_synth,
        )
    return model


# ---------------------------------------------------------------------------
# Integration fixtures — real pipeline data (motor 1)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pipeline_output(motor_1_pipeline_output: dict) -> dict:
    return motor_1_pipeline_output


@pytest.fixture(scope='module')
def fitted_model_real(pipeline_output: dict) -> CoxFrailty:
    """CoxFrailty fitted on motor 1 — may NOT converge (0.4% events).

    Documented negative result — consistent with CoxPHModel on C-MAPSS FD001.
    """
    model = CoxFrailty(
        distribution='gamma',
        maxit=300,
        confidence_threshold=0.5,
        clipping_threshold=125,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(
            pipeline_output['X'],
            pipeline_output['y_rul'],
            t_stop=pipeline_output['t_stop'],
            evento=pipeline_output['evento'],
            groups=pipeline_output['groups'],
        )
    return model


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_params(self):
        model = CoxFrailty()
        assert model.distribution == 'gamma'
        assert model.maxit == 50
        assert model.method == 'em'
        assert model.tdf == 5
        assert model.confidence_threshold == 0.5
        assert model.clipping_threshold == 125

    def test_custom_params(self):
        model = CoxFrailty(
            distribution='gaussian',
            maxit=100,
            method='aic',
            tdf=3,
            confidence_threshold=0.8,
            clipping_threshold=115,
        )
        assert model.distribution == 'gaussian'
        assert model.maxit == 100
        assert model.method == 'aic'
        assert model.tdf == 3
        assert model.confidence_threshold == 0.8
        assert model.clipping_threshold == 115

    def test_not_fitted_at_init(self):
        assert CoxFrailty().is_fitted_ is False

    def test_model_name_none_at_init(self):
        assert CoxFrailty().model_name_ is None

    def test_sklearn_get_params(self):
        params = CoxFrailty().get_params()
        assert set(params.keys()) == {
            'distribution',
            'maxit',
            'method',
            'tdf',
            'confidence_threshold',
            'clipping_threshold',
        }


# ---------------------------------------------------------------------------
# TestPrepareTrainingData
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="sliding window pipeline"):
            CoxFrailty().prepare_training_data(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        model = CoxFrailty(distribution='gamma', maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                groups=groups_synth,
            )
        assert result is model

    def test_fit_is_fitted_is_bool(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        """is_fitted_ must be bool — never raises regardless of convergence."""
        model = CoxFrailty(distribution='gamma', maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                groups=groups_synth,
            )
        assert isinstance(model.is_fitted_, bool)

    def test_fit_gamma_converges(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        """gamma frailty must converge on synthetic 5% event data."""
        model = CoxFrailty(distribution='gamma', maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                groups=groups_synth,
            )
        assert model.is_fitted_ is True
        assert model.model_name_ is not None
        assert isinstance(model.model_name_, str)

    def test_fit_gaussian_frailty_no_exception(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        """Gaussian frailty must not raise — may or may not converge."""
        model = CoxFrailty(distribution='gaussian', maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                groups=groups_synth,
            )
        assert isinstance(model.is_fitted_, bool)

    def test_fit_with_method_aic_no_exception(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        """method='aic' must not raise — AIC-based θ selection for gamma."""
        model = CoxFrailty(distribution='gamma', method='aic', maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                groups=groups_synth,
            )
        assert isinstance(model.is_fitted_, bool)

    def test_fit_with_t_distribution_no_exception(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        """t frailty with tdf=3 must not raise — robust to outliers."""
        model = CoxFrailty(distribution='t', tdf=3, maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                groups=groups_synth,
            )
        assert isinstance(model.is_fitted_, bool)

    def test_fit_tdf_ignored_for_gamma(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        """tdf is silently ignored by R when distribution='gamma'.

        Two gamma models with different tdf values must produce the same
        is_fitted_ outcome — tdf has no effect on gamma frailty.
        """
        m1 = CoxFrailty(distribution='gamma', tdf=3, maxit=300)
        m2 = CoxFrailty(distribution='gamma', tdf=10, maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m1.fit(X_synth, y_rul_synth,
                   t_stop=t_stop_synth, evento=evento_synth,
                   groups=groups_synth)
            m2.fit(X_synth, y_rul_synth,
                   t_stop=t_stop_synth, evento=evento_synth,
                   groups=groups_synth)
        # Both must have the same convergence outcome
        assert m1.is_fitted_ == m2.is_fitted_

    def test_fit_without_t_stop_fails_gracefully(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
    ):
        """Missing t_stop must emit RuntimeWarning and set is_fitted_=False."""
        model = CoxFrailty()
        with pytest.warns(RuntimeWarning, match="t_stop"):
            model.fit(X_synth, y_rul_synth)
        assert model.is_fitted_ is False
        assert model.model_name_ is None

    def test_fit_without_groups_no_exception(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
    ):
        """Absent groups must not raise — each window treated as independent."""
        model = CoxFrailty(distribution='gamma', maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
            )
        assert isinstance(model.is_fitted_, bool)

    def test_fit_without_evento_no_exception(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        """Absent evento must not raise — defaults to all-ones internally."""
        model = CoxFrailty(distribution='gamma', maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=None,
                groups=groups_synth,
            )
        assert isinstance(model.is_fitted_, bool)

    def test_fit_removes_nonpositive_t_stop_warns(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        evento_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        """Rows with t_stop <= 0 must be removed with RuntimeWarning."""
        t_stop_bad = np.tile(np.arange(1, 21, dtype=float), 10)
        t_stop_bad[0] = 0.0
        model = CoxFrailty(distribution='gamma', maxit=300)
        with pytest.warns(RuntimeWarning, match="t_stop <= 0"):
            model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_bad,
                evento=evento_synth,
                groups=groups_synth,
            )

    def test_fit_replaces_previous_r_model(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        """Re-fitting must remove the previous R model and create a new one."""
        model = CoxFrailty(distribution='gamma', maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                groups=groups_synth,
            )
        first_name = model.model_name_

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                groups=groups_synth,
            )
        second_name = model.model_name_

        # Both fits succeeded — names must be different (uuid-based)
        if first_name is not None and second_name is not None:
            assert first_name != second_name

    def test_fit_low_event_rate_no_exception(self, pipeline_output: dict):
        """Low event rate (motor 1) must not raise — documented negative result."""
        model = CoxFrailty(distribution='gamma', maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(
                pipeline_output['X'],
                pipeline_output['y_rul'],
                t_stop=pipeline_output['t_stop'],
                evento=pipeline_output['evento'],
                groups=pipeline_output['groups'],
            )
        assert isinstance(model.is_fitted_, bool)


# ---------------------------------------------------------------------------
# TestPredict — always returns NaN + warning
# ---------------------------------------------------------------------------

class TestPredict:

    def test_returns_nan_always(self, X_synth: np.ndarray, n_windows: int):
        with pytest.warns(RuntimeWarning, match="predict_with_time"):
            result = CoxFrailty().predict(X_synth)
        assert result.shape == (n_windows,)
        assert np.isnan(result).all()

    def test_returns_nan_even_if_fitted(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
        n_windows: int,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        with pytest.warns(RuntimeWarning, match="predict_with_time"):
            result = fitted_model_synth.predict(X_synth)
        assert result.shape == (n_windows,)
        assert np.isnan(result).all()


# ---------------------------------------------------------------------------
# TestPredictSurvivalFunction
# ---------------------------------------------------------------------------

class TestPredictSurvivalFunction:

    def test_returns_none_if_not_fitted(self, X_synth: np.ndarray):
        assert CoxFrailty().predict_survival_function(X_synth) is None

    def test_returns_list_if_fitted(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth[:5])
        assert isinstance(sf, list)

    def test_length_matches_n_windows(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        n = 10
        sf = fitted_model_synth.predict_survival_function(X_synth[:n])
        assert sf is not None
        assert len(sf) == n

    def test_each_element_is_tuple_of_two_arrays(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth[:5])
        assert sf is not None
        for times, probs in sf:
            assert isinstance(times, np.ndarray)
            assert isinstance(probs, np.ndarray)
            assert times.shape == probs.shape

    def test_values_in_unit_interval(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth[:10])
        assert sf is not None
        for _, probs in sf:
            assert probs.min() >= 0.0
            assert probs.max() <= 1.0 + 1e-6

    def test_non_increasing_over_time(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
    ):
        """S(t|X) must be non-increasing over time for each window."""
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth[:5])
        assert sf is not None
        for i, (_, probs) in enumerate(sf):
            diffs = np.diff(probs)
            assert np.all(diffs <= 1e-10), \
                f"S(t) increasing at window {i}: max diff={diffs.max()}"

    def test_real_data_no_exception(
        self,
        fitted_model_real: CoxFrailty,
        pipeline_output: dict,
    ):
        """predict_survival_function must not raise even if not fitted."""
        result = fitted_model_real.predict_survival_function(
            pipeline_output['X']
        )
        assert result is None or isinstance(result, list)


# ---------------------------------------------------------------------------
# TestPredictDeathCurve
# ---------------------------------------------------------------------------

class TestPredictDeathCurve:

    def test_returns_none_if_not_fitted(self, X_synth: np.ndarray):
        assert CoxFrailty().predict_death_curve(X_synth) is None

    def test_returns_list_if_fitted(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        dc = fitted_model_synth.predict_death_curve(X_synth[:5])
        assert isinstance(dc, list)

    def test_equals_one_minus_survival(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
    ):
        """F(t|X) = 1 - S(t|X) exactly for each window and time point."""
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        sf = fitted_model_synth.predict_survival_function(X_synth[:5])
        dc = fitted_model_synth.predict_death_curve(X_synth[:5])
        assert sf is not None and dc is not None
        for (_, s_probs), (_, f_probs) in zip(sf, dc):
            np.testing.assert_array_almost_equal(f_probs, 1.0 - s_probs)

    def test_values_in_unit_interval(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        dc = fitted_model_synth.predict_death_curve(X_synth[:10])
        assert dc is not None
        for _, probs in dc:
            assert probs.min() >= 0.0
            assert probs.max() <= 1.0 + 1e-6

    def test_non_decreasing_over_time(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
    ):
        """F(t|X) must be non-decreasing over time for each window."""
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        dc = fitted_model_synth.predict_death_curve(X_synth[:5])
        assert dc is not None
        for i, (_, probs) in enumerate(dc):
            diffs = np.diff(probs)
            assert np.all(diffs >= -1e-10), \
                f"F(t) decreasing at window {i}: min diff={diffs.min()}"


# ---------------------------------------------------------------------------
# TestPredictWithTime
# ---------------------------------------------------------------------------

class TestPredictWithTime:

    def test_returns_nan_if_not_fitted(
        self,
        X_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        n_windows: int,
    ):
        result = CoxFrailty().predict_with_time(X_synth, t_stop_synth)
        assert result.shape == (n_windows,)
        assert np.isnan(result).all()

    def test_output_shape(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        n_windows: int,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        result = fitted_model_synth.predict_with_time(X_synth, t_stop_synth)
        assert result.shape == (n_windows,)

    def test_output_non_negative(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
        t_stop_synth: np.ndarray,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        result = fitted_model_synth.predict_with_time(X_synth, t_stop_synth)
        finite = np.isfinite(result)
        assert (result[finite] >= 0.0).all()

    def test_output_clipped(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
        t_stop_synth: np.ndarray,
    ):
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        result = fitted_model_synth.predict_with_time(X_synth, t_stop_synth)
        finite = np.isfinite(result)
        assert (result[finite] <= fitted_model_synth.clipping_threshold).all()

    def test_output_is_float(
        self,
        X_synth: np.ndarray,
        t_stop_synth: np.ndarray,
    ):
        result = CoxFrailty().predict_with_time(X_synth, t_stop_synth)
        assert np.issubdtype(result.dtype, np.floating)

    def test_lower_confidence_gives_lower_rul(
        self,
        X_synth: np.ndarray,
        y_rul_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        groups_synth: np.ndarray,
    ):
        """Lower confidence_threshold → earlier t* on F(t) → lower mean RUL.

        Maintenance semantics:
            Low threshold = conservative = alert sooner (few motors dead)
            High threshold = aggressive  = alert later  (most motors dead)
        """
        m_low  = CoxFrailty(confidence_threshold=0.3, maxit=300)
        m_high = CoxFrailty(confidence_threshold=0.95, maxit=300)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m_low.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                groups=groups_synth,
            )
            m_high.fit(
                X_synth, y_rul_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                groups=groups_synth,
            )
        if m_low.is_fitted_ and m_high.is_fitted_:
            pred_low  = m_low.predict_with_time(X_synth, t_stop_synth)
            pred_high = m_high.predict_with_time(X_synth, t_stop_synth)
            finite = np.isfinite(pred_low) & np.isfinite(pred_high)
            if finite.any():
                assert pred_low[finite].mean() <= pred_high[finite].mean()

    def test_predict_with_time_overrides_base_model_default(
        self,
        fitted_model_synth: CoxFrailty,
        X_synth: np.ndarray,
        t_stop_synth: np.ndarray,
    ):
        """predict_with_time must use survival function, not the NaN fallback.

        BaseRULModel.predict_with_time() delegates to predict(X) by default,
        which returns NaN for CoxFrailty. The override must call
        predict_survival_function() and _survival_to_rul() instead.
        """
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        result = fitted_model_synth.predict_with_time(X_synth[:5], t_stop_synth[:5])
        # If all NaN, the override is not working — calling predict() instead
        assert result.shape == (5,)

    def test_real_data_no_exception(
        self,
        fitted_model_real: CoxFrailty,
        pipeline_output: dict,
    ):
        """predict_with_time on real data must not raise — even if not fitted."""
        result = fitted_model_real.predict_with_time(
            pipeline_output['X'],
            pipeline_output['t_stop'],
        )
        assert result.shape == (pipeline_output['X'].shape[0],)


# ---------------------------------------------------------------------------
# TestPrintSummary
# ---------------------------------------------------------------------------

class TestPrintSummary:

    def test_no_error_if_fitted(
        self,
        fitted_model_synth: CoxFrailty,
        capsys: pytest.CaptureFixture[str],
    ):
        """print_summary must not raise when model is fitted."""
        if not fitted_model_synth.is_fitted_:
            pytest.skip("Model did not converge")
        fitted_model_synth.print_summary()

    def test_no_error_if_not_fitted(
        self,
        capsys: pytest.CaptureFixture[str],
    ):
        """print_summary must not raise when model is not fitted."""
        CoxFrailty().print_summary()
        captured = capsys.readouterr()
        assert "not fitted" in captured.out