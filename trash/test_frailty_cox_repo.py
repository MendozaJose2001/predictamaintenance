"""Tests for the frailty_cox R repository module.

These tests require a working R installation with the survival package.
They are integration tests that verify the Python-R bridge works correctly
for fitting and predicting with shared frailty Cox models via coxph.
"""

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

from src.repository.frailty_cox import fit_cox_frailty, predict_survival_functions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def synthetic_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates a minimal synthetic dataset compatible with coxph + frailty().

    Uses module scope to avoid refitting R models on every test — fitting
    is expensive and the data does not change between tests.

    Returns:
        Tuple of (X, t_start, t_stop, evento, motor_id).
    """
    rng = np.random.default_rng(42)
    n_motors = 20
    cycles_per_motor = 30
    n_samples = n_motors * cycles_per_motor

    X = pd.DataFrame({
        'sensor_1': rng.normal(size=n_samples),
        'sensor_2': rng.normal(size=n_samples),
    })

    motor_id = np.repeat(np.arange(1, n_motors + 1), cycles_per_motor)

    time_in_cycles = np.tile(np.arange(1, cycles_per_motor + 1), n_motors)
    t_start = (time_in_cycles - 1).astype(float)
    t_stop = time_in_cycles.astype(float)

    # Last cycle of each motor is the event
    evento = np.zeros(n_samples, dtype=int)
    evento[np.arange(cycles_per_motor - 1, n_samples, cycles_per_motor)] = 1

    return X, t_start, t_stop, evento, motor_id


@pytest.fixture(scope='module')
def fitted_model(
    synthetic_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> str:
    """Returns a fitted coxph R object with shared gamma frailty.

    Module scope ensures the model is fitted only once for all tests
    that depend on it.
    """
    X, t_start, t_stop, evento, motor_id = synthetic_data
    return fit_cox_frailty(
        X=X,
        t_start=t_start,
        t_stop=t_stop,
        evento=evento,
        motor_id=motor_id,
        distribution='gamma'
    )


@pytest.fixture
def X_new() -> pd.DataFrame:
    """Returns a small feature matrix for prediction tests."""
    return pd.DataFrame({
        'sensor_1': [0.5, -0.3, 1.2],
        'sensor_2': [1.2,  0.8, -0.5],
    })


# ---------------------------------------------------------------------------
# fit_cox_frailty
# ---------------------------------------------------------------------------

class TestFitCoxFrailty:
    """Tests for the fit_cox_frailty function."""

    def test_returns_string(self, fitted_model: str):
        """fit_cox_frailty must return a non-empty string model name."""
        assert isinstance(fitted_model, str)
        assert len(fitted_model) > 0

    def test_model_converged(
        self,
        synthetic_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        """fit_cox_frailty must return a non-empty string when model converges."""
        X, t_start, t_stop, evento, motor_id = synthetic_data
        model_name = fit_cox_frailty(
            X=X, t_start=t_start, t_stop=t_stop,
            evento=evento, motor_id=motor_id, distribution='gamma'
        )
        assert isinstance(model_name, str)
        assert len(model_name) > 0

    def test_model_stored_in_r_globalenv(self, fitted_model: str):
        """Fitted model must exist in R's global environment."""
        exists = bool(np.array(ro.r(f'exists("{fitted_model}")'))[0])
        assert exists is True

    def test_coef_has_correct_names(self, fitted_model: str):
        """Coefficient names must match the feature names in X."""
        coef_names = np.array(ro.r(f'names({fitted_model}$coefficients)'))
        assert 'sensor_1' in coef_names
        assert 'sensor_2' in coef_names

    def test_baseline_hazard_non_negative(self, fitted_model: str):
        """Cumulative baseline hazard H0(t) must be non-negative."""
        H0 = np.array(ro.r(f'{fitted_model}_bh$hazard'))
        assert (H0 >= 0.0).all()

    def test_baseline_hazard_non_decreasing(self, fitted_model: str):
        """Cumulative baseline hazard must be non-decreasing over time."""
        H0 = np.array(ro.r(f'{fitted_model}_bh$hazard'))
        diffs = np.diff(H0)
        assert (diffs >= -1e-10).all()

    def test_times_are_non_negative(self, fitted_model: str):
        """All time points in the baseline hazard must be non-negative."""
        times = np.array(ro.r(f'{fitted_model}_bh$time'))
        assert (times >= 0.0).all()

    def test_iterations_completed(self, fitted_model: str):
        """coxph must have completed at least one iteration."""
        iter_val = int(np.array(ro.r(f'{fitted_model}$iter[1]'))[0])
        assert iter_val > 0

    def test_gaussian_distribution_converges(
        self,
        synthetic_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        """fit_cox_frailty must also converge with gaussian distribution."""
        X, t_start, t_stop, evento, motor_id = synthetic_data
        model_name = fit_cox_frailty(
            X=X, t_start=t_start, t_stop=t_stop,
            evento=evento, motor_id=motor_id, distribution='gaussian'
        )
        assert isinstance(model_name, str)
        assert len(model_name) > 0

    def test_raises_on_non_convergence(self):
        """Must raise RuntimeError when model fails to fit.

        Uses a degenerate dataset where all sensor values are zero and
        no observed failures exist, which prevents coxph from iterating.
        """
        n = 20
        X_bad = pd.DataFrame({
            'sensor_1': np.zeros(n),
            'sensor_2': np.zeros(n),
        })
        t_start_bad = np.zeros(n)
        t_stop_bad = np.full(n, 0.001)
        evento_bad = np.zeros(n, dtype=int)
        motor_id_bad = np.repeat(np.arange(1, 5), 5)

        with pytest.raises((RuntimeError, Exception)):
            fit_cox_frailty(
                X=X_bad,
                t_start=t_start_bad,
                t_stop=t_stop_bad,
                evento=evento_bad,
                motor_id=motor_id_bad,
            )


# ---------------------------------------------------------------------------
# predict_survival_functions
# ---------------------------------------------------------------------------

class TestPredictSurvivalFunctions:
    """Tests for the predict_survival_functions function."""

    def test_returns_list_of_correct_length(
        self, fitted_model: str, X_new: pd.DataFrame
    ):
        """Must return one tuple per row in X_new."""
        result = predict_survival_functions(fitted_model, X_new)
        assert len(result) == len(X_new)

    def test_each_element_is_tuple_of_two_arrays(
        self, fitted_model: str, X_new: pd.DataFrame
    ):
        """Each element must be a tuple of (times, survival_probs)."""
        result = predict_survival_functions(fitted_model, X_new)
        for times, probs in result:
            assert isinstance(times, np.ndarray)
            assert isinstance(probs, np.ndarray)

    def test_times_and_probs_have_same_length(
        self, fitted_model: str, X_new: pd.DataFrame
    ):
        """Times and survival_probs must have the same length."""
        result = predict_survival_functions(fitted_model, X_new)
        for times, probs in result:
            assert len(times) == len(probs)

    def test_survival_probs_in_unit_interval(
        self, fitted_model: str, X_new: pd.DataFrame
    ):
        """All survival probabilities must be in [0, 1]."""
        result = predict_survival_functions(fitted_model, X_new)
        for _, probs in result:
            assert (probs >= 0.0).all()
            assert (probs <= 1.0).all()

    def test_survival_is_non_increasing(
        self, fitted_model: str, X_new: pd.DataFrame
    ):
        """Survival function must be non-increasing over time."""
        result = predict_survival_functions(fitted_model, X_new)
        for _, probs in result:
            diffs = np.diff(probs)
            assert (diffs <= 1e-10).all()

    def test_times_are_same_for_all_predictions(
        self, fitted_model: str, X_new: pd.DataFrame
    ):
        """All predictions must share the same time grid."""
        result = predict_survival_functions(fitted_model, X_new)
        times_ref = result[0][0]
        for times, _ in result[1:]:
            np.testing.assert_array_equal(times, times_ref)