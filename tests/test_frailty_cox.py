"""Integration tests for src/repository/frailty_cox.py.

These tests require a working R installation with the survival package.
They verify the full Python→R→Python roundtrip: data conversion, model
fitting, baseline hazard extraction, survival function prediction, and
R globalenv cleanup.

Test structure:
    TestRConnection     — R and survival package are accessible
    TestBuildRDataframe — _build_r_dataframe() produces correct R objects
    TestFitCoxFrailty   — fit_cox_frailty() fits and returns a model name
    TestPredictSurvival — predict_survival_functions() returns valid S(t|X)
    TestRemoveModel     — remove_model() cleans up R globalenv correctly
    TestFullRoundtrip   — fit → predict → remove cycle is clean

Synthetic data design:
    10 motors × 20 windows = 200 windows total, 10 events (5% event rate).
    t_stop = cycles 1..20 repeated for each motor.
    evento = 1 only at the last window of each motor.
    Features = 5 PCA-like components (standardized normal).

    This replicates the GGS fold structure at minimal scale — enough events
    for coxph to converge reliably with frailty(), while keeping test runtime
    under 30 seconds.

rpy2 type helper pattern:
    rpy2's ro.r() returns a generic R object with no Python type hints.
    Direct indexing (result[0]) and iteration cause Pyright errors.
    All R interactions are encapsulated in typed helper functions at the
    module level (_r_bool, _r_int, _r_str, _r_list, _r_array) that cast
    R return values to plain Python/numpy types. Tests call only these
    helpers — never ro.r() directly with indexing.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import BoolVector, FloatVector, IntVector, StrVector

from src.repository.frailty_cox import (
    _build_r_dataframe,
    fit_cox_frailty,
    predict_survival_functions,
    remove_model,
)


# ---------------------------------------------------------------------------
# rpy2 typed helpers — encapsulate all R object access
# ---------------------------------------------------------------------------

def _r_bool(expr: str) -> bool:
    """Evaluates an R expression and returns a Python bool."""
    result: BoolVector = ro.r(expr)  # type: ignore[assignment]
    return bool(result[0])


def _r_int(expr: str) -> int:
    """Evaluates an R expression and returns a Python int."""
    result: IntVector = ro.r(expr)  # type: ignore[assignment]
    return int(result[0])


def _r_float(expr: str) -> float:
    """Evaluates an R expression and returns a Python float."""
    result: FloatVector = ro.r(expr)  # type: ignore[assignment]
    return float(result[0])


def _r_str(expr: str) -> str:
    """Evaluates an R expression and returns a Python str."""
    result: StrVector = ro.r(expr)  # type: ignore[assignment]
    return str(result[0])


def _r_list(expr: str) -> list[str]:
    """Evaluates an R expression and returns a Python list of strings."""
    result: StrVector = ro.r(expr)  # type: ignore[assignment]
    return [str(x) for x in result]


def _r_array(expr: str) -> np.ndarray:
    """Evaluates an R expression and returns a numpy array."""
    result: FloatVector = ro.r(expr)  # type: ignore[assignment]
    return np.array(result)


def _r_exec(expr: str) -> None:
    """Executes an R expression with no return value."""
    ro.r(expr)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def n_motors() -> int:
    return 10


@pytest.fixture(scope='module')
def n_windows_per_motor() -> int:
    return 20


@pytest.fixture(scope='module')
def n_components() -> int:
    return 5


@pytest.fixture(scope='module')
def X_synth(n_motors: int, n_windows_per_motor: int, n_components: int) -> np.ndarray:
    """Synthetic PCA-like feature matrix — standardized normal."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(n_motors * n_windows_per_motor, n_components))


@pytest.fixture(scope='module')
def t_stop_synth(n_motors: int, n_windows_per_motor: int) -> np.ndarray:
    """t_stop — cycles 1..20 repeated for each motor."""
    return np.tile(np.arange(1, n_windows_per_motor + 1, dtype=float), n_motors)


@pytest.fixture(scope='module')
def t_start_synth(t_stop_synth: np.ndarray) -> np.ndarray:
    """t_start = t_stop - 1 (Andersen-Gill counting process format)."""
    return t_stop_synth - 1.0


@pytest.fixture(scope='module')
def evento_synth(n_motors: int, n_windows_per_motor: int) -> np.ndarray:
    """evento=1 at the last window of each motor."""
    n_total = n_motors * n_windows_per_motor
    evento = np.zeros(n_total, dtype=int)
    for i in range(n_motors):
        evento[(i + 1) * n_windows_per_motor - 1] = 1
    return evento


@pytest.fixture(scope='module')
def motor_id_synth(n_motors: int, n_windows_per_motor: int) -> np.ndarray:
    """Motor ID per window — 0..9 each repeated 20 times."""
    return np.repeat(np.arange(n_motors, dtype=int), n_windows_per_motor)


@pytest.fixture(scope='module')
def X_df_synth(X_synth: np.ndarray, n_components: int) -> pd.DataFrame:
    """X as DataFrame with PC_1..PC_n column names."""
    return pd.DataFrame(
        X_synth,
        columns=[f'PC_{i+1}' for i in range(n_components)],
    )


@pytest.fixture(scope='module')
def fitted_model_name(
    X_df_synth: pd.DataFrame,
    t_start_synth: np.ndarray,
    t_stop_synth: np.ndarray,
    evento_synth: np.ndarray,
    motor_id_synth: np.ndarray,
) -> str:
    """Fits a CoxFrailty model on synthetic data and returns the model name.

    Module-scoped — fits only once per test session. Shared across all
    tests that need a fitted model.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_name = fit_cox_frailty(
            X=X_df_synth,
            t_start=t_start_synth,
            t_stop=t_stop_synth,
            evento=evento_synth,
            motor_id=motor_id_synth,
            distribution='gamma',
            maxit=300,
        )
    return model_name


# ---------------------------------------------------------------------------
# TestRConnection — verify R and survival are accessible
# ---------------------------------------------------------------------------

class TestRConnection:

    def test_r_is_accessible(self):
        """R must be importable and executable."""
        result = _r_float('1 + 1')
        assert result == 2.0

    def test_r_version_is_4_or_higher(self):
        """R version must be 4.x or higher."""
        version_str = _r_str('R.version.string')
        major = int(version_str.split()[2].split('.')[0])
        assert major >= 4, f"R version too old: {version_str}"

    def test_survival_package_loads(self):
        """survival package must load without errors."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            survival = importr('survival')
        assert survival.__rname__ == 'survival'

    def test_coxph_callable_from_python(self):
        """survival::coxph must be callable via ro.r()."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = _r_bool('exists("coxph", mode="function")')
        assert result is True

    def test_frailty_cox_imports(self):
        """frailty_cox module must import without errors."""
        assert callable(fit_cox_frailty)
        assert callable(predict_survival_functions)
        assert callable(remove_model)
        assert callable(_build_r_dataframe)


# ---------------------------------------------------------------------------
# TestBuildRDataframe — _build_r_dataframe() correctness
# ---------------------------------------------------------------------------

class TestBuildRDataframe:

    def test_returns_string(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """Must return a plain Python str (R variable name)."""
        df_name = _build_r_dataframe(
            X_df_synth, t_start_synth, t_stop_synth,
            evento_synth, motor_id_synth,
        )
        assert isinstance(df_name, str)
        _r_exec(f'rm({df_name})')

    def test_dataframe_exists_in_r_globalenv(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """The returned name must refer to an existing R object."""
        df_name = _build_r_dataframe(
            X_df_synth, t_start_synth, t_stop_synth,
            evento_synth, motor_id_synth,
        )
        assert _r_bool(f'exists("{df_name}")') is True
        _r_exec(f'rm({df_name})')

    def test_dataframe_nrow_matches_input(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """R dataframe must have same number of rows as input."""
        df_name = _build_r_dataframe(
            X_df_synth, t_start_synth, t_stop_synth,
            evento_synth, motor_id_synth,
        )
        assert _r_int(f'nrow({df_name})') == len(X_df_synth)
        _r_exec(f'rm({df_name})')

    def test_dataframe_has_required_columns(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """R dataframe must have t_start, t_stop, evento, motor_id columns."""
        df_name = _build_r_dataframe(
            X_df_synth, t_start_synth, t_stop_synth,
            evento_synth, motor_id_synth,
        )
        col_names = _r_list(f'names({df_name})')
        for required in ['t_start', 't_stop', 'evento', 'motor_id']:
            assert required in col_names, f"Missing column: {required}"
        _r_exec(f'rm({df_name})')

    def test_dataframe_has_feature_columns(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
        n_components: int,
    ):
        """R dataframe must contain all PC feature columns."""
        df_name = _build_r_dataframe(
            X_df_synth, t_start_synth, t_stop_synth,
            evento_synth, motor_id_synth,
        )
        col_names = _r_list(f'names({df_name})')
        for i in range(1, n_components + 1):
            assert f'PC_{i}' in col_names, f"Missing feature column: PC_{i}"
        _r_exec(f'rm({df_name})')

    def test_t_start_less_than_t_stop_in_r(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """All rows must satisfy t_start < t_stop in R — required by coxph."""
        df_name = _build_r_dataframe(
            X_df_synth, t_start_synth, t_stop_synth,
            evento_synth, motor_id_synth,
        )
        assert _r_bool(f'all({df_name}$t_start < {df_name}$t_stop)') is True
        _r_exec(f'rm({df_name})')

    def test_csv_file_removed_after_load(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Temporary CSV must be deleted from disk after loading into R."""
        import os
        import tempfile

        created_files: list[str] = []
        original_mktemp = tempfile.mktemp

        def tracking_mktemp(
            suffix: str = '',
            prefix: str = 'tmp',
            dir: str | None = None,
        ) -> str:
            path: str = original_mktemp(suffix=suffix, prefix=prefix, dir=dir)
            created_files.append(path)
            return path

        monkeypatch.setattr(tempfile, 'mktemp', tracking_mktemp)
        df_name = _build_r_dataframe(
            X_df_synth, t_start_synth, t_stop_synth,
            evento_synth, motor_id_synth,
        )
        for path in created_files:
            assert not os.path.exists(path), f"Temp CSV not deleted: {path}"
        _r_exec(f'rm({df_name})')


# ---------------------------------------------------------------------------
# TestFitCoxFrailty — fit_cox_frailty() correctness
# ---------------------------------------------------------------------------

class TestFitCoxFrailty:

    def test_returns_string(self, fitted_model_name: str):
        """fit_cox_frailty must return a plain Python str."""
        assert isinstance(fitted_model_name, str)

    def test_model_exists_in_r_globalenv(self, fitted_model_name: str):
        """Model name must exist as an object in R globalenv."""
        assert _r_bool(f'exists("{fitted_model_name}")') is True

    def test_baseline_hazard_exists_in_r_globalenv(self, fitted_model_name: str):
        """Pre-computed baseline hazard must be stored in R globalenv."""
        bh_name = f'{fitted_model_name}_bh'
        assert _r_bool(f'exists("{bh_name}")') is True

    def test_model_is_coxph_class(self, fitted_model_name: str):
        """Fitted R object must be of class 'coxph'."""
        cls = _r_list(f'class({fitted_model_name})')
        assert 'coxph' in cls, f"Expected coxph, got: {cls}"

    def test_model_completed_iterations(self, fitted_model_name: str):
        """coxph must complete at least one iteration."""
        n_iter = _r_int(f'{fitted_model_name}$iter[1]')
        assert n_iter > 0, f"No iterations completed: iter={n_iter}"

    def test_coefficients_are_finite(self, fitted_model_name: str):
        """Model coefficients must be finite — no divergence."""
        coefs = _r_array(f'{fitted_model_name}$coefficients')
        assert np.all(np.isfinite(coefs)), \
            f"Non-finite coefficients: {coefs}"

    def test_baseline_hazard_is_non_negative(self, fitted_model_name: str):
        """Cumulative baseline hazard H0(t) must be non-negative."""
        cum_hazard = _r_array(f'{fitted_model_name}_bh$hazard')
        assert np.all(cum_hazard >= 0), \
            f"Negative baseline hazard: min={cum_hazard.min()}"

    def test_baseline_hazard_is_non_decreasing(self, fitted_model_name: str):
        """H0(t) must be non-decreasing — cumulative hazard property."""
        cum_hazard = _r_array(f'{fitted_model_name}_bh$hazard')
        diffs = np.diff(cum_hazard)
        assert np.all(diffs >= -1e-10), \
            f"Baseline hazard is decreasing: min diff={diffs.min()}"

    def test_fit_with_gaussian_distribution(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """Gaussian frailty must also fit without errors."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_name = fit_cox_frailty(
                X=X_df_synth,
                t_start=t_start_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                motor_id=motor_id_synth,
                distribution='gaussian',
                maxit=300,
            )
        assert isinstance(model_name, str)
        assert _r_bool(f'exists("{model_name}")') is True
        remove_model(model_name)

    def test_fit_with_maxit_1_does_not_crash(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """maxit=1 must not produce unhandled exceptions — may raise RuntimeError."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model_name = fit_cox_frailty(
                    X=X_df_synth,
                    t_start=t_start_synth,
                    t_stop=t_stop_synth,
                    evento=evento_synth,
                    motor_id=motor_id_synth,
                    distribution='gamma',
                    maxit=1,
                )
            remove_model(model_name)
        except RuntimeError:
            pass  # Expected — zero iterations with maxit=1


# ---------------------------------------------------------------------------
# TestPredictSurvival — predict_survival_functions() correctness
# ---------------------------------------------------------------------------

class TestPredictSurvival:

    def test_returns_list(
        self, fitted_model_name: str, X_df_synth: pd.DataFrame,
    ):
        """Must return a list."""
        result = predict_survival_functions(fitted_model_name, X_df_synth[:5])
        assert isinstance(result, list)

    def test_length_matches_n_windows(
        self, fitted_model_name: str, X_df_synth: pd.DataFrame,
    ):
        """Must return one tuple per input window."""
        n = 10
        result = predict_survival_functions(fitted_model_name, X_df_synth[:n])
        assert len(result) == n

    def test_each_element_is_tuple_of_two_arrays(
        self, fitted_model_name: str, X_df_synth: pd.DataFrame,
    ):
        """Each element must be (times, probs) — two numpy arrays."""
        result = predict_survival_functions(fitted_model_name, X_df_synth[:5])
        for times, probs in result:
            assert isinstance(times, np.ndarray), "times must be ndarray"
            assert isinstance(probs, np.ndarray), "probs must be ndarray"
            assert times.shape == probs.shape, \
                "times and probs must have same shape"

    def test_survival_probs_in_unit_interval(
        self, fitted_model_name: str, X_df_synth: pd.DataFrame,
    ):
        """S(t|X) must be in [0, 1] for all windows and time points."""
        result = predict_survival_functions(fitted_model_name, X_df_synth[:10])
        for times, probs in result:
            assert probs.min() >= 0.0, f"S(t) < 0: min={probs.min()}"
            assert probs.max() <= 1.0 + 1e-6, f"S(t) > 1: max={probs.max()}"

    def test_survival_function_non_increasing(
        self, fitted_model_name: str, X_df_synth: pd.DataFrame,
    ):
        """S(t|X) must be non-increasing over time for each window."""
        result = predict_survival_functions(fitted_model_name, X_df_synth[:5])
        for i, (times, probs) in enumerate(result):
            diffs = np.diff(probs)
            assert np.all(diffs <= 1e-10), \
                f"S(t) is increasing at window {i}: max diff={diffs.max()}"

    def test_times_are_positive(
        self, fitted_model_name: str, X_df_synth: pd.DataFrame,
    ):
        """Time points must be positive — absolute cycle counts."""
        result = predict_survival_functions(fitted_model_name, X_df_synth[:5])
        for times, _ in result:
            assert np.all(times > 0), \
                f"Non-positive time point: min={times.min()}"

    def test_times_are_non_decreasing(
        self, fitted_model_name: str, X_df_synth: pd.DataFrame,
    ):
        """Time axis must be non-decreasing."""
        result = predict_survival_functions(fitted_model_name, X_df_synth[:5])
        for times, _ in result:
            diffs = np.diff(times)
            assert np.all(diffs >= 0), \
                f"Time axis is decreasing: min diff={diffs.min()}"

    def test_full_dataset_prediction(
        self, fitted_model_name: str, X_df_synth: pd.DataFrame,
    ):
        """predict_survival_functions must handle the full training dataset."""
        result = predict_survival_functions(fitted_model_name, X_df_synth)
        assert len(result) == len(X_df_synth)


# ---------------------------------------------------------------------------
# TestRemoveModel — remove_model() correctness
# ---------------------------------------------------------------------------

class TestRemoveModel:

    def test_removes_model_from_r_globalenv(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """After remove_model, the model must not exist in R globalenv."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_name = fit_cox_frailty(
                X=X_df_synth,
                t_start=t_start_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                motor_id=motor_id_synth,
                distribution='gamma',
                maxit=300,
            )
        remove_model(model_name)
        assert _r_bool(f'exists("{model_name}")') is False

    def test_removes_baseline_hazard_from_r_globalenv(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """After remove_model, the _bh object must not exist in R globalenv."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_name = fit_cox_frailty(
                X=X_df_synth,
                t_start=t_start_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                motor_id=motor_id_synth,
                distribution='gamma',
                maxit=300,
            )
        bh_name = f'{model_name}_bh'
        remove_model(model_name)
        assert _r_bool(f'exists("{bh_name}")') is False

    def test_remove_nonexistent_model_no_exception(self):
        """remove_model on a non-existent name must not raise."""
        remove_model('nonexistent_model_xyz_999')

    def test_remove_cleans_up_r_memory(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """Fitting and removing multiple models must not leak R objects."""
        names: list[str] = []
        for _ in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                name = fit_cox_frailty(
                    X=X_df_synth,
                    t_start=t_start_synth,
                    t_stop=t_stop_synth,
                    evento=evento_synth,
                    motor_id=motor_id_synth,
                    distribution='gamma',
                    maxit=300,
                )
            names.append(name)

        for name in names:
            remove_model(name)

        for name in names:
            assert _r_bool(f'exists("{name}")') is False
            assert _r_bool(f'exists("{name}_bh")') is False


# ---------------------------------------------------------------------------
# TestFullRoundtrip — fit → predict → remove cycle
# ---------------------------------------------------------------------------

class TestFullRoundtrip:

    def test_fit_predict_remove_cycle(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """Full fit → predict → remove cycle must complete without errors."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_name = fit_cox_frailty(
                X=X_df_synth,
                t_start=t_start_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                motor_id=motor_id_synth,
                distribution='gamma',
                maxit=300,
            )

        result = predict_survival_functions(model_name, X_df_synth[:5])
        assert len(result) == 5

        remove_model(model_name)
        assert _r_bool(f'exists("{model_name}")') is False

    def test_refit_produces_unique_model_name(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """Fitting twice must produce unique model names (uuid-based)."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            name1 = fit_cox_frailty(
                X=X_df_synth,
                t_start=t_start_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                motor_id=motor_id_synth,
            )
        remove_model(name1)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            name2 = fit_cox_frailty(
                X=X_df_synth,
                t_start=t_start_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                motor_id=motor_id_synth,
            )

        assert name1 != name2, "Model names must be unique (uuid-based)"
        assert _r_bool(f'exists("{name2}")') is True
        remove_model(name2)

    def test_survival_to_rul_via_repository(
        self,
        X_df_synth: pd.DataFrame,
        t_start_synth: np.ndarray,
        t_stop_synth: np.ndarray,
        evento_synth: np.ndarray,
        motor_id_synth: np.ndarray,
    ):
        """Full roundtrip must produce finite non-negative RUL estimates."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_name = fit_cox_frailty(
                X=X_df_synth,
                t_start=t_start_synth,
                t_stop=t_stop_synth,
                evento=evento_synth,
                motor_id=motor_id_synth,
            )

        sf = predict_survival_functions(model_name, X_df_synth[:10])
        t_stop_val = t_stop_synth[:10]

        confidence_threshold = 0.5
        clipping_threshold   = 125
        rul_list: list[float] = []

        for i, (times, probs) in enumerate(sf):
            failure_probs = 1.0 - probs
            exceeds = np.where(failure_probs >= confidence_threshold)[0]
            t_failure = (
                float(times[exceeds[0]])
                if len(exceeds) > 0
                else float(times[-1])
            )
            rul = max(t_failure - float(t_stop_val[i]), 0.0)
            rul_list.append(min(rul, clipping_threshold))

        rul_arr = np.array(rul_list)
        assert np.all(np.isfinite(rul_arr)), "RUL contains non-finite values"
        assert np.all(rul_arr >= 0),         "RUL contains negative values"
        assert np.all(rul_arr <= clipping_threshold), \
            "RUL exceeds clipping_threshold"

        remove_model(model_name)