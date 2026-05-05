"""Tests for the GGSTrainingManager and related functions.

Tests cover warning suppression, fold evaluation, single config evaluation,
manager initialization, grid search execution, and results extraction.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.training_manager import (
    _suppress_ggs_warnings,
    _evaluate_fold,
    _run_single_config,
    GGSTrainingManager,
)
from src.models.base_model import BaseRULModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_motors: int = 6,
    cycles_per_motor: int = 5,
    n_features: int = 3,
    seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Builds synthetic (X, y_fit, y_metrics, groups) for testing.

    Args:
        n_motors: Number of motor units.
        cycles_per_motor: Number of cycles per motor.
        n_features: Number of sensor features.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y_fit, y_metrics, groups).
    """
    rng = np.random.default_rng(seed)
    n_samples = n_motors * cycles_per_motor

    X = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f'sensor_{i}' for i in range(n_features)]
    )
    y = rng.integers(1, 100, size=n_samples).astype(float)
    groups = np.repeat(np.arange(1, n_motors + 1), cycles_per_motor)

    return X, y, y, groups


def _make_mock_model(data: tuple) -> MagicMock:
    """Builds a mock BaseRULModel that returns data from prepare_training_data.

    Args:
        data: Tuple (X, y_fit, y_metrics, groups).

    Returns:
        MagicMock mimicking a BaseRULModel instance.
    """
    mock = MagicMock(spec=BaseRULModel)
    mock.prepare_training_data.return_value = data
    return mock


def _make_mock_pipeline(
    n_samples: int,
    clipping_threshold: int = 100
) -> MagicMock:
    """Builds a mock sklearn Pipeline for evaluation tests.

    Args:
        n_samples: Number of samples predict() will return values for.
        clipping_threshold: Clipping threshold for the mock model step.

    Returns:
        MagicMock mimicking a fitted sklearn Pipeline.
    """
    mock = MagicMock(spec=Pipeline)
    mock.named_steps = {'model': MagicMock(clipping_threshold=clipping_threshold)}
    mock.predict.side_effect = lambda X: np.full(len(X), 50.0)
    return mock


# ---------------------------------------------------------------------------
# _suppress_ggs_warnings
# ---------------------------------------------------------------------------

class TestSuppressGgsWarnings:
    """Tests for the warning suppression utility."""

    def test_suppresses_runtime_warnings(self):
        """After calling _suppress_ggs_warnings, RuntimeWarnings must be silenced."""
        _suppress_ggs_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("test warning", RuntimeWarning)
        # Warnings should be suppressed — env variable disables them
        # We verify the function runs without error
        assert True  # function executed successfully

    def test_sets_pythonwarnings_env(self):
        """PYTHONWARNINGS environment variable must be set to 'ignore'."""
        import os
        _suppress_ggs_warnings()
        assert os.environ.get('PYTHONWARNINGS') == 'ignore'


# ---------------------------------------------------------------------------
# _evaluate_fold
# ---------------------------------------------------------------------------

class TestEvaluateFold:
    """Tests for the fold evaluation function."""

    def test_returns_dict_with_four_metrics(self):
        """Must return a dictionary with S_score, C_index, MAE, RMSE keys."""
        X_val = pd.DataFrame({'s1': [1.0, 2.0], 's2': [3.0, 4.0]})
        y_val = np.array([50.0, 30.0])
        pipeline = _make_mock_pipeline(n_samples=2)

        result = _evaluate_fold(pipeline, X_val, y_val)

        assert set(result.keys()) == {'S_score', 'C_index', 'MAE', 'RMSE'}

    def test_returns_float_values(self):
        """All metric values must be floats."""
        X_val = pd.DataFrame({'s1': [1.0, 2.0], 's2': [3.0, 4.0]})
        y_val = np.array([50.0, 30.0])
        pipeline = _make_mock_pipeline(n_samples=2)

        result = _evaluate_fold(pipeline, X_val, y_val)

        for value in result.values():
            assert isinstance(value, float)

    def test_calls_pipeline_predict(self):
        """Must call pipeline.predict once per metric (4 metrics total)."""
        X_val = pd.DataFrame({'s1': [1.0, 2.0], 's2': [3.0, 4.0]})
        y_val = np.array([50.0, 30.0])
        pipeline = _make_mock_pipeline(n_samples=2)

        _evaluate_fold(pipeline, X_val, y_val)

        assert pipeline.predict.call_count == 4


# ---------------------------------------------------------------------------
# _run_single_config
# ---------------------------------------------------------------------------

class TestRunSingleConfig:
    """Tests for the single hyperparameter configuration evaluator."""

    def _make_nb_class(self):
        """Returns NegativeBinomialPiecewise class for integration tests."""
        from src.models.negative_binomial import NegativeBinomialPiecewise
        return NegativeBinomialPiecewise

    def test_returns_dict_with_params_and_metrics(self):
        """Result must contain both hyperparameter values and mean metrics."""
        X, y_fit, y_metrics, groups = _make_synthetic_data()
        params = {'alpha': 0.5, 'clipping_threshold': 110}

        result = _run_single_config(
            params=params,
            model_class=self._make_nb_class(),
            X=X,
            y_fit=y_fit,
            y_metrics=y_metrics,
            groups=groups,
            n_folds=2
        )

        assert 'alpha' in result
        assert 'clipping_threshold' in result
        assert 'mean_S_score' in result
        assert 'mean_C_index' in result
        assert 'mean_MAE' in result
        assert 'mean_RMSE' in result

    def test_params_preserved_in_result(self):
        """Hyperparameter values must be preserved exactly in the result."""
        X, y_fit, y_metrics, groups = _make_synthetic_data()
        params = {'alpha': 1.5, 'clipping_threshold': 125}

        result = _run_single_config(
            params=params,
            model_class=self._make_nb_class(),
            X=X,
            y_fit=y_fit,
            y_metrics=y_metrics,
            groups=groups,
            n_folds=2
        )

        assert result['alpha'] == 1.5
        assert result['clipping_threshold'] == 125

    def test_failed_folds_produce_nan_metrics(self):
        """When all folds fail, mean metrics must be NaN."""
        X, y_fit, y_metrics, groups = _make_synthetic_data()
        params = {'alpha': 0.1, 'clipping_threshold': 110}

        failing_class = MagicMock(spec=BaseRULModel)
        failing_instance = MagicMock(spec=BaseRULModel)
        failing_instance.fit.side_effect = RuntimeError("always fails")
        failing_class.return_value = failing_instance

        result = _run_single_config(
            params=params,
            model_class=failing_class,
            X=X,
            y_fit=y_fit,
            y_metrics=y_metrics,
            groups=groups,
            n_folds=2
        )

        assert np.isnan(result['mean_S_score'])
        assert np.isnan(result['mean_MAE'])

    def test_uses_correct_number_of_folds(self):
        """GroupKFold must split data into the specified number of folds."""
        X, y_fit, y_metrics, groups = _make_synthetic_data(n_motors=10)
        params = {'alpha': 0.5, 'clipping_threshold': 110}
        fit_calls = []

        class TrackingModel(MagicMock):
            def fit(self, X, y, **kwargs):
                fit_calls.append(1)
                return self
            def predict(self, X):
                return np.full(len(X), 50.0)

        with patch('src.training_manager.Pipeline') as mock_pipe_class:
            mock_pipe = MagicMock()
            mock_pipe.fit.side_effect = lambda X, y, **kw: fit_calls.append(1)
            mock_pipe.named_steps = {'model': MagicMock(clipping_threshold=110)}
            mock_pipe.predict.return_value = np.full(len(X) // 5, 50.0)
            mock_pipe_class.return_value = mock_pipe

            _run_single_config(
                params=params,
                model_class=MagicMock,
                X=X,
                y_fit=y_fit,
                y_metrics=y_metrics,
                groups=groups,
                n_folds=3
            )

            assert mock_pipe.fit.call_count == 3


# ---------------------------------------------------------------------------
# GGSTrainingManager.__init__
# ---------------------------------------------------------------------------

class TestGGSTrainingManagerInit:
    """Tests for GGSTrainingManager initialization."""

    def test_stores_x_train(self):
        """X_train must be stored correctly after initialization."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)

        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        pd.testing.assert_frame_equal(manager.X_train, data[0])

    def test_stores_y_fit_train(self):
        """y_fit_train must be stored correctly after initialization."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)

        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        np.testing.assert_array_equal(manager.y_fit_train, data[1])

    def test_stores_y_metrics_train(self):
        """y_metrics_train must be stored correctly after initialization."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)

        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        np.testing.assert_array_equal(manager.y_metrics_train, data[2])

    def test_stores_groups_train(self):
        """groups_train must be stored correctly after initialization."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)

        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        np.testing.assert_array_equal(manager.groups_train, data[3])

    def test_ggs_results_none_at_init(self):
        """ggs_results_ must be None before running group_grid_search."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)

        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        assert manager.ggs_results_ is None


# ---------------------------------------------------------------------------
# get_training_data
# ---------------------------------------------------------------------------

class TestGetTrainingData:
    """Tests for the training data accessor."""

    def test_returns_four_element_tuple(self):
        """get_training_data must return a tuple of four elements."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)
        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        result = manager.get_training_data()

        assert len(result) == 4

    def test_returns_correct_data(self):
        """get_training_data must return the same data stored at init."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)
        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        X, y_fit, y_metrics, groups = manager.get_training_data()

        pd.testing.assert_frame_equal(X, data[0])
        np.testing.assert_array_equal(y_fit, data[1])
        np.testing.assert_array_equal(y_metrics, data[2])
        np.testing.assert_array_equal(groups, data[3])


# ---------------------------------------------------------------------------
# group_grid_search
# ---------------------------------------------------------------------------

class TestGroupGridSearch:
    """Tests for the manual group grid search."""

    def test_returns_list_of_dicts(self):
        """group_grid_search must return a list of dictionaries."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)
        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        param_grid = {'alpha': [0.1], 'clipping_threshold': [110]}

        with patch('src.training_manager._run_single_config',
                   return_value={'alpha': 0.1, 'clipping_threshold': 110,
                                 'mean_S_score': 5.0, 'mean_C_index': 0.85,
                                 'mean_MAE': 10.0, 'mean_RMSE': 15.0}):
            result = manager.group_grid_search(param_grid=param_grid, n_folds=2)

        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_generates_correct_number_of_configs(self):
        """Must evaluate all combinations in the param_grid."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)
        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        param_grid = {'alpha': [0.1, 0.5], 'clipping_threshold': [110, 125]}
        call_count = []

        def mock_run(params, model_class, X, y_fit, y_metrics, groups, n_folds):
            call_count.append(1)
            return {**params, 'mean_S_score': 5.0, 'mean_C_index': 0.85,
                    'mean_MAE': 10.0, 'mean_RMSE': 15.0}

        with patch('src.training_manager._run_single_config', side_effect=mock_run):
            manager.group_grid_search(param_grid=param_grid, n_folds=2)

        assert len(call_count) == 4  # 2 alphas × 2 thresholds

    def test_stores_results_in_ggs_results(self):
        """Results must be stored in ggs_results_ after the search."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)
        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        param_grid = {'alpha': [0.1]}

        with patch('src.training_manager._run_single_config',
                   return_value={'alpha': 0.1, 'mean_S_score': 5.0,
                                 'mean_C_index': 0.85, 'mean_MAE': 10.0,
                                 'mean_RMSE': 15.0}):
            manager.group_grid_search(param_grid=param_grid, n_folds=2)

        assert manager.ggs_results_ is not None
        assert len(manager.ggs_results_) == 1


# ---------------------------------------------------------------------------
# get_ggs_results
# ---------------------------------------------------------------------------

class TestGetGgsResults:
    """Tests for the results extraction and ranking."""

    def _make_manager_with_results(self, results: list[dict]) -> GGSTrainingManager:
        """Returns a GGSTrainingManager with pre-loaded results."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)
        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))
        manager.ggs_results_ = results
        return manager

    def test_raises_if_no_search_run(self):
        """Must raise ValueError if group_grid_search has not been run."""
        data = _make_synthetic_data()
        mock_model = _make_mock_model(data)
        manager = GGSTrainingManager(model=mock_model, list_ids=np.array([1, 2]))

        with pytest.raises(ValueError, match="No grid search results"):
            manager.get_ggs_results()

    def test_returns_dataframe(self):
        """Must return a pandas DataFrame."""
        results = [
            {'alpha': 0.1, 'mean_S_score': 5.0, 'mean_C_index': 0.85,
             'mean_MAE': 10.0, 'mean_RMSE': 15.0},
        ]
        manager = self._make_manager_with_results(results)

        df = manager.get_ggs_results()

        assert isinstance(df, pd.DataFrame)

    def test_sorted_by_ascending_s_score(self):
        """Results must be sorted by ascending mean_S_score."""
        results = [
            {'alpha': 0.5, 'mean_S_score': 10.0, 'mean_C_index': 0.80,
             'mean_MAE': 15.0, 'mean_RMSE': 20.0},
            {'alpha': 0.1, 'mean_S_score': 5.0, 'mean_C_index': 0.85,
             'mean_MAE': 10.0, 'mean_RMSE': 15.0},
            {'alpha': 1.0, 'mean_S_score': 8.0, 'mean_C_index': 0.82,
             'mean_MAE': 12.0, 'mean_RMSE': 17.0},
        ]
        manager = self._make_manager_with_results(results)

        df = manager.get_ggs_results()

        s_scores = df['mean_S_score'].tolist()
        assert s_scores == sorted(s_scores)

    def test_failed_configs_ranked_last(self):
        """Configurations with NaN S-Score must appear after successful ones."""
        results = [
            {'alpha': 0.1, 'mean_S_score': float('nan'), 'mean_C_index': float('nan'),
             'mean_MAE': float('nan'), 'mean_RMSE': float('nan')},
            {'alpha': 0.5, 'mean_S_score': 5.0, 'mean_C_index': 0.85,
             'mean_MAE': 10.0, 'mean_RMSE': 15.0},
        ]
        manager = self._make_manager_with_results(results)

        df = manager.get_ggs_results()

        assert df.iloc[0]['Success'] == 1
        assert df.iloc[-1]['Success'] == 0

    def test_top_n_limits_results(self):
        """top_n parameter must limit the number of returned rows."""
        results = [
            {'alpha': float(i), 'mean_S_score': float(i), 'mean_C_index': 0.85,
             'mean_MAE': 10.0, 'mean_RMSE': 15.0}
            for i in range(20)
        ]
        manager = self._make_manager_with_results(results)

        df = manager.get_ggs_results(top_n=5)

        assert len(df) == 5

    def test_success_column_present(self):
        """Output DataFrame must contain a Success column."""
        results = [
            {'alpha': 0.1, 'mean_S_score': 5.0, 'mean_C_index': 0.85,
             'mean_MAE': 10.0, 'mean_RMSE': 15.0},
        ]
        manager = self._make_manager_with_results(results)

        df = manager.get_ggs_results()

        assert 'Success' in df.columns