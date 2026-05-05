"""Tests for the TestManager class.

Tests cover initialization, last-index extraction, scenario evaluation,
and the full four-scenario evaluation pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.test_manager import TestManager
from src.models.base_model import BaseRULModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_motors: int = 4,
    cycles_per_motor: int = 5,
    n_features: int = 3,
    seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Builds synthetic (X, y_fit, y_metrics, groups) for testing.

    For simplicity, y_fit == y_metrics (regression case).

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


def _make_mock_model(
    train_data: tuple,
    test_data: tuple
) -> MagicMock:
    """Builds a mock BaseRULModel whose prepare_training_data returns
    train_data on the first call and test_data on the second.

    Args:
        train_data: Tuple (X, y_fit, y_metrics, groups) for training.
        test_data: Tuple (X, y_fit, y_metrics, groups) for test.

    Returns:
        MagicMock mimicking a BaseRULModel instance.
    """
    mock = MagicMock(spec=BaseRULModel)
    mock.prepare_training_data.side_effect = [train_data, test_data]
    return mock


def _make_mock_pipeline(
    n_samples: int,
    clipping_threshold: int = 100
) -> MagicMock:
    """Builds a mock sklearn Pipeline for scenario evaluation tests.

    Args:
        n_samples: Number of samples predict() will return values for.
        clipping_threshold: Clipping threshold for the mock model step.

    Returns:
        MagicMock mimicking a fitted sklearn Pipeline.
    """
    mock = MagicMock()
    mock.named_steps = {'model': MagicMock(clipping_threshold=clipping_threshold)}
    mock.predict.side_effect = lambda X: np.full(len(X), 50.0)
    return mock


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for TestManager initialization."""

    def test_prepare_training_data_called_twice(self):
        """prepare_training_data must be called once for train and once for test."""
        train_data = _make_synthetic_data(n_motors=4, seed=0)
        test_data = _make_synthetic_data(n_motors=2, seed=1)
        mock_model = _make_mock_model(train_data, test_data)

        TestManager(model=mock_model, train_ids=np.array([1, 2, 3, 4]), test_ids=np.array([5, 6]))

        assert mock_model.prepare_training_data.call_count == 2

    def test_train_ids_passed_in_first_call(self):
        """prepare_training_data must receive train_ids in the first call."""
        train_data = _make_synthetic_data(n_motors=4, seed=0)
        test_data = _make_synthetic_data(n_motors=2, seed=1)
        mock_model = _make_mock_model(train_data, test_data)
        train_ids = np.array([1, 2, 3, 4])

        TestManager(model=mock_model, train_ids=train_ids, test_ids=np.array([5, 6]))

        first_call_args = mock_model.prepare_training_data.call_args_list[0]
        np.testing.assert_array_equal(first_call_args[0][0], train_ids)

    def test_test_ids_passed_in_second_call(self):
        """prepare_training_data must receive test_ids in the second call."""
        train_data = _make_synthetic_data(n_motors=4, seed=0)
        test_data = _make_synthetic_data(n_motors=2, seed=1)
        mock_model = _make_mock_model(train_data, test_data)
        test_ids = np.array([5, 6])

        TestManager(model=mock_model, train_ids=np.array([1, 2, 3, 4]), test_ids=test_ids)

        second_call_args = mock_model.prepare_training_data.call_args_list[1]
        np.testing.assert_array_equal(second_call_args[0][0], test_ids)

    def test_stores_all_training_attributes(self):
        """All training attributes must be stored correctly after initialization."""
        train_data = _make_synthetic_data(n_motors=4, seed=0)
        test_data = _make_synthetic_data(n_motors=2, seed=1)
        mock_model = _make_mock_model(train_data, test_data)

        tm = TestManager(model=mock_model, train_ids=np.array([1, 2, 3, 4]), test_ids=np.array([5, 6]))

        pd.testing.assert_frame_equal(tm.X_train, train_data[0])
        np.testing.assert_array_equal(tm.y_fit_train, train_data[1])
        np.testing.assert_array_equal(tm.y_metrics_train, train_data[2])
        np.testing.assert_array_equal(tm.groups_train, train_data[3])

    def test_stores_all_test_attributes(self):
        """All test attributes must be stored correctly after initialization."""
        train_data = _make_synthetic_data(n_motors=4, seed=0)
        test_data = _make_synthetic_data(n_motors=2, seed=1)
        mock_model = _make_mock_model(train_data, test_data)

        tm = TestManager(model=mock_model, train_ids=np.array([1, 2, 3, 4]), test_ids=np.array([5, 6]))

        pd.testing.assert_frame_equal(tm.X_test, test_data[0])
        np.testing.assert_array_equal(tm.y_fit_test, test_data[1])
        np.testing.assert_array_equal(tm.y_metrics_test, test_data[2])
        np.testing.assert_array_equal(tm.groups_test, test_data[3])

    def test_model_stored(self):
        """The model instance must be stored as an attribute."""
        train_data = _make_synthetic_data(n_motors=4, seed=0)
        test_data = _make_synthetic_data(n_motors=2, seed=1)
        mock_model = _make_mock_model(train_data, test_data)

        tm = TestManager(model=mock_model, train_ids=np.array([1, 2]), test_ids=np.array([3]))

        assert tm.model is mock_model


# ---------------------------------------------------------------------------
# _get_last_indices
# ---------------------------------------------------------------------------

class TestGetLastIndices:
    """Tests for the last-cycle index extraction."""

    def _make_manager(self) -> TestManager:
        """Returns a TestManager with minimal synthetic data."""
        data = _make_synthetic_data(n_motors=2, cycles_per_motor=3)
        mock_model = _make_mock_model(data, data)
        return TestManager(
            model=mock_model,
            train_ids=np.array([1, 2]),
            test_ids=np.array([1, 2])
        )

    def test_returns_one_index_per_motor(self):
        """Must return exactly one index per unique motor unit."""
        tm = self._make_manager()
        groups = np.array([1, 1, 1, 2, 2, 2])
        indices = tm._get_last_indices(groups)
        assert len(indices) == 2

    def test_returns_last_index_of_each_motor(self):
        """Returned indices must point to the last row of each motor."""
        tm = self._make_manager()
        groups = np.array([1, 1, 1, 2, 2, 2])
        indices = tm._get_last_indices(groups)
        np.testing.assert_array_equal(sorted(indices), [2, 5])

    def test_single_cycle_motor(self):
        """Motors with a single cycle must return that cycle's index."""
        tm = self._make_manager()
        groups = np.array([1, 2, 3])
        indices = tm._get_last_indices(groups)
        np.testing.assert_array_equal(sorted(indices), [0, 1, 2])

    def test_returns_numpy_array(self):
        """Result must be a numpy array."""
        tm = self._make_manager()
        groups = np.array([1, 1, 2, 2])
        indices = tm._get_last_indices(groups)
        assert isinstance(indices, np.ndarray)


# ---------------------------------------------------------------------------
# _evaluate_scenario
# ---------------------------------------------------------------------------

class TestEvaluateScenario:
    """Tests for single scenario evaluation."""

    def _setup(self) -> tuple[TestManager, MagicMock, tuple]:
        """Returns a TestManager, mock pipeline, and synthetic data."""
        n_motors, cycles = 3, 4
        data = _make_synthetic_data(n_motors=n_motors, cycles_per_motor=cycles)
        mock_model = _make_mock_model(data, data)
        tm = TestManager(
            model=mock_model,
            train_ids=np.arange(n_motors),
            test_ids=np.arange(n_motors)
        )
        pipeline = _make_mock_pipeline(n_samples=n_motors * cycles)
        return tm, pipeline, data

    def test_trajectory_uses_all_rows(self):
        """Trajectory mode must evaluate all rows in the dataset."""
        tm, pipeline, (X, y_fit, y_metrics, groups) = self._setup()
        result = tm._evaluate_scenario(pipeline, X, y_metrics, groups, only_last=False)
        assert result['N'] == len(X)

    def test_deployment_uses_one_row_per_motor(self):
        """Deployment mode must evaluate exactly one row per motor unit."""
        tm, pipeline, (X, y_fit, y_metrics, groups) = self._setup()
        n_motors = len(np.unique(groups))
        result = tm._evaluate_scenario(pipeline, X, y_metrics, groups, only_last=True)
        assert result['N'] == n_motors

    def test_result_contains_expected_keys(self):
        """Result must contain N, S-Score, C-Index, MAE, RMSE keys."""
        tm, pipeline, (X, y_fit, y_metrics, groups) = self._setup()
        result = tm._evaluate_scenario(pipeline, X, y_metrics, groups, only_last=False)
        assert set(result.keys()) == {'N', 'S-Score', 'C-Index', 'MAE', 'RMSE'}

    def test_n_is_integer(self):
        """N in the result must be an integer."""
        tm, pipeline, (X, y_fit, y_metrics, groups) = self._setup()
        result = tm._evaluate_scenario(pipeline, X, y_metrics, groups, only_last=False)
        assert isinstance(result['N'], int)

    def test_deployment_predict_receives_fewer_rows(self):
        """Deployment mode must pass fewer rows to predict than trajectory mode."""
        tm, pipeline, (X, y_fit, y_metrics, groups) = self._setup()

        tm._evaluate_scenario(pipeline, X, y_metrics, groups, only_last=False)
        traj_rows = pipeline.predict.call_args_list[-1][0][0]

        pipeline.predict.reset_mock()
        pipeline.predict.side_effect = lambda X: np.full(len(X), 50.0)

        tm._evaluate_scenario(pipeline, X, y_metrics, groups, only_last=True)
        deploy_rows = pipeline.predict.call_args_list[-1][0][0]

        assert len(deploy_rows) < len(traj_rows)


# ---------------------------------------------------------------------------
# evaluate_best_model
# ---------------------------------------------------------------------------

class TestEvaluateBestModel:
    """Integration tests for the full four-scenario evaluation."""

    @pytest.fixture
    def manager_with_nb(self):
        """Returns a TestManager using NegativeBinomialPiecewise with
        small synthetic data, bypassing CSV loading via mocks."""
        from src.models.negative_binomial import NegativeBinomialPiecewise

        train_data = _make_synthetic_data(n_motors=6, cycles_per_motor=10, seed=0)
        test_data = _make_synthetic_data(n_motors=3, cycles_per_motor=10, seed=1)
        model = NegativeBinomialPiecewise()

        with patch.object(model, 'prepare_training_data',
                          side_effect=[train_data, test_data]):
            tm = TestManager(
                model=model,
                train_ids=np.arange(6),
                test_ids=np.arange(3)
            )

        return tm

    def test_returns_dataframe(self, manager_with_nb):
        """evaluate_best_model must return a pandas DataFrame."""
        result = manager_with_nb.evaluate_best_model({'clipping_threshold': 80})
        assert isinstance(result, pd.DataFrame)

    def test_returns_four_rows(self, manager_with_nb):
        """Result must contain exactly four rows."""
        result = manager_with_nb.evaluate_best_model({'clipping_threshold': 80})
        assert len(result) == 4

    def test_contains_expected_columns(self, manager_with_nb):
        """Result must contain all expected columns."""
        result = manager_with_nb.evaluate_best_model({'clipping_threshold': 80})
        expected = {'Partition', 'Mode', 'N', 'S-Score', 'C-Index', 'MAE', 'RMSE'}
        assert set(result.columns) == expected

    def test_row_order_is_correct(self, manager_with_nb):
        """Rows must follow Train/Trajectory, Train/Deployment,
        Test/Trajectory, Test/Deployment order."""
        result = manager_with_nb.evaluate_best_model({'clipping_threshold': 80})
        expected = [
            ('Train', 'Trajectory'),
            ('Train', 'Deployment'),
            ('Test',  'Trajectory'),
            ('Test',  'Deployment'),
        ]
        actual = list(zip(result['Partition'], result['Mode']))
        assert actual == expected

    def test_train_trajectory_n_greater_than_test(self, manager_with_nb):
        """Train trajectory N must exceed test trajectory N."""
        result = manager_with_nb.evaluate_best_model({'clipping_threshold': 80})
        train_n = result.loc[
            (result['Partition'] == 'Train') & (result['Mode'] == 'Trajectory')
        ]['N'].iloc[0]
        test_n = result.loc[
            (result['Partition'] == 'Test') & (result['Mode'] == 'Trajectory')
        ]['N'].iloc[0]
        assert train_n > test_n

    def test_deployment_n_equals_number_of_motors(self, manager_with_nb):
        """Deployment N must equal the number of motors in each partition."""
        result = manager_with_nb.evaluate_best_model({'clipping_threshold': 80})
        train_n = result.loc[
            (result['Partition'] == 'Train') & (result['Mode'] == 'Deployment')
        ]['N'].iloc[0]
        test_n = result.loc[
            (result['Partition'] == 'Test') & (result['Mode'] == 'Deployment')
        ]['N'].iloc[0]
        assert train_n == 6
        assert test_n == 3

    def test_all_metric_values_are_finite(self, manager_with_nb):
        """All metric values in the result must be finite numbers."""
        result = manager_with_nb.evaluate_best_model({'clipping_threshold': 80})
        for col in ['S-Score', 'C-Index', 'MAE', 'RMSE']:
            assert result[col].notna().all()
            assert np.isfinite(result[col]).all()