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
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Builds synthetic (X, y, groups) for testing.

    Args:
        n_motors: Number of motor units.
        cycles_per_motor: Number of cycles per motor.
        n_features: Number of sensor features.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y_count, groups).
    """
    rng = np.random.default_rng(seed)
    n_samples = n_motors * cycles_per_motor

    X = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f'sensor_{i}' for i in range(n_features)]
    )
    y = rng.integers(1, 100, size=n_samples).astype(float)
    groups = np.repeat(np.arange(1, n_motors + 1), cycles_per_motor)

    return X, y, groups


def _make_mock_model(
    train_data: tuple,
    test_data: tuple
) -> MagicMock:
    """Builds a mock BaseRULModel whose prepare_training_data returns
    train_data on the first call and test_data on the second.

    Args:
        train_data: Tuple (X, y, groups) for the training partition.
        test_data: Tuple (X, y, groups) for the test partition.

    Returns:
        MagicMock mimicking a BaseRULModel instance.
    """
    mock = MagicMock(spec=BaseRULModel)
    mock.prepare_training_data.side_effect = [train_data, test_data]
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

        TestManager(model=mock_model, train_ids=[1, 2, 3, 4], test_ids=[5, 6])

        assert mock_model.prepare_training_data.call_count == 2

    def test_train_ids_passed_correctly(self):
        """prepare_training_data must receive train_ids in the first call."""
        train_data = _make_synthetic_data(n_motors=4, seed=0)
        test_data = _make_synthetic_data(n_motors=2, seed=1)
        mock_model = _make_mock_model(train_data, test_data)

        TestManager(model=mock_model, train_ids=[1, 2, 3, 4], test_ids=[5, 6])

        first_call_args = mock_model.prepare_training_data.call_args_list[0]
        assert first_call_args == (([1, 2, 3, 4],),)

    def test_test_ids_passed_correctly(self):
        """prepare_training_data must receive test_ids in the second call."""
        train_data = _make_synthetic_data(n_motors=4, seed=0)
        test_data = _make_synthetic_data(n_motors=2, seed=1)
        mock_model = _make_mock_model(train_data, test_data)

        TestManager(model=mock_model, train_ids=[1, 2, 3, 4], test_ids=[5, 6])

        second_call_args = mock_model.prepare_training_data.call_args_list[1]
        assert second_call_args == (([5, 6],),)

    def test_stores_training_data(self):
        """X_train, y_train and groups_train must be stored correctly."""
        train_data = _make_synthetic_data(n_motors=4, seed=0)
        test_data = _make_synthetic_data(n_motors=2, seed=1)
        mock_model = _make_mock_model(train_data, test_data)

        tm = TestManager(model=mock_model, train_ids=[1, 2, 3, 4], test_ids=[5, 6])

        pd.testing.assert_frame_equal(tm.X_train, train_data[0])
        np.testing.assert_array_equal(tm.y_train, train_data[1])
        np.testing.assert_array_equal(tm.groups_train, train_data[2])

    def test_stores_test_data(self):
        """X_test, y_test and groups_test must be stored correctly."""
        train_data = _make_synthetic_data(n_motors=4, seed=0)
        test_data = _make_synthetic_data(n_motors=2, seed=1)
        mock_model = _make_mock_model(train_data, test_data)

        tm = TestManager(model=mock_model, train_ids=[1, 2, 3, 4], test_ids=[5, 6])

        pd.testing.assert_frame_equal(tm.X_test, test_data[0])
        np.testing.assert_array_equal(tm.y_test, test_data[1])
        np.testing.assert_array_equal(tm.groups_test, test_data[2])


# ---------------------------------------------------------------------------
# _get_last_indices
# ---------------------------------------------------------------------------

class TestGetLastIndices:
    """Tests for the last-cycle index extraction."""

    def _make_manager(self) -> TestManager:
        """Returns a TestManager with minimal synthetic data."""
        data = _make_synthetic_data(n_motors=2, cycles_per_motor=3)
        mock_model = _make_mock_model(data, data)
        return TestManager(model=mock_model, train_ids=[1, 2], test_ids=[1, 2])

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


# ---------------------------------------------------------------------------
# _evaluate_scenario
# ---------------------------------------------------------------------------

class TestEvaluateScenario:
    """Tests for single scenario evaluation."""

    def _make_manager_and_pipeline(self):
        """Returns a TestManager and a mock pipeline for scenario tests."""
        n_motors, cycles = 3, 4
        data = _make_synthetic_data(n_motors=n_motors, cycles_per_motor=cycles)
        mock_model = _make_mock_model(data, data)
        tm = TestManager(model=mock_model, train_ids=list(range(n_motors)),
                         test_ids=list(range(n_motors)))

        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {
            'model': MagicMock(clipping_threshold=100)
        }
        mock_pipeline.predict.side_effect = lambda X: np.full(len(X), 50.0)

        return tm, mock_pipeline, data

    def test_trajectory_mode_uses_all_rows(self):
        """Trajectory mode must evaluate all rows in the dataset."""
        tm, pipeline, (X, y, groups) = self._make_manager_and_pipeline()
        result = tm._evaluate_scenario(pipeline, X, y, groups, only_last=False)
        assert result['N'] == len(X)

    def test_deployment_mode_uses_one_row_per_motor(self):
        """Deployment mode must evaluate exactly one row per motor unit."""
        tm, pipeline, (X, y, groups) = self._make_manager_and_pipeline()
        n_motors = len(np.unique(groups))
        result = tm._evaluate_scenario(pipeline, X, y, groups, only_last=True)
        assert result['N'] == n_motors

    def test_result_contains_expected_keys(self):
        """Result dictionary must contain N, S-Score, C-Index, MAE, RMSE."""
        tm, pipeline, (X, y, groups) = self._make_manager_and_pipeline()
        result = tm._evaluate_scenario(pipeline, X, y, groups, only_last=False)
        assert set(result.keys()) == {'N', 'S-Score', 'C-Index', 'MAE', 'RMSE'}


# ---------------------------------------------------------------------------
# evaluate_best_model
# ---------------------------------------------------------------------------

class TestEvaluateBestModel:
    """Integration tests for the full four-scenario evaluation."""

    @pytest.fixture
    def manager_with_real_model(self):
        """Returns a TestManager using a real NegativeBinomialPiecewise model
        with small synthetic data, bypassing CSV loading via mocks."""
        from src.models.negative_binomial import NegativeBinomialPiecewise

        train_data = _make_synthetic_data(n_motors=6, cycles_per_motor=10, seed=0)
        test_data = _make_synthetic_data(n_motors=3, cycles_per_motor=10, seed=1)

        model = NegativeBinomialPiecewise()

        with patch.object(model, 'prepare_training_data',
                          side_effect=[train_data, test_data]):
            tm = TestManager(
                model=model,
                train_ids=list(range(6)),
                test_ids=list(range(3))
            )

        return tm

    def test_returns_dataframe(self, manager_with_real_model):
        """evaluate_best_model must return a pandas DataFrame."""
        param_grid = {'model__clipping_threshold': 80}
        result = manager_with_real_model.evaluate_best_model(param_grid)
        assert isinstance(result, pd.DataFrame)

    def test_returns_four_rows(self, manager_with_real_model):
        """Result must contain exactly four rows (one per scenario)."""
        param_grid = {'model__clipping_threshold': 80}
        result = manager_with_real_model.evaluate_best_model(param_grid)
        assert len(result) == 4

    def test_contains_expected_columns(self, manager_with_real_model):
        """Result must contain all expected columns."""
        param_grid = {'model__clipping_threshold': 80}
        result = manager_with_real_model.evaluate_best_model(param_grid)
        expected = {'Partition', 'Mode', 'N', 'S-Score', 'C-Index', 'MAE', 'RMSE'}
        assert set(result.columns) == expected

    def test_row_order_is_correct(self, manager_with_real_model):
        """Rows must follow Train/Trajectory, Train/Deployment,
        Test/Trajectory, Test/Deployment order."""
        param_grid = {'model__clipping_threshold': 80}
        result = manager_with_real_model.evaluate_best_model(param_grid)

        expected = [
            ('Train', 'Trajectory'),
            ('Train', 'Deployment'),
            ('Test',  'Trajectory'),
            ('Test',  'Deployment'),
        ]
        actual = list(zip(result['Partition'], result['Mode']))
        assert actual == expected

    def test_train_n_greater_than_test_n_in_trajectory(self, manager_with_real_model):
        """Train trajectory N must exceed test trajectory N given more train motors."""
        param_grid = {'model__clipping_threshold': 80}
        result = manager_with_real_model.evaluate_best_model(param_grid)

        train_traj_n = result.loc[result['Mode'] == 'Trajectory'].iloc[0]['N']
        test_traj_n = result.loc[result['Mode'] == 'Trajectory'].iloc[1]['N']
        assert train_traj_n > test_traj_n

    def test_deployment_n_equals_number_of_motors(self, manager_with_real_model):
        """Deployment N must equal the number of motors in each partition."""
        param_grid = {'model__clipping_threshold': 80}
        result = manager_with_real_model.evaluate_best_model(param_grid)

        train_deploy_n = result.loc[
            (result['Partition'] == 'Train') & (result['Mode'] == 'Deployment')
        ]['N'].iloc[0]
        test_deploy_n = result.loc[
            (result['Partition'] == 'Test') & (result['Mode'] == 'Deployment')
        ]['N'].iloc[0]

        assert train_deploy_n == 6
        assert test_deploy_n == 3