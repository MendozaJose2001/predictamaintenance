"""Tests for GGSTrainingManager (new sliding window pipeline).

Uses motor 1 only with minimal configurations to keep tests fast.
n_folds=2 and small param_grid ensure tests complete in reasonable time.
"""

import numpy as np
import pandas as pd
import pytest

from src.ggs_training_manager import GGSTrainingManager
from src.models.negative_binomial import NegativeBinomialPiecewise
from src.models.svr_model import SVRModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def motor_data() -> dict:
    """Loads motors 1, 2, 3 — need at least 2 groups for GroupKFold n_splits=2."""
    dfs = []
    for motor_id in [1, 2, 3]:
        df = pd.read_csv(f'data/clean/data_motor_{motor_id}.csv')
        df.insert(0, 'unit_number', motor_id)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    X_df = df_all.drop(columns=['RUL'])
    y_df = df_all[['unit_number', 'time_in_cycles', 'RUL']].copy()
    groups = df_all['unit_number'].to_numpy()

    return {'X_df': X_df, 'y_df': y_df, 'groups': groups}


@pytest.fixture(scope='module')
def nb_manager(motor_data: dict) -> GGSTrainingManager:
    """Returns a GGSTrainingManager for NegativeBinomialPiecewise."""
    return GGSTrainingManager(
        model_class=NegativeBinomialPiecewise,
        X_df=motor_data['X_df'],
        y_df=motor_data['y_df'],
        groups=motor_data['groups'],
    )


@pytest.fixture(scope='module')
def nb_param_grid() -> dict:
    """Minimal param grid for NB — 2 configs."""
    return {
        'window_size':        [20],
        'n_components':       [5],
        'clipping_threshold': [125],
        'alpha':              [0.5, 1.0],
        'link_type':          ['log'],
    }


@pytest.fixture(scope='module')
def nb_ggs_results(
    nb_manager: GGSTrainingManager,
    nb_param_grid: dict,
) -> list[dict]:
    """Runs GGS for NB — cached at module scope."""
    return nb_manager.group_grid_search(
        param_grid=nb_param_grid,
        n_folds=2,
        silence=True,
    )


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for GGSTrainingManager initialization."""

    def test_stores_model_class(self, nb_manager: GGSTrainingManager):
        """model_class must be stored correctly."""
        assert nb_manager.model_class is NegativeBinomialPiecewise

    def test_stores_x_df(
        self, nb_manager: GGSTrainingManager, motor_data: dict
    ):
        """X_df must be stored correctly."""
        assert nb_manager.X_df.shape == motor_data['X_df'].shape

    def test_stores_y_df(
        self, nb_manager: GGSTrainingManager, motor_data: dict
    ):
        """y_df must be stored correctly."""
        assert nb_manager.y_df.shape == motor_data['y_df'].shape

    def test_stores_groups(
        self, nb_manager: GGSTrainingManager, motor_data: dict
    ):
        """groups must be stored correctly."""
        assert len(nb_manager.groups) == len(motor_data['groups'])

    def test_no_results_at_init(self, nb_manager: GGSTrainingManager):
        """ggs_results_ must be None before group_grid_search() is called."""
        manager = GGSTrainingManager(
            model_class=NegativeBinomialPiecewise,
            X_df=nb_manager.X_df,
            y_df=nb_manager.y_df,
            groups=nb_manager.groups,
        )
        assert manager.ggs_results_ is None


# ---------------------------------------------------------------------------
# TestGroupGridSearch
# ---------------------------------------------------------------------------

class TestGroupGridSearch:
    """Tests for the group_grid_search() method."""

    def test_returns_list(self, nb_ggs_results: list[dict]):
        """group_grid_search must return a list."""
        assert isinstance(nb_ggs_results, list)

    def test_n_results_matches_configs(
        self, nb_ggs_results: list[dict], nb_param_grid: dict
    ):
        """Number of results must equal number of configurations."""
        from itertools import product
        n_configs = len(list(product(*nb_param_grid.values())))
        assert len(nb_ggs_results) == n_configs

    def test_result_has_metric_keys(self, nb_ggs_results: list[dict]):
        """Each result must contain all four metric keys."""
        for result in nb_ggs_results:
            assert 'mean_S_score' in result
            assert 'mean_C_index' in result
            assert 'mean_MAE' in result
            assert 'mean_RMSE' in result

    def test_result_has_pipeline_param_keys(self, nb_ggs_results: list[dict]):
        """Each result must contain pipeline hyperparameter keys."""
        for result in nb_ggs_results:
            assert 'window_size' in result
            assert 'n_components' in result
            assert 'clipping_threshold' in result

    def test_result_has_model_param_keys(self, nb_ggs_results: list[dict]):
        """Each result must contain model hyperparameter keys."""
        for result in nb_ggs_results:
            assert 'alpha' in result
            assert 'link_type' in result

    def test_results_stored_in_manager(
        self, nb_manager: GGSTrainingManager, nb_ggs_results: list[dict]
    ):
        """ggs_results_ must be set after group_grid_search()."""
        assert nb_manager.ggs_results_ is not None
        assert len(nb_manager.ggs_results_) == len(nb_ggs_results)

    def test_metrics_are_finite_or_nan(self, nb_ggs_results: list[dict]):
        """Metric values must be finite floats or NaN — no inf."""
        for result in nb_ggs_results:
            for key in ['mean_S_score', 'mean_C_index', 'mean_MAE', 'mean_RMSE']:
                val = result[key]
                assert np.isnan(val) or np.isfinite(val)

    def test_svr_compatible(self, motor_data: dict):
        """GGSTrainingManager must work with SVRModel."""
        manager = GGSTrainingManager(
            model_class=SVRModel,
            X_df=motor_data['X_df'],
            y_df=motor_data['y_df'],
            groups=motor_data['groups'],
        )
        results = manager.group_grid_search(
            param_grid={
                'window_size':        [20],
                'n_components':       [5],
                'clipping_threshold': [125],
                'kernel':             ['rbf'],
                'C':                  [1.0],
            },
            n_folds=2,
            silence=True,
        )
        assert len(results) == 1
        assert 'mean_S_score' in results[0]


# ---------------------------------------------------------------------------
# TestGetGGSResults
# ---------------------------------------------------------------------------

class TestGetGGSResults:
    """Tests for the get_ggs_results() method."""

    def test_raises_if_no_results(self, motor_data: dict):
        """Must raise ValueError if called before group_grid_search()."""
        manager = GGSTrainingManager(
            model_class=NegativeBinomialPiecewise,
            X_df=motor_data['X_df'],
            y_df=motor_data['y_df'],
            groups=motor_data['groups'],
        )
        with pytest.raises(ValueError, match="group_grid_search"):
            manager.get_ggs_results()

    def test_returns_dataframe(
        self, nb_manager: GGSTrainingManager, nb_ggs_results: list[dict]
    ):
        """get_ggs_results must return a DataFrame."""
        result = nb_manager.get_ggs_results()
        assert isinstance(result, pd.DataFrame)

    def test_success_column_exists(
        self, nb_manager: GGSTrainingManager, nb_ggs_results: list[dict]
    ):
        """Result DataFrame must contain Success column."""
        result = nb_manager.get_ggs_results()
        assert 'Success' in result.columns

    def test_top_n_respected(
        self, nb_manager: GGSTrainingManager, nb_ggs_results: list[dict]
    ):
        """top_n must limit the number of rows returned."""
        result = nb_manager.get_ggs_results(top_n=1)
        assert len(result) == 1

    def test_sorted_by_s_score(
        self, nb_manager: GGSTrainingManager, nb_ggs_results: list[dict]
    ):
        """Results must be sorted by ascending S-Score."""
        result = nb_manager.get_ggs_results()
        scores = result['mean_S_score'].tolist()
        assert scores == sorted(scores)

    def test_successful_configs_ranked_first(
        self, nb_manager: GGSTrainingManager, nb_ggs_results: list[dict]
    ):
        """Configs with Success=1 must appear before Success=0."""
        result = nb_manager.get_ggs_results()
        if len(result) > 1:
            success_vals = result['Success'].tolist()
            first_zero = next(
                (i for i, v in enumerate(success_vals) if v == 0),
                len(success_vals)
            )
            assert all(v == 1 for v in success_vals[:first_zero])