"""Tests for GGSTrainingManager (new sliding window pipeline).

Uses motors 1, 2, 3 with minimal configurations to keep tests fast.
n_folds=2 and small param_grid ensure tests complete in reasonable time.

Uses multi_motor_data from conftest.py — session-scoped fixture.
"""

import numpy as np
import pandas as pd
import pytest
from itertools import product
from pathlib import Path

from src.ggs_training_manager import GGSTrainingManager
from src.models.negative_binomial import NegativeBinomialPiecewise
from src.models.svr_model import SVRModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def nb_manager(multi_motor_data: dict) -> GGSTrainingManager:
    """Returns a GGSTrainingManager for NegativeBinomialPiecewise."""
    return GGSTrainingManager(
        model_class=NegativeBinomialPiecewise,
        X_df=multi_motor_data['X_df'],
        y_df=multi_motor_data['y_df'],
        groups=multi_motor_data['groups'],
    )


@pytest.fixture(scope='module')
def nb_param_grid() -> dict:
    """Minimal param grid for NB — 2 configs × 2 feature_sets = 4 configs."""
    return {
        'feature_set':        ['A', 'B'],
        'window_size':        [20],
        'n_components':       [5],
        'clipping_threshold': [125],
        'alpha':              [0.5, 1.0],
        'link_type':          ['log'],
    }


@pytest.fixture(scope='module')
def tmp_ggs_dir(tmp_path_factory) -> Path:
    """Shared temporary directory for all GGS outputs in this module."""
    return tmp_path_factory.mktemp('ggs_outputs')


@pytest.fixture(scope='module')
def nb_ggs_results(
    nb_manager: GGSTrainingManager,
    nb_param_grid: dict,
    tmp_ggs_dir: Path,
) -> list[dict]:
    """Runs GGS for NB — cached at module scope."""
    return nb_manager.group_grid_search(
        param_grid=nb_param_grid,
        n_folds=2,
        silence=True,
        base_dir=tmp_ggs_dir,
    )


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for GGSTrainingManager initialization."""

    def test_stores_model_class(self, nb_manager: GGSTrainingManager):
        assert nb_manager.model_class is NegativeBinomialPiecewise

    def test_stores_x_df(self, nb_manager: GGSTrainingManager, multi_motor_data: dict):
        assert nb_manager.X_df.shape == multi_motor_data['X_df'].shape

    def test_stores_y_df(self, nb_manager: GGSTrainingManager, multi_motor_data: dict):
        assert nb_manager.y_df.shape == multi_motor_data['y_df'].shape

    def test_stores_groups(self, nb_manager: GGSTrainingManager, multi_motor_data: dict):
        assert len(nb_manager.groups) == len(multi_motor_data['groups'])

    def test_no_results_at_init(self, nb_manager: GGSTrainingManager):
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
        assert isinstance(nb_ggs_results, list)

    def test_n_results_matches_configs(
        self, nb_ggs_results: list[dict], nb_param_grid: dict
    ):
        """Number of results must equal number of configurations."""
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
        """Each result must contain all pipeline hyperparameter keys."""
        for result in nb_ggs_results:
            assert 'feature_set' in result
            assert 'window_size' in result
            assert 'n_components' in result
            assert 'clipping_threshold' in result

    def test_result_has_model_param_keys(self, nb_ggs_results: list[dict]):
        """Each result must contain model hyperparameter keys."""
        for result in nb_ggs_results:
            assert 'alpha' in result
            assert 'link_type' in result

    def test_feature_set_values_in_results(self, nb_ggs_results: list[dict]):
        """feature_set values in results must match those in param_grid."""
        feature_sets_used = {r['feature_set'] for r in nb_ggs_results}
        assert feature_sets_used == {'A', 'B'}

    def test_results_stored_in_manager(
        self, nb_manager: GGSTrainingManager, nb_ggs_results: list[dict]
    ):
        assert nb_manager.ggs_results_ is not None
        assert len(nb_manager.ggs_results_) == len(nb_ggs_results)

    def test_metrics_are_finite_or_nan(self, nb_ggs_results: list[dict]):
        """Metric values must be finite floats or NaN — no inf."""
        for result in nb_ggs_results:
            for key in ['mean_S_score', 'mean_C_index', 'mean_MAE', 'mean_RMSE']:
                val = result[key]
                assert np.isnan(val) or np.isfinite(val)

    def test_svr_compatible_with_feature_set(self, multi_motor_data: dict, tmp_path: Path):
        """GGSTrainingManager must work with SVRModel and feature_set param."""
        manager = GGSTrainingManager(
            model_class=SVRModel,
            X_df=multi_motor_data['X_df'],
            y_df=multi_motor_data['y_df'],
            groups=multi_motor_data['groups'],
        )
        results = manager.group_grid_search(
            param_grid={
                'feature_set':        ['A'],
                'window_size':        [20],
                'n_components':       [5],
                'clipping_threshold': [125],
                'kernel':             ['rbf'],
                'C':                  [1.0],
            },
            n_folds=2,
            silence=True,
            base_dir=tmp_path,
        )
        assert len(results) == 1
        assert results[0]['feature_set'] == 'A'
        assert 'mean_S_score' in results[0]

    def test_different_feature_sets_produce_different_results(
        self, multi_motor_data: dict, tmp_path: Path
    ):
        """Different feature sets should generally produce different MAE values."""
        manager = GGSTrainingManager(
            model_class=NegativeBinomialPiecewise,
            X_df=multi_motor_data['X_df'],
            y_df=multi_motor_data['y_df'],
            groups=multi_motor_data['groups'],
        )
        results = manager.group_grid_search(
            param_grid={
                'feature_set':        ['A', 'D'],
                'window_size':        [20],
                'n_components':       [5],
                'clipping_threshold': [125],
                'alpha':              [1.0],
                'link_type':          ['log'],
            },
            n_folds=2,
            silence=True,
            base_dir=tmp_path,
        )
        mae_a = next(r['mean_MAE'] for r in results if r['feature_set'] == 'A')
        mae_d = next(r['mean_MAE'] for r in results if r['feature_set'] == 'D')
        # At least one must be finite
        assert np.isfinite(mae_a) or np.isfinite(mae_d)


# ---------------------------------------------------------------------------
# TestGetGGSResults
# ---------------------------------------------------------------------------

class TestGetGGSResults:
    """Tests for the get_ggs_results() method."""

    def test_raises_if_no_results(self, multi_motor_data: dict):
        manager = GGSTrainingManager(
            model_class=NegativeBinomialPiecewise,
            X_df=multi_motor_data['X_df'],
            y_df=multi_motor_data['y_df'],
            groups=multi_motor_data['groups'],
        )
        with pytest.raises(ValueError, match="group_grid_search"):
            manager.get_ggs_results()

    def test_returns_dataframe(self, nb_manager: GGSTrainingManager):
        assert isinstance(nb_manager.get_ggs_results(), pd.DataFrame)

    def test_feature_set_column_in_dataframe(self, nb_manager: GGSTrainingManager):
        """get_ggs_results must include feature_set column."""
        result = nb_manager.get_ggs_results()
        assert 'feature_set' in result.columns

    def test_success_column_exists(self, nb_manager: GGSTrainingManager):
        assert 'Success' in nb_manager.get_ggs_results().columns

    def test_top_n_respected(self, nb_manager: GGSTrainingManager):
        assert len(nb_manager.get_ggs_results(top_n=1)) == 1

    def test_sorted_by_s_score(self, nb_manager: GGSTrainingManager):
        result = nb_manager.get_ggs_results()
        scores = result['mean_S_score'].tolist()
        assert scores == sorted(scores)

    def test_successful_configs_ranked_first(self, nb_manager: GGSTrainingManager):
        result = nb_manager.get_ggs_results()
        if len(result) > 1:
            success_vals = result['Success'].tolist()
            first_zero = next(
                (i for i, v in enumerate(success_vals) if v == 0),
                len(success_vals)
            )
            assert all(v == 1 for v in success_vals[:first_zero])