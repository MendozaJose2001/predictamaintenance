"""Tests for RULPipeline orchestrator (Nodos 2-4).

Uses fixtures from conftest.py — session-scoped to avoid re-running
the expensive Nodo 3 multiple times.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.rul_pipeline import RULPipeline, PipelineOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pipeline(motor_1_pipeline_output: dict) -> RULPipeline:
    """Returns the fitted pipeline from conftest."""
    return motor_1_pipeline_output['pipeline']


@pytest.fixture(scope='module')
def fit_output(motor_1_pipeline_output: dict) -> PipelineOutput:
    """Returns the fit_transform output from conftest."""
    d = motor_1_pipeline_output
    return d['X'], d['y_rul'], d['t_stop'], d['evento'], d['groups']


@pytest.fixture(scope='module')
def X_df(motor_1_pipeline_output: dict) -> pd.DataFrame:
    return motor_1_pipeline_output['X_df']


@pytest.fixture(scope='module')
def y_df(motor_1_pipeline_output: dict) -> pd.DataFrame:
    return motor_1_pipeline_output['y_df']


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_params(self):
        p = RULPipeline()
        assert p.window_size == 30
        assert p.clipping_threshold == 125
        assert p.n_components == 10

    def test_custom_params(self):
        p = RULPipeline(window_size=20, clipping_threshold=110, n_components=5)
        assert p.window_size == 20
        assert p.clipping_threshold == 110
        assert p.n_components == 5

    def test_not_fitted_at_init(self):
        assert RULPipeline().is_fitted_ is False


# ---------------------------------------------------------------------------
# TestFitTransform
# ---------------------------------------------------------------------------

class TestFitTransform:

    def test_returns_five_arrays(self, fit_output: PipelineOutput):
        assert len(fit_output) == 5

    def test_x_shape_correct(self, fit_output: PipelineOutput, pipeline: RULPipeline):
        X, *_ = fit_output
        assert X.ndim == 2
        assert X.shape[1] == pipeline.n_components

    def test_all_arrays_same_length(self, fit_output: PipelineOutput):
        X, y_rul, t_stop, evento, groups = fit_output
        n = X.shape[0]
        assert len(y_rul) == n
        assert len(t_stop) == n
        assert len(evento) == n
        assert len(groups) == n

    def test_y_rul_no_nan(self, fit_output: PipelineOutput):
        _, y_rul, *_ = fit_output
        assert not np.isnan(y_rul).any()

    def test_y_rul_clipped(self, fit_output: PipelineOutput, pipeline: RULPipeline):
        _, y_rul, *_ = fit_output
        assert (y_rul <= pipeline.clipping_threshold).all()

    def test_y_rul_non_negative(self, fit_output: PipelineOutput):
        _, y_rul, *_ = fit_output
        assert (y_rul >= 0.0).all()

    def test_t_stop_non_decreasing(self, fit_output: PipelineOutput):
        X, y_rul, t_stop, evento, groups = fit_output
        for motor_id in np.unique(groups):
            assert (np.diff(t_stop[groups == motor_id]) >= 0).all()

    def test_evento_sum_correct(self, fit_output: PipelineOutput):
        _, _, _, evento, _ = fit_output
        assert evento.sum() == 1

    def test_evento_at_last_window(self, fit_output: PipelineOutput):
        _, _, _, evento, _ = fit_output
        assert evento[-1] == 1

    def test_groups_contains_motor_id(self, fit_output: PipelineOutput):
        _, _, _, _, groups = fit_output
        assert set(groups.tolist()) == {1}

    def test_sets_is_fitted(self, pipeline: RULPipeline):
        assert pipeline.is_fitted_ is True

    def test_x_no_nan(self, fit_output: PipelineOutput):
        X, *_ = fit_output
        assert not np.isnan(X).any()

    def test_x_no_inf(self, fit_output: PipelineOutput):
        X, *_ = fit_output
        assert np.isfinite(X).all()

    def test_explained_variance_available(self, pipeline: RULPipeline):
        evr = pipeline.explained_variance_ratio()
        assert len(evr) == pipeline.n_components
        assert (evr >= 0.0).all()
        assert evr.sum() <= 1.0 + 1e-10

    def test_y_rul_nan_without_y_df(self, X_df: pd.DataFrame):
        p = RULPipeline(window_size=20, clipping_threshold=125, n_components=5)
        X, y_rul, *_ = p.fit_transform(X_df, y_df=None)
        assert np.isnan(y_rul).all()


# ---------------------------------------------------------------------------
# TestTransform
# ---------------------------------------------------------------------------

class TestTransform:

    def test_raises_if_not_fitted(self, X_df: pd.DataFrame):
        with pytest.raises(RuntimeError, match="not fitted"):
            RULPipeline(window_size=20, n_components=5).transform(X_df)

    def test_returns_five_arrays(self, pipeline: RULPipeline, X_df: pd.DataFrame, y_df: pd.DataFrame):
        assert len(pipeline.transform(X_df, y_df)) == 5

    def test_x_shape_matches_fit(self, pipeline: RULPipeline, fit_output: PipelineOutput, X_df: pd.DataFrame, y_df: pd.DataFrame):
        X_val, *_ = pipeline.transform(X_df, y_df)
        X_train, *_ = fit_output
        assert X_val.shape[1] == X_train.shape[1]

    def test_y_rul_no_nan_with_y_df(self, pipeline: RULPipeline, X_df: pd.DataFrame, y_df: pd.DataFrame):
        _, y_rul, *_ = pipeline.transform(X_df, y_df)
        assert not np.isnan(y_rul).any()

    def test_y_rul_nan_without_y_df(self, pipeline: RULPipeline, X_df: pd.DataFrame):
        _, y_rul, *_ = pipeline.transform(X_df, y_df=None)
        assert np.isnan(y_rul).all()

    def test_x_no_nan(self, pipeline: RULPipeline, X_df: pd.DataFrame, y_df: pd.DataFrame):
        X, *_ = pipeline.transform(X_df, y_df)
        assert not np.isnan(X).any()

    def test_transform_consistent_with_fit_transform(self, pipeline: RULPipeline, fit_output: PipelineOutput, X_df: pd.DataFrame, y_df: pd.DataFrame):
        X_fit, *_ = fit_output
        X_transform, *_ = pipeline.transform(X_df, y_df)
        np.testing.assert_array_almost_equal(X_fit, X_transform)