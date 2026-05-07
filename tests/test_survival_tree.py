"""Tests for SurvivalTreeModel (Nodo 5).

Uses motor_1_pipeline_output from conftest.py — session-scoped fixture.
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sksurv.util import Surv

from src.models.survival_tree import SurvivalTreeModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pipeline_output(motor_1_pipeline_output: dict) -> dict:
    return motor_1_pipeline_output


@pytest.fixture(scope='module')
def y_surv_structured(pipeline_output: dict) -> np.ndarray:
    return np.array(
        list(zip(pipeline_output['evento'].astype(bool),
                 pipeline_output['t_stop'].astype(float))),
        dtype=[('evento', bool), ('t_stop', float)]
    )


@pytest.fixture(scope='module')
def y_surv_2d(pipeline_output: dict) -> np.ndarray:
    return np.column_stack([
        pipeline_output['evento'].astype(float),
        pipeline_output['t_stop'].astype(float),
    ])


@pytest.fixture(scope='module')
def fitted_model(pipeline_output: dict, y_surv_structured: np.ndarray) -> SurvivalTreeModel:
    model = SurvivalTreeModel(max_depth=3, min_samples_leaf=10, min_samples_split=20)
    model.fit(pipeline_output['X'], y_surv_structured)
    return model


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_params(self):
        model = SurvivalTreeModel()
        assert model.max_depth == 5
        assert model.min_samples_leaf == 20
        assert model.min_samples_split == 40
        assert model.confidence_threshold == 0.5
        assert model.clipping_threshold == 125

    def test_custom_params(self):
        model = SurvivalTreeModel(max_depth=3, min_samples_leaf=10,
                                   confidence_threshold=0.7, clipping_threshold=110)
        assert model.max_depth == 3
        assert model.min_samples_leaf == 10
        assert model.confidence_threshold == 0.7
        assert model.clipping_threshold == 110

    def test_not_fitted_at_init(self):
        assert SurvivalTreeModel().is_fitted_ is False


# ---------------------------------------------------------------------------
# TestPrepareTrainingData
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="sliding window pipeline"):
            SurvivalTreeModel().prepare_training_data(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(self, pipeline_output: dict, y_surv_structured: np.ndarray):
        model = SurvivalTreeModel(max_depth=3, min_samples_leaf=10, min_samples_split=20)
        assert model.fit(pipeline_output['X'], y_surv_structured) is model

    def test_fit_sets_is_fitted(self, fitted_model: SurvivalTreeModel):
        assert fitted_model.is_fitted_ is True

    def test_fit_sets_model_(self, fitted_model: SurvivalTreeModel):
        assert hasattr(fitted_model, 'model_')

    def test_fit_accepts_structured_array(self, pipeline_output: dict, y_surv_structured: np.ndarray):
        model = SurvivalTreeModel(max_depth=3, min_samples_leaf=10, min_samples_split=20)
        model.fit(pipeline_output['X'], y_surv_structured)
        assert model.is_fitted_

    def test_fit_accepts_2d_array(self, pipeline_output: dict, y_surv_2d: np.ndarray):
        model = SurvivalTreeModel(max_depth=3, min_samples_leaf=10, min_samples_split=20)
        model.fit(pipeline_output['X'], y_surv_2d)
        assert model.is_fitted_

    def test_fit_handles_bad_y_gracefully(self, pipeline_output: dict):
        model = SurvivalTreeModel()
        bad_y = np.zeros((len(pipeline_output['X']), 3))
        with pytest.warns(RuntimeWarning):
            model.fit(pipeline_output['X'], bad_y)
        assert model.is_fitted_ is False

    def test_fit_ignores_groups_kwarg(self, pipeline_output: dict, y_surv_structured: np.ndarray):
        model = SurvivalTreeModel(max_depth=3, min_samples_leaf=10, min_samples_split=20)
        model.fit(pipeline_output['X'], y_surv_structured,
                  groups=pipeline_output['groups'])
        assert model.is_fitted_


# ---------------------------------------------------------------------------
# TestPredictSurvivalFunction
# ---------------------------------------------------------------------------

class TestPredictSurvivalFunction:

    def test_returns_list_of_tuples(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        result = fitted_model.predict_survival_function(pipeline_output['X'])
        assert isinstance(result, list) and isinstance(result[0], tuple) and len(result[0]) == 2

    def test_n_tuples_matches_n_windows(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        assert len(fitted_model.predict_survival_function(pipeline_output['X'])) == pipeline_output['X'].shape[0]

    def test_times_non_decreasing(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        for times, _ in fitted_model.predict_survival_function(pipeline_output['X'][:5]):
            assert (np.diff(times) >= 0).all()

    def test_probs_in_unit_interval(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        for _, probs in fitted_model.predict_survival_function(pipeline_output['X'][:5]):
            assert (probs >= 0.0).all() and (probs <= 1.0 + 1e-10).all()

    def test_probs_non_increasing(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        for _, probs in fitted_model.predict_survival_function(pipeline_output['X'][:5]):
            assert (np.diff(probs) <= 1e-10).all()

    def test_raises_if_not_fitted(self, pipeline_output: dict):
        with pytest.raises(NotFittedError):
            SurvivalTreeModel().predict_survival_function(pipeline_output['X'])


# ---------------------------------------------------------------------------
# TestPredictWithTime
# ---------------------------------------------------------------------------

class TestPredictWithTime:

    def test_output_shape(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        result = fitted_model.predict_with_time(pipeline_output['X'], pipeline_output['t_stop'])
        assert result.shape == (pipeline_output['X'].shape[0],)

    def test_output_non_negative(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        assert (fitted_model.predict_with_time(pipeline_output['X'], pipeline_output['t_stop']) >= 0.0).all()

    def test_output_clipped_to_threshold(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        result = fitted_model.predict_with_time(pipeline_output['X'], pipeline_output['t_stop'])
        assert (result <= fitted_model.clipping_threshold).all()

    def test_returns_nan_if_not_fitted(self, pipeline_output: dict):
        result = SurvivalTreeModel().predict_with_time(pipeline_output['X'], pipeline_output['t_stop'])
        assert np.isnan(result).all()

    def test_output_is_float_array(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        result = fitted_model.predict_with_time(pipeline_output['X'], pipeline_output['t_stop'])
        assert np.issubdtype(result.dtype, np.floating)


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_output_shape(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        assert fitted_model.predict(pipeline_output['X']).shape == (pipeline_output['X'].shape[0],)

    def test_returns_nan_if_not_fitted(self, pipeline_output: dict):
        assert np.isnan(SurvivalTreeModel().predict(pipeline_output['X'])).all()

    def test_output_non_negative(self, fitted_model: SurvivalTreeModel, pipeline_output: dict):
        assert (fitted_model.predict(pipeline_output['X']) >= 0.0).all()