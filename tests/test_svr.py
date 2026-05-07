"""Tests for SVRModel.

Uses motor_1_pipeline_output from conftest.py — session-scoped fixture.
"""

import numpy as np
import pytest

from src.models.svr_model import SVRModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pipeline_output(motor_1_pipeline_output: dict) -> dict:
    return motor_1_pipeline_output


@pytest.fixture(scope='module')
def fitted_model(pipeline_output: dict) -> SVRModel:
    model = SVRModel(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(pipeline_output['X'], pipeline_output['y_rul'])
    return model


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_params(self):
        model = SVRModel()
        assert model.kernel == 'rbf'
        assert model.C == 1.0
        assert model.epsilon == 0.1
        assert model.gamma == 'scale'
        assert model.degree == 3
        assert model.clipping_threshold == 125

    def test_custom_params(self):
        model = SVRModel(kernel='linear', C=10.0, epsilon=0.5,
                         gamma='auto', degree=2, clipping_threshold=110)
        assert model.kernel == 'linear'
        assert model.C == 10.0
        assert model.epsilon == 0.5
        assert model.gamma == 'auto'
        assert model.degree == 2
        assert model.clipping_threshold == 110

    def test_not_fitted_at_init(self):
        assert SVRModel().is_fitted_ is False

    def test_model_none_at_init(self):
        assert SVRModel().model_ is None


# ---------------------------------------------------------------------------
# TestPrepareTrainingData
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="sliding window pipeline"):
            SVRModel().prepare_training_data(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(self, pipeline_output: dict):
        model = SVRModel()
        assert model.fit(pipeline_output['X'], pipeline_output['y_rul']) is model

    def test_fit_sets_is_fitted(self, fitted_model: SVRModel):
        assert fitted_model.is_fitted_ is True

    def test_fit_sets_model_(self, fitted_model: SVRModel):
        assert fitted_model.model_ is not None

    def test_fit_with_rbf_kernel(self, pipeline_output: dict):
        model = SVRModel(kernel='rbf')
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_

    def test_fit_with_linear_kernel(self, pipeline_output: dict):
        model = SVRModel(kernel='linear')
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_

    def test_fit_with_poly_kernel(self, pipeline_output: dict):
        model = SVRModel(kernel='poly', degree=2)
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_

    def test_fit_handles_failure_gracefully(self, pipeline_output: dict):
        model = SVRModel()
        bad_X = pipeline_output['X'].copy()
        bad_X[0, 0] = np.nan
        with pytest.warns(RuntimeWarning):
            model.fit(bad_X, pipeline_output['y_rul'])
        assert model.is_fitted_ is False

    def test_fit_ignores_groups_kwarg(self, pipeline_output: dict):
        model = SVRModel()
        model.fit(pipeline_output['X'], pipeline_output['y_rul'],
                  groups=pipeline_output['groups'])
        assert model.is_fitted_


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_output_shape(self, fitted_model: SVRModel, pipeline_output: dict):
        assert fitted_model.predict(pipeline_output['X']).shape == (pipeline_output['X'].shape[0],)

    def test_output_non_negative(self, fitted_model: SVRModel, pipeline_output: dict):
        assert (fitted_model.predict(pipeline_output['X']) >= 0.0).all()

    def test_output_clipped_to_threshold(self, fitted_model: SVRModel, pipeline_output: dict):
        assert (fitted_model.predict(pipeline_output['X']) <= fitted_model.clipping_threshold).all()

    def test_returns_nan_if_not_fitted(self, pipeline_output: dict):
        assert np.isnan(SVRModel().predict(pipeline_output['X'])).all()

    def test_output_is_float_array(self, fitted_model: SVRModel, pipeline_output: dict):
        assert np.issubdtype(fitted_model.predict(pipeline_output['X']).dtype, np.floating)

    def test_predictions_non_trivial(self, fitted_model: SVRModel, pipeline_output: dict):
        assert fitted_model.predict(pipeline_output['X']).std() > 0.0

    def test_different_kernels_give_different_predictions(self, pipeline_output: dict):
        m_rbf = SVRModel(kernel='rbf')
        m_rbf.fit(pipeline_output['X'], pipeline_output['y_rul'])
        m_lin = SVRModel(kernel='linear')
        m_lin.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert not np.allclose(
            m_rbf.predict(pipeline_output['X']),
            m_lin.predict(pipeline_output['X'])
        )