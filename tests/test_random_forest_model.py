"""Tests for RandomForestModel.

Uses motor_1_pipeline_output from conftest.py — session-scoped fixture.
Follows the same structure as test_decision_tree_model.py and test_svr_model.py.
"""

import numpy as np
import pytest

from src.models.random_forest import RandomForestModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pipeline_output(motor_1_pipeline_output: dict) -> dict:
    return motor_1_pipeline_output


@pytest.fixture(scope='module')
def fitted_model(pipeline_output: dict) -> RandomForestModel:
    model = RandomForestModel(n_estimators=10, max_depth=5, min_samples_leaf=10)
    model.fit(pipeline_output['X'], pipeline_output['y_rul'])
    return model


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_params(self):
        model = RandomForestModel()
        assert model.n_estimators == 100
        assert model.max_depth is None
        assert model.min_samples_split == 2
        assert model.min_samples_leaf == 1
        assert model.max_features == 1.0
        assert model.clipping_threshold == 125

    def test_custom_params(self):
        model = RandomForestModel(
            n_estimators=50,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            clipping_threshold=115,
        )
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.min_samples_split == 10
        assert model.min_samples_leaf == 5
        assert model.max_features == 'sqrt'
        assert model.clipping_threshold == 115

    def test_not_fitted_at_init(self):
        assert RandomForestModel().is_fitted_ is False

    def test_model_none_at_init(self):
        assert RandomForestModel().model_ is None


# ---------------------------------------------------------------------------
# TestPrepareTrainingData
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="sliding window pipeline"):
            RandomForestModel().prepare_training_data(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(self, pipeline_output: dict):
        model = RandomForestModel(n_estimators=5, max_depth=3)
        result = model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert result is model

    def test_fit_sets_is_fitted(self, fitted_model: RandomForestModel):
        assert fitted_model.is_fitted_ is True

    def test_fit_sets_model_(self, fitted_model: RandomForestModel):
        assert fitted_model.model_ is not None

    def test_fit_ignores_groups(self, pipeline_output: dict):
        model = RandomForestModel(n_estimators=5, max_depth=3)
        model.fit(
            pipeline_output['X'],
            pipeline_output['y_rul'],
            groups=pipeline_output['groups'],
        )
        assert model.is_fitted_ is True

    def test_fit_with_max_features_sqrt(self, pipeline_output: dict):
        model = RandomForestModel(n_estimators=5, max_features='sqrt')
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True

    def test_fit_with_max_features_log2(self, pipeline_output: dict):
        model = RandomForestModel(n_estimators=5, max_features='log2')
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True

    def test_fit_with_max_features_float(self, pipeline_output: dict):
        model = RandomForestModel(n_estimators=5, max_features=0.5)
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True

    def test_fit_deterministic(self, pipeline_output: dict):
        """Same data → same predictions (random_state=42 fixed)."""
        m1 = RandomForestModel(n_estimators=5, max_depth=3)
        m2 = RandomForestModel(n_estimators=5, max_depth=3)
        m1.fit(pipeline_output['X'], pipeline_output['y_rul'])
        m2.fit(pipeline_output['X'], pipeline_output['y_rul'])
        np.testing.assert_array_equal(
            m1.predict(pipeline_output['X']),
            m2.predict(pipeline_output['X']),
        )

    def test_n_estimators_stored(self, pipeline_output: dict):
        model = RandomForestModel(n_estimators=7, max_depth=3)
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True
        assert model.n_estimators == 7


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_output_shape(self, fitted_model: RandomForestModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert result.shape == (pipeline_output['X'].shape[0],)

    def test_output_non_negative(self, fitted_model: RandomForestModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert (result >= 0.0).all()

    def test_output_clipped(self, fitted_model: RandomForestModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert (result <= fitted_model.clipping_threshold).all()

    def test_output_is_float(self, fitted_model: RandomForestModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert np.issubdtype(result.dtype, np.floating)

    def test_returns_nan_if_not_fitted(self, pipeline_output: dict):
        result = RandomForestModel().predict(pipeline_output['X'])
        assert np.isnan(result).all()

    def test_nan_shape_if_not_fitted(self, pipeline_output: dict):
        result = RandomForestModel().predict(pipeline_output['X'])
        assert result.shape == (pipeline_output['X'].shape[0],)

    def test_more_estimators_reduces_variance(self, pipeline_output: dict):
        """More trees → lower variance (std of predictions should decrease)."""
        m_few  = RandomForestModel(n_estimators=5,   max_depth=5)
        m_many = RandomForestModel(n_estimators=100, max_depth=5)
        m_few.fit(pipeline_output['X'],  pipeline_output['y_rul'])
        m_many.fit(pipeline_output['X'], pipeline_output['y_rul'])

        std_few  = float(m_few.predict(pipeline_output['X']).std())
        std_many = float(m_many.predict(pipeline_output['X']).std())
        # RF with more trees produces smoother (lower std) predictions
        assert std_many <= std_few