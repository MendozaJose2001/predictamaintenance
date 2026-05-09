"""Tests for DecisionTreeModel.

Uses motor_1_pipeline_output from conftest.py — session-scoped fixture.
Follows the same structure as test_svr_model.py.
"""

import numpy as np
import pytest

from src.models.decision_tree import DecisionTreeModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pipeline_output(motor_1_pipeline_output: dict) -> dict:
    return motor_1_pipeline_output


@pytest.fixture(scope='module')
def fitted_model(pipeline_output: dict) -> DecisionTreeModel:
    model = DecisionTreeModel(max_depth=5, min_samples_leaf=10)
    model.fit(pipeline_output['X'], pipeline_output['y_rul'])
    return model


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_params(self):
        model = DecisionTreeModel()
        assert model.max_depth is None
        assert model.min_samples_split == 2
        assert model.min_samples_leaf == 1
        assert model.max_features is None
        assert model.clipping_threshold == 125

    def test_custom_params(self):
        model = DecisionTreeModel(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            clipping_threshold=115,
        )
        assert model.max_depth == 5
        assert model.min_samples_split == 10
        assert model.min_samples_leaf == 5
        assert model.max_features == 'sqrt'
        assert model.clipping_threshold == 115

    def test_not_fitted_at_init(self):
        assert DecisionTreeModel().is_fitted_ is False

    def test_model_none_at_init(self):
        assert DecisionTreeModel().model_ is None


# ---------------------------------------------------------------------------
# TestPrepareTrainingData
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="sliding window pipeline"):
            DecisionTreeModel().prepare_training_data(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(self, pipeline_output: dict):
        model = DecisionTreeModel(max_depth=3)
        result = model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert result is model

    def test_fit_sets_is_fitted(self, fitted_model: DecisionTreeModel):
        assert fitted_model.is_fitted_ is True

    def test_fit_sets_model_(self, fitted_model: DecisionTreeModel):
        assert fitted_model.model_ is not None

    def test_fit_ignores_groups(self, pipeline_output: dict):
        model = DecisionTreeModel(max_depth=3)
        model.fit(
            pipeline_output['X'],
            pipeline_output['y_rul'],
            groups=pipeline_output['groups'],
        )
        assert model.is_fitted_ is True

    def test_fit_with_max_features_sqrt(self, pipeline_output: dict):
        model = DecisionTreeModel(max_features='sqrt')
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True

    def test_fit_with_max_features_log2(self, pipeline_output: dict):
        model = DecisionTreeModel(max_features='log2')
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True

    def test_fit_with_max_features_float(self, pipeline_output: dict):
        model = DecisionTreeModel(max_features=0.5)
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True

    def test_fit_deterministic(self, pipeline_output: dict):
        """Same data → same predictions (random_state=42 fixed)."""
        m1 = DecisionTreeModel(max_depth=3)
        m2 = DecisionTreeModel(max_depth=3)
        m1.fit(pipeline_output['X'], pipeline_output['y_rul'])
        m2.fit(pipeline_output['X'], pipeline_output['y_rul'])
        np.testing.assert_array_equal(
            m1.predict(pipeline_output['X']),
            m2.predict(pipeline_output['X']),
        )


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_output_shape(self, fitted_model: DecisionTreeModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert result.shape == (pipeline_output['X'].shape[0],)

    def test_output_non_negative(self, fitted_model: DecisionTreeModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert (result >= 0.0).all()

    def test_output_clipped(self, fitted_model: DecisionTreeModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert (result <= fitted_model.clipping_threshold).all()

    def test_output_is_float(self, fitted_model: DecisionTreeModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert np.issubdtype(result.dtype, np.floating)

    def test_returns_nan_if_not_fitted(self, pipeline_output: dict):
        result = DecisionTreeModel().predict(pipeline_output['X'])
        assert np.isnan(result).all()

    def test_nan_shape_if_not_fitted(self, pipeline_output: dict):
        result = DecisionTreeModel().predict(pipeline_output['X'])
        assert result.shape == (pipeline_output['X'].shape[0],)

    def test_deeper_tree_fits_training_better(self, pipeline_output: dict):
        """Deeper tree should have lower training MAE."""
        m_shallow = DecisionTreeModel(max_depth=2)
        m_deep    = DecisionTreeModel(max_depth=10)
        m_shallow.fit(pipeline_output['X'], pipeline_output['y_rul'])
        m_deep.fit(pipeline_output['X'], pipeline_output['y_rul'])

        mae_shallow = float(np.mean(np.abs(
            m_shallow.predict(pipeline_output['X']) - pipeline_output['y_rul']
        )))
        mae_deep = float(np.mean(np.abs(
            m_deep.predict(pipeline_output['X']) - pipeline_output['y_rul']
        )))
        assert mae_deep <= mae_shallow