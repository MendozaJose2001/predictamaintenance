"""Tests for XGBModel (src/models/xgb_model.py).

Uses motor_1_pipeline_output from conftest.py — session-scoped fixture.
Follows the same structure as test_decision_tree.py and test_random_forest.py.
"""

import numpy as np
import pytest

from src.models.xgb_model import XGBModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pipeline_output(motor_1_pipeline_output: dict) -> dict:
    return motor_1_pipeline_output


@pytest.fixture(scope='module')
def fitted_model(pipeline_output: dict) -> XGBModel:
    model = XGBModel(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
    )
    model.fit(pipeline_output['X'], pipeline_output['y_rul'])
    return model


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_params(self):
        model = XGBModel()
        assert model.n_estimators     == 200
        assert model.learning_rate    == 0.1
        assert model.max_depth        == 5
        assert model.subsample        == 1.0
        assert model.colsample_bytree == 1.0
        assert model.reg_lambda       == 1.0
        assert model.min_child_weight == 1
        assert model.clipping_threshold == 125

    def test_custom_params(self):
        model = XGBModel(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            min_child_weight=5,
            clipping_threshold=115,
        )
        assert model.n_estimators     == 300
        assert model.learning_rate    == 0.05
        assert model.max_depth        == 7
        assert model.subsample        == 0.8
        assert model.colsample_bytree == 0.8
        assert model.reg_lambda       == 10.0
        assert model.min_child_weight == 5
        assert model.clipping_threshold == 115

    def test_not_fitted_at_init(self):
        assert XGBModel().is_fitted_ is False

    def test_model_none_at_init(self):
        assert XGBModel().model_ is None


# ---------------------------------------------------------------------------
# TestPrepareTrainingData
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="sliding window pipeline"):
            XGBModel().prepare_training_data(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(self, pipeline_output: dict):
        model = XGBModel(n_estimators=50, max_depth=3)
        result = model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert result is model

    def test_fit_sets_is_fitted(self, fitted_model: XGBModel):
        assert fitted_model.is_fitted_ is True

    def test_fit_sets_model_(self, fitted_model: XGBModel):
        assert fitted_model.model_ is not None

    def test_fit_ignores_groups(self, pipeline_output: dict):
        model = XGBModel(n_estimators=50, max_depth=3)
        model.fit(
            pipeline_output['X'],
            pipeline_output['y_rul'],
            groups=pipeline_output['groups'],
        )
        assert model.is_fitted_ is True

    def test_fit_deterministic(self, pipeline_output: dict):
        """Same data → same predictions (random_state=42 fixed)."""
        m1 = XGBModel(n_estimators=50, max_depth=3)
        m2 = XGBModel(n_estimators=50, max_depth=3)
        m1.fit(pipeline_output['X'], pipeline_output['y_rul'])
        m2.fit(pipeline_output['X'], pipeline_output['y_rul'])
        np.testing.assert_array_equal(
            m1.predict(pipeline_output['X']),
            m2.predict(pipeline_output['X']),
        )

    def test_fit_internal_params_set(self, pipeline_output: dict):
        """Fixed internal params must be passed to XGBRegressor."""
        model = XGBModel(n_estimators=50, max_depth=3)
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert getattr(model.model_, 'objective',    None) == 'reg:squarederror'
        assert getattr(model.model_, 'random_state', None) == 42
        assert getattr(model.model_, 'n_jobs',       None) == 1

    def test_fit_with_subsample(self, pipeline_output: dict):
        model = XGBModel(n_estimators=50, max_depth=3, subsample=0.8)
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True

    def test_fit_with_colsample_bytree(self, pipeline_output: dict):
        model = XGBModel(n_estimators=50, max_depth=3, colsample_bytree=0.8)
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True

    def test_fit_with_reg_lambda(self, pipeline_output: dict):
        model = XGBModel(n_estimators=50, max_depth=3, reg_lambda=10.0)
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True

    def test_fit_with_min_child_weight(self, pipeline_output: dict):
        model = XGBModel(n_estimators=50, max_depth=3, min_child_weight=5)
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_ is True


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_output_shape(self, fitted_model: XGBModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert result.shape == (pipeline_output['X'].shape[0],)

    def test_output_non_negative(self, fitted_model: XGBModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert (result >= 0.0).all()

    def test_output_clipped(self, fitted_model: XGBModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert (result <= fitted_model.clipping_threshold).all()

    def test_output_is_float(self, fitted_model: XGBModel, pipeline_output: dict):
        result = fitted_model.predict(pipeline_output['X'])
        assert np.issubdtype(result.dtype, np.floating)

    def test_returns_nan_if_not_fitted(self, pipeline_output: dict):
        result = XGBModel().predict(pipeline_output['X'])
        assert np.isnan(result).all()

    def test_nan_shape_if_not_fitted(self, pipeline_output: dict):
        result = XGBModel().predict(pipeline_output['X'])
        assert result.shape == (pipeline_output['X'].shape[0],)

    def test_more_estimators_fits_training_better(self, pipeline_output: dict):
        """More estimators should achieve lower training MAE."""
        m_few  = XGBModel(n_estimators=10,  max_depth=3, learning_rate=0.1)
        m_many = XGBModel(n_estimators=200, max_depth=3, learning_rate=0.1)
        m_few.fit(pipeline_output['X'],  pipeline_output['y_rul'])
        m_many.fit(pipeline_output['X'], pipeline_output['y_rul'])

        mae_few  = float(np.mean(np.abs(
            m_few.predict(pipeline_output['X'])  - pipeline_output['y_rul']
        )))
        mae_many = float(np.mean(np.abs(
            m_many.predict(pipeline_output['X']) - pipeline_output['y_rul']
        )))
        assert mae_many <= mae_few