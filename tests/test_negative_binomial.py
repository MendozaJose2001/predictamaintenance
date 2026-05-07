"""Tests for NegativeBinomialPiecewise v2 (new pipeline version).

Uses motor_1_pipeline_output from conftest.py — session-scoped fixture
that runs Nodos 2-4 once for the entire test session.
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.models.negative_binomial import NegativeBinomialPiecewise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pipeline_output(motor_1_pipeline_output: dict) -> dict:
    """Alias to conftest fixture for local use."""
    return motor_1_pipeline_output


@pytest.fixture(scope='module')
def fitted_model(pipeline_output: dict) -> NegativeBinomialPiecewise:
    """Returns a fitted NegativeBinomialPiecewise on motor 1 data."""
    model = NegativeBinomialPiecewise(alpha=1.0, link_type='log')
    model.fit(pipeline_output['X'], pipeline_output['y_rul'])
    return model


@pytest.fixture(scope='module')
def fitted_model_regularized(
    pipeline_output: dict,
) -> NegativeBinomialPiecewise:
    """Returns a fitted regularized NegativeBinomialPiecewise."""
    model = NegativeBinomialPiecewise(alpha=1.0, alpha_reg=0.1, l1_ratio=0.5)
    model.fit(pipeline_output['X'], pipeline_output['y_rul'])
    return model


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_params(self):
        model = NegativeBinomialPiecewise()
        assert model.alpha == 1.0
        assert model.clipping_threshold == 125
        assert model.alpha_reg == 0.0
        assert model.l1_ratio == 0.5
        assert model.link_type == 'log'

    def test_custom_params(self):
        model = NegativeBinomialPiecewise(
            alpha=2.0, clipping_threshold=110,
            alpha_reg=0.1, l1_ratio=1.0, link_type='sqrt',
        )
        assert model.alpha == 2.0
        assert model.clipping_threshold == 110
        assert model.alpha_reg == 0.1
        assert model.l1_ratio == 1.0
        assert model.link_type == 'sqrt'

    def test_not_fitted_at_init(self):
        assert NegativeBinomialPiecewise().is_fitted_ is False

    def test_model_stats_none_at_init(self):
        assert NegativeBinomialPiecewise().model_stats_ is None


# ---------------------------------------------------------------------------
# TestPrepareTrainingData
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="sliding window pipeline"):
            NegativeBinomialPiecewise().prepare_training_data(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_returns_self(self, pipeline_output: dict):
        model = NegativeBinomialPiecewise()
        assert model.fit(pipeline_output['X'], pipeline_output['y_rul']) is model

    def test_fit_sets_is_fitted(self, fitted_model: NegativeBinomialPiecewise):
        assert fitted_model.is_fitted_ is True

    def test_fit_sets_model_stats_(self, fitted_model: NegativeBinomialPiecewise):
        assert fitted_model.model_stats_ is not None

    def test_fit_with_log_link(self, pipeline_output: dict):
        model = NegativeBinomialPiecewise(link_type='log')
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_

    def test_fit_with_identity_link(self, pipeline_output: dict):
        model = NegativeBinomialPiecewise(link_type='identity')
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_

    def test_fit_with_sqrt_link(self, pipeline_output: dict):
        model = NegativeBinomialPiecewise(link_type='sqrt')
        model.fit(pipeline_output['X'], pipeline_output['y_rul'])
        assert model.is_fitted_

    def test_fit_with_regularization(
        self, fitted_model_regularized: NegativeBinomialPiecewise
    ):
        assert fitted_model_regularized.is_fitted_

    def test_fit_handles_failure_gracefully(self, pipeline_output: dict):
        model = NegativeBinomialPiecewise()
        bad_X = pipeline_output['X'].copy()
        bad_X[0, 0] = np.nan
        with pytest.warns(RuntimeWarning):
            model.fit(bad_X, pipeline_output['y_rul'])
        assert model.is_fitted_ is False

    def test_fit_ignores_groups_kwarg(self, pipeline_output: dict):
        model = NegativeBinomialPiecewise()
        model.fit(
            pipeline_output['X'],
            pipeline_output['y_rul'],
            groups=pipeline_output['groups'],
        )
        assert model.is_fitted_


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_output_shape(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        assert fitted_model.predict(pipeline_output['X']).shape == (pipeline_output['X'].shape[0],)

    def test_output_non_negative(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        assert (fitted_model.predict(pipeline_output['X']) >= 0.0).all()

    def test_output_clipped_to_threshold(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        assert (fitted_model.predict(pipeline_output['X']) <= fitted_model.clipping_threshold).all()

    def test_returns_nan_if_not_fitted(self, pipeline_output: dict):
        assert np.isnan(NegativeBinomialPiecewise().predict(pipeline_output['X'])).all()

    def test_output_is_float_array(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        assert np.issubdtype(fitted_model.predict(pipeline_output['X']).dtype, np.floating)

    def test_predictions_non_trivial(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        assert fitted_model.predict(pipeline_output['X']).std() > 0.0


# ---------------------------------------------------------------------------
# TestPredictWithConfidence
# ---------------------------------------------------------------------------

class TestPredictWithConfidence:

    def test_returns_three_arrays(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        result = fitted_model.predict_with_confidence(pipeline_output['X'])
        assert isinstance(result, tuple) and len(result) == 3

    def test_output_shapes(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        n = pipeline_output['X'].shape[0]
        y_pred, ic_lower, ic_upper = fitted_model.predict_with_confidence(pipeline_output['X'])
        assert y_pred.shape == (n,) and ic_lower.shape == (n,) and ic_upper.shape == (n,)

    def test_ic_lower_leq_pred_leq_upper(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        y_pred, ic_lower, ic_upper = fitted_model.predict_with_confidence(pipeline_output['X'])
        assert (ic_lower <= y_pred + 1e-6).all()
        assert (y_pred <= ic_upper + 1e-6).all()

    def test_output_non_negative(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        y_pred, ic_lower, ic_upper = fitted_model.predict_with_confidence(pipeline_output['X'])
        assert (y_pred >= 0.0).all() and (ic_lower >= 0.0).all() and (ic_upper >= 0.0).all()

    def test_output_clipped(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        y_pred, ic_lower, ic_upper = fitted_model.predict_with_confidence(pipeline_output['X'])
        ct = fitted_model.clipping_threshold
        assert (y_pred <= ct).all() and (ic_lower <= ct).all() and (ic_upper <= ct).all()

    def test_raises_if_not_fitted(self, pipeline_output: dict):
        with pytest.raises(NotFittedError):
            NegativeBinomialPiecewise().predict_with_confidence(pipeline_output['X'])

    def test_raises_if_regularized(self, fitted_model_regularized: NegativeBinomialPiecewise, pipeline_output: dict):
        with pytest.raises(ValueError, match="alpha_reg=0"):
            fitted_model_regularized.predict_with_confidence(pipeline_output['X'])

    def test_wider_ci_at_higher_confidence(self, fitted_model: NegativeBinomialPiecewise, pipeline_output: dict):
        _, lo_90, hi_90 = fitted_model.predict_with_confidence(pipeline_output['X'], confidence=0.90)
        _, lo_99, hi_99 = fitted_model.predict_with_confidence(pipeline_output['X'], confidence=0.99)
        assert (hi_99 - lo_99).mean() > (hi_90 - lo_90).mean()