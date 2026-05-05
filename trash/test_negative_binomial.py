import warnings
from unittest.mock import patch

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from statsmodels.base.elastic_net import RegularizedResultsWrapper

from src.models.negative_binomial import NegativeBinomialPiecewise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Returns a seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generates a minimal synthetic dataset for fitting tests.

    Produces integer non-negative targets compatible with the Negative
    Binomial distribution, and a feature matrix with moderate variance.

    Returns:
        Tuple of (X, y) where X has shape (100, 5) and y has shape (100,).
    """
    X = rng.normal(loc=0.0, scale=1.0, size=(100, 5))
    y = rng.integers(low=1, high=100, size=100).astype(float)
    return X, y


@pytest.fixture
def fitted_model(synthetic_data: tuple[np.ndarray, np.ndarray]) -> NegativeBinomialPiecewise:
    """Returns a fitted NegativeBinomialPiecewise model with default parameters."""
    X, y = synthetic_data
    model = NegativeBinomialPiecewise(clipping_threshold=80)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for NegativeBinomialPiecewise initialization."""

    def test_default_parameters(self):
        """Default parameters must match the documented defaults."""
        model = NegativeBinomialPiecewise()
        assert model.alpha == 1.0
        assert model.clipping_threshold == 125
        assert model.alpha_reg == 0.0
        assert model.l1_ratio == 0.5
        assert model.link_type == 'log'

    def test_custom_parameters(self):
        """Custom parameters must be stored as provided."""
        model = NegativeBinomialPiecewise(
            alpha=1.5,
            clipping_threshold=110,
            alpha_reg=0.05,
            l1_ratio=1.0,
            link_type='sqrt'
        )
        assert model.alpha == 1.5
        assert model.clipping_threshold == 110
        assert model.alpha_reg == 0.05
        assert model.l1_ratio == 1.0
        assert model.link_type == 'sqrt'

    def test_is_fitted_false_at_init(self):
        """Model must not be marked as fitted before calling fit()."""
        model = NegativeBinomialPiecewise()
        assert model.is_fitted_ is False

    def test_model_stats_none_at_init(self):
        """model_stats_ must be None before calling fit()."""
        model = NegativeBinomialPiecewise()
        assert model.model_stats_ is None


# ---------------------------------------------------------------------------
# _get_link
# ---------------------------------------------------------------------------

class TestGetLink:
    """Tests for the link function factory."""

    def test_log_link(self):
        """'log' must return a Log link instance."""
        model = NegativeBinomialPiecewise(link_type='log')
        assert isinstance(model._get_link(), sm.families.links.Log)

    def test_identity_link(self):
        """'identity' must return an Identity link instance."""
        model = NegativeBinomialPiecewise(link_type='identity')
        assert isinstance(model._get_link(), sm.families.links.Identity)

    def test_sqrt_link(self):
        """'sqrt' must return a Sqrt link instance."""
        model = NegativeBinomialPiecewise(link_type='sqrt')
        assert isinstance(model._get_link(), sm.families.links.Sqrt)

    def test_unknown_link_falls_back_to_log(self):
        """An unrecognized link_type must fall back to the Log link."""
        model = NegativeBinomialPiecewise(link_type='unknown')
        assert isinstance(model._get_link(), sm.families.links.Log)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class TestFit:
    """Tests for the fit method."""

    def test_fit_marks_model_as_fitted(self, synthetic_data):
        """After successful fit, is_fitted_ must be True."""
        X, y = synthetic_data
        model = NegativeBinomialPiecewise(clipping_threshold=80)
        model.fit(X, y)
        assert model.is_fitted_ is True

    def test_fit_sets_model_stats(self, synthetic_data):
        """After successful fit, model_stats_ must not be None."""
        X, y = synthetic_data
        model = NegativeBinomialPiecewise(clipping_threshold=80)
        model.fit(X, y)
        assert model.model_stats_ is not None

    def test_fit_without_regularization_returns_glm_wrapper(self, synthetic_data):
        """Fit without regularization must produce a GLMResultsWrapper."""
        X, y = synthetic_data
        model = NegativeBinomialPiecewise(alpha_reg=0.0, clipping_threshold=80)
        model.fit(X, y)
        assert isinstance(model.model_stats_, GLMResultsWrapper)

    def test_fit_with_regularization_returns_regularized_wrapper(self, synthetic_data):
        """Fit with regularization must produce a RegularizedResultsWrapper."""
        X, y = synthetic_data
        model = NegativeBinomialPiecewise(alpha_reg=0.05, l1_ratio=1.0, clipping_threshold=80)
        model.fit(X, y)
        assert isinstance(model.model_stats_, RegularizedResultsWrapper)

    def test_fit_returns_self(self, synthetic_data):
        """fit() must return self for sklearn pipeline compatibility."""
        X, y = synthetic_data
        model = NegativeBinomialPiecewise(clipping_threshold=80)
        result = model.fit(X, y)
        assert result is model

    def test_fit_failure_marks_model_as_not_fitted(self, synthetic_data):
        """On fitting failure, is_fitted_ must remain False.

        Mocks statsmodels GLM.fit to raise a controlled exception, isolating
        the error-handling logic in the except block.
        """
        X, y = synthetic_data
        model = NegativeBinomialPiecewise()
        with patch('statsmodels.genmod.generalized_linear_model.GLM.fit',
                   side_effect=RuntimeError("mocked failure")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
        assert model.is_fitted_ is False

    def test_fit_failure_sets_model_stats_to_none(self, synthetic_data):
        """On fitting failure, model_stats_ must be None.

        Mocks statsmodels GLM.fit to raise a controlled exception, isolating
        the error-handling logic in the except block.
        """
        X, y = synthetic_data
        model = NegativeBinomialPiecewise()
        with patch('statsmodels.genmod.generalized_linear_model.GLM.fit',
                   side_effect=RuntimeError("mocked failure")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
        assert model.model_stats_ is None

    def test_fit_failure_emits_runtime_warning(self, synthetic_data):
        """On fitting failure, a RuntimeWarning must be emitted.

        Mocks statsmodels GLM.fit to raise a controlled exception, isolating
        the error-handling logic in the except block.
        """
        X, y = synthetic_data
        model = NegativeBinomialPiecewise()
        with patch('statsmodels.genmod.generalized_linear_model.GLM.fit',
                   side_effect=RuntimeError("mocked failure")):
            with pytest.warns(RuntimeWarning, match="Fit failed"):
                model.fit(X, y)

    def test_clipping_applied_to_target(self, rng):
        """Targets above clipping_threshold must be clipped before fitting."""
        X = rng.normal(size=(100, 3))
        y = np.full(100, 200.0)  # all values above any reasonable threshold
        model = NegativeBinomialPiecewise(clipping_threshold=80)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        # If clipping wasn't applied the model would train on 200 and
        # predictions would be far above 80. After clipping, predictions
        # must be at or below the threshold.
        preds = model.predict(X)
        assert (preds <= 80).all()


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    """Tests for the predict method."""

    def test_predict_before_fit_returns_nan(self):
        """Calling predict before fit must return an array of NaN values."""
        X = np.random.default_rng(0).normal(size=(10, 5))
        model = NegativeBinomialPiecewise()
        preds = model.predict(X)
        assert np.all(np.isnan(preds))

    def test_predict_before_fit_returns_correct_shape(self):
        """NaN array returned before fit must match the number of samples."""
        X = np.random.default_rng(0).normal(size=(15, 5))
        model = NegativeBinomialPiecewise()
        preds = model.predict(X)
        assert preds.shape == (15,)

    def test_predict_returns_correct_shape(self, fitted_model, synthetic_data):
        """Predictions must have the same number of rows as the input."""
        X, _ = synthetic_data
        preds = fitted_model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_clips_to_threshold(self, fitted_model, synthetic_data):
        """All predictions must be at or below clipping_threshold."""
        X, _ = synthetic_data
        preds = fitted_model.predict(X)
        assert (preds <= fitted_model.clipping_threshold).all()

    def test_predict_log_link_returns_positive_values(self, synthetic_data):
        """Log link predictions must always be strictly positive."""
        X, y = synthetic_data
        model = NegativeBinomialPiecewise(link_type='log', clipping_threshold=80)
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds > 0).all()

    def test_predict_returns_float_array(self, fitted_model, synthetic_data):
        """Predictions must be returned as a floating point numpy array."""
        X, _ = synthetic_data
        preds = fitted_model.predict(X)
        assert preds.dtype.kind == 'f'