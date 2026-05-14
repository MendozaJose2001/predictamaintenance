# ./src/models/negative_binomial.py

"""Negative Binomial GLM for RUL estimation.

This module implements a Negative Binomial GLM as a BaseRULModel subclass
compatible with the sliding window pipeline. Unlike the original
implementation, this version does not load per-motor CSVs — it receives the
feature matrix directly from the GGS loop after the dimensionality reduction
stage and flatten_windows().

Model rationale:
    The Negative Binomial GLM is the primary model of this project because:
    - It converges reliably on C-MAPSS FD001 under GroupKFold CV
    - It produces native confidence intervals via MLE (the only classical
      model in this project with rigorous individual prediction uncertainty)
    - It handles overdispersion in RUL count data via the alpha parameter
    - It is interpretable and computationally efficient

Link functions:
    Three link functions are supported:
    - log:      guarantees strictly positive predictions (recommended)
    - identity: linear predictor, may produce negative predictions
    - sqrt:     intermediate, common for count data with moderate range

Confidence intervals:
    Prediction intervals are derived from the GLM's linear predictor
    variance. For a new observation x, the variance of the linear
    predictor is Var(eta) = x^T * Cov(beta) * x, where Cov(beta) is
    the parameter covariance matrix from MLE. The CI on the response
    scale is obtained by back-transforming the CI on the link scale.
    This is only available when alpha_reg=0 (standard MLE fit).
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from statsmodels.base.elastic_net import RegularizedResultsWrapper

from src.models.base_model import BaseRULModel


class NegativeBinomialPiecewise(BaseRULModel, BaseEstimator, RegressorMixin):
    """Negative Binomial GLM for piecewise RUL estimation.

    Wraps a statsmodels Negative Binomial GLM as a sklearn-compatible
    estimator. Receives PCA-reduced window features from the dimensionality
    reduction stage and predicts clipped RUL.

    The target is expected to be pre-clipped by the pipeline's
    clipping_threshold — no additional clipping is applied to y during
    fit(). Predictions are clipped to clipping_threshold for consistency
    with the piecewise RUL convention.

    Supports elastic net regularization and three link functions.
    Confidence intervals are available via predict_with_confidence()
    when alpha_reg=0 (standard MLE fit).

    Args:
        alpha: Dispersion parameter of the Negative Binomial distribution.
            Controls overdispersion relative to Poisson. Defaults to 1.0.
        clipping_threshold: Maximum RUL value for prediction clipping.
            Should match the clipping_threshold used in the pipeline's
            build_windows(). Defaults to 125.
        alpha_reg: Regularization strength for elastic net. When 0.0,
            standard MLE fit is used and CIs are available. Defaults to 0.0.
        l1_ratio: Elastic net mixing parameter. 1.0 = pure L1 (Lasso),
            0.0 = pure L2 (Ridge). Only used when alpha_reg > 0.
            Defaults to 0.5.
        link_type: GLM link function. One of 'log', 'identity', or 'sqrt'.
            Log link guarantees strictly positive predictions and is
            recommended for RUL estimation. Defaults to 'log'.

    Attributes:
        model_stats_: Fitted statsmodels GLMResultsWrapper or
            RegularizedResultsWrapper. Available after fit().
        is_fitted_: Boolean flag indicating successful fit.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        clipping_threshold: int = 125,
        alpha_reg: float = 0.0,
        l1_ratio: float = 0.5,
        link_type: str = 'log',
    ) -> None:
        self.alpha              = alpha
        self.clipping_threshold = clipping_threshold
        self.alpha_reg          = alpha_reg
        self.l1_ratio           = l1_ratio
        self.link_type          = link_type
        self.is_fitted_: bool   = False
        self.model_stats_: GLMResultsWrapper | RegularizedResultsWrapper | None = None

    def _get_link(self) -> sm.families.links.Link:
        """Returns the statsmodels link function for link_type.

        Falls back to log link if link_type is unrecognized.

        Returns:
            statsmodels link function instance.
        """
        links: dict[str, sm.families.links.Link] = {
            'log':      sm.families.links.Log(),
            'identity': sm.families.links.Identity(),
            'sqrt':     sm.families.links.Sqrt(),
        }
        return links.get(self.link_type, sm.families.links.Log())

    def prepare_training_data(
        self,
        list_ids: np.ndarray,
    ) -> tuple:
        """Not implemented — this model uses the sliding window pipeline.

        NegativeBinomialPiecewise is designed for the sliding window pipeline
        and does not load per-motor CSVs. Call fit() directly with the output
        of flatten_windows() after the dimensionality reduction stage.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "NegativeBinomialPiecewise uses the sliding window pipeline. "
            "Call fit(X, y_rul) directly with the output of "
            "flatten_windows() after the dimensionality reduction stage. "
            "y_rul must be the pre-clipped RUL array from flatten_windows()."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object,
    ) -> 'NegativeBinomialPiecewise':
        """Fits the Negative Binomial GLM on PCA-reduced window features.

        The target y is expected to be pre-clipped by the pipeline
        (y_rul from flatten_windows). No additional clipping is applied
        during fit — the pipeline owns the clipping convention.

        When alpha_reg=0 (default), uses standard MLE fit and the
        resulting GLMResultsWrapper supports confidence interval
        computation via predict_with_confidence(). When alpha_reg>0,
        uses elastic net regularization (fit_regularized) and CIs
        are not available.

        On fitting failure, emits RuntimeWarning and sets is_fitted_=False,
        allowing the GGS loop to handle failed configurations gracefully.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
                Output of the dimensionality reduction stage after
                flatten_windows().
            y: Pre-clipped RUL array of shape (n_windows,).
                y_rul from flatten_windows() — already clipped to
                clipping_threshold by build_windows().
            **kwargs: Accepts but ignores 'groups' for GGS compatibility.

        Returns:
            Self.
        """
        try:
            X_arr = check_array(X)
            y_arr = np.asarray(y, dtype=float)

            X_with_const = sm.add_constant(X_arr, has_constant='add')

            family_nb = sm.families.NegativeBinomial(
                alpha=self.alpha,
                link=self._get_link(),
            )
            model = sm.GLM(y_arr, X_with_const, family=family_nb)

            if self.alpha_reg > 0:
                self.model_stats_ = model.fit_regularized(
                    method='elastic_net',
                    alpha=self.alpha_reg,
                    L1_wt=self.l1_ratio,
                    maxiter=500,
                )
            else:
                self.model_stats_ = model.fit()

            self.is_fitted_ = True

        except Exception as e:
            self.is_fitted_   = False
            self.model_stats_ = None
            warnings.warn(
                f"Fit failed for link='{self.link_type}', "
                f"alpha={self.alpha}, alpha_reg={self.alpha_reg}: "
                f"{type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates clipped RUL predictions for the given feature matrix.

        Returns NaN predictions if the model was not successfully fitted,
        allowing the GGS loop to handle failed configurations gracefully.
        Predictions are clipped to clipping_threshold for consistency
        with the piecewise training space.

        Args:
            X: Feature matrix of shape (n_windows, n_components).

        Returns:
            Predicted RUL array of shape (n_windows,), clipped to
            clipping_threshold. Contains NaN if the model is not fitted.
        """
        if not self.is_fitted_ or self.model_stats_ is None:
            return np.full(X.shape[0], np.nan)

        X_arr        = check_array(X)
        X_with_const = sm.add_constant(X_arr, has_constant='add')
        raw          = self.model_stats_.predict(X_with_const)

        return np.minimum(raw, self.clipping_threshold)

    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates RUL predictions with confidence intervals via MLE.

        Computes prediction intervals on the link scale using the parameter
        covariance matrix from MLE, then back-transforms to the response
        scale. Only available when alpha_reg=0 (standard MLE fit).

        The variance of the linear predictor for a new observation x is:
            Var(eta) = x^T * Cov(beta) * x
        The CI on the link scale is:
            [eta - z * sqrt(Var(eta)), eta + z * sqrt(Var(eta))]
        Back-transformed to the response scale via the inverse link function.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            confidence: Confidence level for the interval. Defaults to 0.95.

        Returns:
            Tuple of (y_pred, ic_lower, ic_upper) each of shape (n_windows,),
            clipped to clipping_threshold.

        Raises:
            NotFittedError: If fit() has not been called.
            ValueError: If model was fitted with alpha_reg > 0 (regularized
                fit does not produce a covariance matrix).
        """
        if not self.is_fitted_ or self.model_stats_ is None:
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using 'predict_with_confidence'."
            )

        if not isinstance(self.model_stats_, GLMResultsWrapper):
            raise ValueError(
                "Confidence intervals require standard MLE fit (alpha_reg=0). "
                "Regularized fits do not produce a parameter covariance matrix."
            )

        from scipy import stats as scipy_stats

        X_arr        = check_array(X)
        X_with_const = sm.add_constant(X_arr, has_constant='add')

        eta       = X_with_const @ self.model_stats_.params
        cov_beta  = self.model_stats_.cov_params()
        var_eta   = np.einsum('ij,jk,ik->i', X_with_const, cov_beta, X_with_const)
        se_eta    = np.sqrt(np.maximum(var_eta, 0.0))

        z         = scipy_stats.norm.ppf((1 + confidence) / 2)
        eta_lower = eta - z * se_eta
        eta_upper = eta + z * se_eta

        inv_link = self.model_stats_.family.link.inverse

        y_pred   = np.minimum(np.maximum(inv_link(eta),       0.0), self.clipping_threshold)
        ic_lower = np.minimum(np.maximum(inv_link(eta_lower), 0.0), self.clipping_threshold)
        ic_upper = np.minimum(np.maximum(inv_link(eta_upper), 0.0), self.clipping_threshold)

        return y_pred, ic_lower, ic_upper