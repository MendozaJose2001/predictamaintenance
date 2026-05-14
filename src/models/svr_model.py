#./src/models/svr_model.py

"""Support Vector Regression model for RUL estimation.

This module implements an SVR model as a BaseRULModel subclass compatible
with the sliding window pipeline. It receives PCA-reduced window features
directly from the GGS loop after the dimensionality reduction stage and
flatten_windows().

Model rationale:
    SVR is included in this project as a classical ML regressor to:
    - Provide a non-probabilistic baseline for comparison with NB
    - Explore kernel-based non-linear regression on PCA features
    - Document empirically whether classical ML regressors produce
      competitive RUL predictions without survival analysis framework

Confidence intervals:
    SVR does not produce native confidence intervals. Predictions are
    point estimates only. This is a known limitation documented in the
    project's methodological guidelines — SVR is evaluated on MAE/RMSE
    and S-Score only, without individual prediction uncertainty.
"""

import warnings
from typing import Literal, cast

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
from sklearn.utils.validation import check_array, check_is_fitted

from src.models.base_model import BaseRULModel


class SVRModel(BaseRULModel, BaseEstimator, RegressorMixin):
    """Support Vector Regression for piecewise RUL estimation.

    Wraps sklearn's SVR as a BaseRULModel-compatible estimator.
    Receives PCA-reduced window features from the dimensionality
    reduction stage and predicts clipped RUL.

    All hyperparameters are exposed for GGS optimization. The kernel
    parameter controls the feature space — RBF is recommended as the
    default for non-linear degradation patterns.

    Args:
        kernel: SVR kernel type. One of 'rbf', 'linear', or 'poly'.
            RBF is recommended for non-linear degradation patterns.
            Linear is faster and interpretable. Poly can capture
            polynomial degradation curves. Defaults to 'rbf'.
        C: Regularization parameter. Higher values allow less margin
            violation (less regularization). Defaults to 1.0.
        epsilon: Width of the epsilon-insensitive tube. Predictions
            within epsilon of the true value incur no penalty.
            Defaults to 0.1.
        gamma: Kernel coefficient for 'rbf' and 'poly'. 'scale' uses
            1 / (n_features * X.var()), 'auto' uses 1 / n_features.
            Can also be a float for manual tuning. Defaults to 'scale'.
        degree: Degree of the polynomial kernel. Only used when
            kernel='poly'. Defaults to 3.
        clipping_threshold: Maximum RUL value for prediction clipping.
            Should match the clipping_threshold used in build_windows().
            Defaults to 125.

    Attributes:
        model_: Fitted sklearn SVR instance. Available after fit().
        is_fitted_: Boolean flag indicating successful fit.
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: str | float = 'scale',
        degree: int = 3,
        clipping_threshold: int = 125,
    ) -> None:
        self.kernel             = kernel
        self.C                  = C
        self.epsilon            = epsilon
        self.gamma              = gamma
        self.degree             = degree
        self.clipping_threshold = clipping_threshold
        self.is_fitted_: bool   = False
        self.model_: SVR | None = None

    def prepare_training_data(
        self,
        list_ids: np.ndarray,
    ) -> tuple:
        """Not implemented — this model uses the sliding window pipeline.

        SVRModel is designed for the sliding window pipeline and does not
        load per-motor CSVs. Call fit() directly with the output of
        flatten_windows() after the dimensionality reduction stage.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "SVRModel uses the sliding window pipeline. "
            "Call fit(X, y_rul) directly with the output of "
            "flatten_windows() after the dimensionality reduction stage. "
            "y_rul must be the pre-clipped RUL array from flatten_windows()."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object,
    ) -> 'SVRModel':
        """Fits the SVR on PCA-reduced window features.

        The target y is expected to be pre-clipped by the pipeline
        (y_rul from flatten_windows). No additional clipping is applied
        during fit — the pipeline owns the clipping convention.

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

            self.model_ = SVR(
                kernel=cast(Literal['linear', 'poly', 'rbf', 'sigmoid'], self.kernel),
                C=self.C,
                epsilon=self.epsilon,
                gamma=cast(float | Literal['scale', 'auto'], self.gamma),
                degree=self.degree,
            )
            self.model_.fit(X_arr, y_arr)
            self.is_fitted_ = True

        except Exception as e:
            self.is_fitted_ = False
            self.model_     = None
            warnings.warn(
                f"Fit failed for kernel='{self.kernel}', C={self.C}, "
                f"epsilon={self.epsilon}: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates clipped RUL predictions for the given feature matrix.

        Returns NaN predictions if the model was not successfully fitted,
        allowing the GGS loop to handle failed configurations gracefully.
        Predictions are clipped to clipping_threshold for consistency
        with the piecewise training space and non-negativity is enforced.

        Args:
            X: Feature matrix of shape (n_windows, n_components).

        Returns:
            Predicted RUL array of shape (n_windows,), clipped to
            [0, clipping_threshold]. Contains NaN if not fitted.
        """
        if not self.is_fitted_ or self.model_ is None:
            return np.full(X.shape[0], np.nan)

        X_arr = check_array(X)
        raw   = self.model_.predict(X_arr)

        return np.clip(raw, 0.0, self.clipping_threshold)