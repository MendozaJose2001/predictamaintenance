#./src/models/random_forest.py

"""Random Forest Regressor model for RUL estimation.

This module implements a RandomForestModel as a BaseRULModel subclass
compatible with the sliding window pipeline. It receives PCA-reduced
window features directly from the GGS loop and predicts clipped RUL
as a point estimate.

Model rationale:
    Random Forest is included as an ensemble extension of DecisionTreeModel:
    - Reduces variance through bagging of multiple trees
    - More robust to overfitting than a single decision tree
    - Provides implicit feature importance estimates
    - Expected to outperform DecisionTreeModel on unseen data

Interface:
    Parallel to DecisionTreeModel and SVRModel — fit(X, y_rul) and
    predict(X) only. No survival analysis, no t_stop, no confidence curves.
    Predictions are clipped to [0, clipping_threshold].
"""

from typing import Literal

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_array

from src.models.base_model import BaseRULModel


# Custom type for max_features — RandomForestRegressor does not accept None
MaxFeatures = float | Literal['sqrt', 'log2']


class RandomForestModel(BaseRULModel, BaseEstimator, RegressorMixin):
    """Random Forest Regressor for RUL estimation.

    Wraps sklearn's RandomForestRegressor as a BaseRULModel-compatible
    estimator. Receives PCA-reduced window features from the dimensionality
    reduction stage and predicts clipped RUL.

    All sklearn hyperparameters are exposed for GGS optimization.

    Args:
        n_estimators: Number of trees in the forest. Higher values
            reduce variance but increase compute. Defaults to 100.
        max_depth: Maximum depth of each tree. None means unlimited.
            Defaults to None.
        min_samples_split: Minimum samples required to split an internal
            node. Defaults to 2.
        min_samples_leaf: Minimum samples required at a leaf node.
            Defaults to 1.
        max_features: Number of features to consider for best split.
            'sqrt' is the standard for Random Forest. Defaults to 1.0.
        clipping_threshold: Maximum RUL value for prediction clipping.
            Should match the clipping_threshold used in the pipeline.
            Defaults to 125.

    Attributes:
        model_: Fitted RandomForestRegressor. Available after fit().
        is_fitted_: Boolean flag indicating successful fit.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = 1.0,
        clipping_threshold: int = 125,
    ) -> None:
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features: MaxFeatures = max_features
        self.clipping_threshold = clipping_threshold
        self.is_fitted_: bool   = False
        self.model_: RandomForestRegressor | None = None

    def prepare_training_data(self, list_ids: np.ndarray) -> tuple:
        """Not implemented — this model uses the sliding window pipeline.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "RandomForestModel uses the sliding window pipeline. "
            "Call fit(X, y_rul) directly with the pipeline output."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object,
    ) -> 'RandomForestModel':
        """Fits the Random Forest on PCA-reduced window features.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            y: Pre-clipped RUL array of shape (n_windows,).
            **kwargs: Accepts but ignores 'groups' for GGS compatibility.

        Returns:
            Self.
        """
        try:
            X_arr = check_array(X)
            y_arr = np.asarray(y, dtype=float)

            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=42,
                n_jobs=1,  # GGS manages parallelism externally
            )
            self.model_.fit(X_arr, y_arr)
            self.is_fitted_ = True

        except Exception as e:
            self.is_fitted_ = False
            self.model_     = None
            warnings.warn(
                f"Fit failed for RandomForestModel: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates clipped RUL predictions for the given feature matrix.

        Args:
            X: Feature matrix of shape (n_windows, n_components).

        Returns:
            Predicted RUL array of shape (n_windows,), clipped to
            [0, clipping_threshold]. NaN array if not fitted.
        """
        if not self.is_fitted_ or self.model_ is None:
            return np.full(X.shape[0], np.nan)

        X_arr = check_array(X)
        raw   = self.model_.predict(X_arr)
        return np.clip(raw, 0.0, float(self.clipping_threshold))