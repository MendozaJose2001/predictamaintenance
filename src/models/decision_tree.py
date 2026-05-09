"""Decision Tree Regressor model for RUL estimation.

This module implements a DecisionTreeModel as a BaseRULModel subclass
compatible with the sliding window pipeline (Nodos 2-4). It receives
PCA-reduced window features directly from the GGS loop and predicts
clipped RUL as a point estimate.

Model rationale:
    Decision Tree is included as a classical ML regressor to:
    - Provide an interpretable non-linear baseline
    - Explore axis-aligned partitioning of PCA feature space
    - Serve as a building block for Random Forest (ensemble extension)
    - Document empirically whether tree-based regressors produce
      competitive RUL predictions on C-MAPSS FD001

Interface:
    Parallel to SVRModel — fit(X, y_rul) and predict(X) only.
    No survival analysis, no t_stop, no confidence curves.
    Predictions are clipped to [0, clipping_threshold].
"""

from typing import Literal

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_array

from src.models.base_model import BaseRULModel


# Tipo propio para max_features — coincide exactamente con sklearn
MaxFeatures = float | Literal['sqrt', 'log2'] | None


class DecisionTreeModel(BaseRULModel, BaseEstimator, RegressorMixin):
    """Decision Tree Regressor for RUL estimation.

    Wraps sklearn's DecisionTreeRegressor as a BaseRULModel-compatible
    estimator. Receives PCA-reduced window features from the sliding
    window pipeline (Nodo 4 output) and predicts clipped RUL.

    All sklearn hyperparameters are exposed for GGS optimization.

    Args:
        max_depth: Maximum depth of the tree. None means unlimited.
            Controls overfitting — deeper trees fit training data better
            but may not generalize. Defaults to None.
        min_samples_split: Minimum samples required to split an internal
            node. Higher values prevent overfitting. Defaults to 2.
        min_samples_leaf: Minimum samples required at a leaf node.
            Higher values smooth predictions. Defaults to 1.
        max_features: Number of features to consider for best split.
            None uses all features. 'sqrt' and 'log2' reduce variance.
            Defaults to None.
        clipping_threshold: Maximum RUL value for prediction clipping.
            Should match the clipping_threshold used in the pipeline.
            Defaults to 125.

    Attributes:
        model_: Fitted DecisionTreeRegressor. Available after fit().
        is_fitted_: Boolean flag indicating successful fit.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        clipping_threshold: int = 125,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features: MaxFeatures = max_features
        self.clipping_threshold = clipping_threshold
        self.is_fitted_: bool = False
        self.model_: DecisionTreeRegressor | None = None

    def prepare_training_data(self, list_ids: np.ndarray) -> tuple:
        """Not implemented — this model uses the sliding window pipeline.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "DecisionTreeModel uses the sliding window pipeline (Nodos 2-4). "
            "Call fit(X, y_rul) directly with the pipeline output."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object,
    ) -> 'DecisionTreeModel':
        """Fits the Decision Tree on PCA-reduced window features.

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

            self.model_ = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=42,
            )
            self.model_.fit(X_arr, y_arr)
            self.is_fitted_ = True

        except Exception as e:
            self.is_fitted_ = False
            self.model_ = None
            warnings.warn(
                f"Fit failed for DecisionTreeModel: {type(e).__name__}: {e}",
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
        raw = self.model_.predict(X_arr)
        return np.clip(raw, 0.0, float(self.clipping_threshold))