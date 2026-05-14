#./src/models/xgb_model.py

"""XGBoost Regressor model for RUL estimation.

This module implements XGBModel as a BaseRULModel subclass compatible
with the sliding window pipeline. It receives PCA-reduced
window features directly from the GGS loop and predicts clipped RUL
as a point estimate.

Model rationale:
    XGBoost is included as a gradient boosting alternative to the
    bagging-based RandomForestModel:
    - Learns sequentially correcting errors of previous trees (boosting)
      vs independently (bagging), typically achieving better
      generalization with fewer estimators
    - Handles non-linearities without kernel selection (unlike SVR)
    - Makes no distributional assumptions on residuals (unlike NB)
    - Native L1/L2 regularization via reg_alpha/reg_lambda
    - Computationally efficient on tabular PCA-reduced data via
      histogram-based tree construction (tree_method='hist')

Fixed internal parameters (not GGS hyperparameters):
    objective:    'reg:squarederror' — MSE minimization, consistent
                  with RMSE/MAE metrics used across the project.
    tree_method:  'hist' — histogram-based split finding. 3-5x faster
                  than 'exact' for moderate datasets (~15k-20k windows)
                  with negligible precision loss. Essential for GGS
                  efficiency across thousands of configurations.
    n_jobs:       1 — GGS manages parallelism externally via joblib.
                  Setting n_jobs>1 internally would conflict with the
                  loky backend used in parallel GGS mode.
    random_state: 42 — consistent with all other models in the project.

Interface:
    Parallel to RandomForestModel and DecisionTreeModel — fit(X, y_rul)
    and predict(X) only. No survival analysis, no t_stop, no confidence
    curves. Predictions are clipped to [0, clipping_threshold].
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array

from src.models.base_model import BaseRULModel


class XGBModel(BaseRULModel, BaseEstimator, RegressorMixin):
    """XGBoost Regressor for RUL estimation.

    Wraps xgboost.XGBRegressor as a BaseRULModel-compatible estimator.
    Receives PCA-reduced window features from the sliding window pipeline
    (dimensionality reduction stage output) and predicts clipped RUL.

    All GGS-relevant hyperparameters are exposed as constructor arguments.
    Internal parameters (objective, tree_method, n_jobs, random_state)
    are fixed — see module docstring for rationale.

    Args:
        n_estimators: Number of boosting rounds. Controls model capacity —
            too few causes underfitting, too many risks overfitting without
            early stopping. Defaults to 200.
        learning_rate: Shrinkage factor applied to each tree's contribution.
            Lower values require more estimators but improve generalization.
            Interacts directly with n_estimators. Defaults to 0.1.
        max_depth: Maximum depth of each tree. Shallower trees produce more
            conservative models. With PCA-reduced features, excessive depth
            is rarely beneficial. Defaults to 5.
        subsample: Fraction of training windows sampled per boosting round.
            Introduces stochasticity and reduces overfitting. 1.0 disables
            subsampling. Defaults to 1.0.
        colsample_bytree: Fraction of PCA components used per tree.
            Introduces feature-level randomness analogous to max_features
            in RandomForest. Defaults to 1.0.
        reg_lambda: L2 regularization term on leaf weights. Smooths
            predictions and prevents overfitting on PCA components.
            XGBoost default is 1.0. Defaults to 1.0.
        min_child_weight: Minimum sum of instance weights required in a
            child node. Higher values make the model more conservative
            by pruning splits on small, potentially noisy nodes.
            Defaults to 1.
        clipping_threshold: Maximum RUL value for prediction clipping.
            Should match the clipping_threshold used in the pipeline.
            Defaults to 125.

    Attributes:
        model_: Fitted XGBRegressor instance. Available after fit().
        is_fitted_: Boolean flag indicating successful fit.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        clipping_threshold: int = 125,
    ) -> None:
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda       = reg_lambda
        self.min_child_weight = min_child_weight
        self.clipping_threshold = clipping_threshold
        self.is_fitted_: bool = False
        self.model_           = None

    def prepare_training_data(self, list_ids: np.ndarray) -> tuple:
        """Not implemented — this model uses the sliding window pipeline.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "XGBModel uses the sliding window pipeline. "
            "Call fit(X, y_rul) directly with the pipeline output."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object,
    ) -> 'XGBModel':
        """Fits the XGBoost regressor on PCA-reduced window features.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            y: Pre-clipped RUL array of shape (n_windows,).
            **kwargs: Accepts but ignores 'groups' for GGS compatibility.

        Returns:
            Self.
        """
        try:
            from xgboost import XGBRegressor

            X_arr = check_array(X)
            y_arr = np.asarray(y, dtype=float)

            self.model_ = XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight,
                objective='reg:squarederror',
                tree_method='hist',
                n_jobs=1,
                random_state=42,
                verbosity=0,
            )
            self.model_.fit(X_arr, y_arr)
            self.is_fitted_ = True

        except Exception as e:
            self.is_fitted_ = False
            self.model_ = None
            warnings.warn(
                f"Fit failed for XGBModel: {type(e).__name__}: {e}",
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