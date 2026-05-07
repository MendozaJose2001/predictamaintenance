"""Survival Tree model for RUL estimation (Nodo 5).

This module implements a scikit-survival SurvivalTree wrapped as a
BaseRULModel subclass. The Survival Tree serves as the first end-to-end
validation model for the new sliding window pipeline, establishing the
connection between Nodo 4 (PCA) output and Nodo 6 (S(t) → RUL conversion).

Model rationale:
    The Survival Tree was chosen as the initial Nodo 5 implementation because:
    - It always converges — no optimization failures like frailty models
    - It produces a valid survival function S(t) directly
    - It requires no hyperparameter tuning to produce non-trivial results
    - It is a natural stepping stone toward RSF (Random Survival Forest)

Survival target format:
    scikit-survival uses Surv(evento, t_stop) — one row per window.
    The counting process format Surv(t_start, t_stop, evento) is not
    supported by SurvivalTree. This is acceptable for the sliding window
    pipeline because:
    - Feature engineering (Nodos 1-3) already captures temporal history
    - GroupKFold prevents data leakage between motors
    - Independence violation (multiple windows per motor) is mitigated
      by the GGS structure and feature-based representation

Independence note:
    Multiple windows per motor violate the independence assumption of
    standard survival models. This is a known limitation of applying
    survival analysis to longitudinal sensor data without frailty terms.
    For this pipeline, the violation is accepted in exchange for:
    - Richer training data (one sample per cycle rather than per motor)
    - Compatibility with scikit-survival's standard survival format
    - Convergence guarantees absent in frailty-based approaches

prepare_training_data:
    Not implemented — this model uses the new sliding window pipeline
    (Nodos 1-4) rather than loading per-motor CSVs. The GGS manager
    for this model calls fit() directly with the Nodo 4 output.
"""

import warnings

import numpy as np
from sksurv.tree import SurvivalTree as _SurvivalTree
from sksurv.util import Surv
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from src.models.base_model import BaseRULModel


class SurvivalTreeModel(BaseRULModel, BaseEstimator, RegressorMixin):
    """Survival Tree for RUL estimation via survival function prediction.

    Wraps scikit-survival's SurvivalTree to produce survival function
    estimates S(t|X) from PCA-reduced window features. RUL predictions
    are obtained by finding the first time t where F(t) = 1 - S(t)
    exceeds the confidence_threshold.

    Args:
        max_depth: Maximum depth of the survival tree. Deeper trees
            capture more complex degradation patterns but risk overfitting.
            Defaults to 5.
        min_samples_leaf: Minimum number of windows required at each leaf.
            Higher values produce smoother survival functions. Defaults to 20.
        min_samples_split: Minimum number of windows required to split a
            node. Defaults to 40.
        confidence_threshold: Cumulative failure probability F(t) = 1 - S(t)
            at which the predicted failure cycle is declared. Lower values
            are more conservative (earlier maintenance). Defaults to 0.5.
        clipping_threshold: Maximum RUL value applied to predictions.
            Consistent with the piecewise linear RUL convention used across
            all models in this project. Defaults to 125.

    Attributes:
        model_: Fitted SurvivalTree instance. Available after fit().
        is_fitted_: Boolean flag indicating successful fit. Available
            after fit().
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        min_samples_split: int = 40,
        confidence_threshold: float = 0.5,
        clipping_threshold: int = 125,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.confidence_threshold = confidence_threshold
        self.clipping_threshold = clipping_threshold
        self.is_fitted_: bool = False

    def prepare_training_data(
        self,
        list_ids: np.ndarray
    ) -> tuple:
        """Not implemented — this model uses the sliding window pipeline.

        SurvivalTreeModel is designed for the new Nodo 1-4 pipeline and
        does not load per-motor CSVs. Call fit() directly with the output
        of flatten_windows() after DimReducer.transform().

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "SurvivalTreeModel uses the sliding window pipeline (Nodos 1-4). "
            "Call fit(X, y_surv) directly with the output of flatten_windows() "
            "after DimReducer.transform(). "
            "y_surv must be a structured array with dtype "
            "[('evento', bool), ('t_stop', float)]."
        )

    def _build_y_surv(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        """Validates and normalizes the survival target array.

        Accepts either a pre-built structured array with dtype
        [('evento', bool), ('t_stop', float)] or a 2D array of shape
        (n_samples, 2) with columns [evento, t_stop].

        Args:
            y: Survival target. Either structured array or 2D numeric array.

        Returns:
            Structured numpy array compatible with scikit-survival.

        Raises:
            ValueError: If y has an unexpected shape or dtype.
        """
        if y.dtype.names is not None:
            # Already a structured array — validate fields
            if 'evento' not in y.dtype.names or 't_stop' not in y.dtype.names:
                raise ValueError(
                    f"Structured array must have fields 'evento' and 't_stop'. "
                    f"Got: {y.dtype.names}"
                )
            return Surv.from_arrays(
                event=y['evento'].astype(bool),
                time=y['t_stop'].astype(float),
            )

        if y.ndim == 2 and y.shape[1] == 2:
            return Surv.from_arrays(
                event=y[:, 0].astype(bool),
                time=y[:, 1].astype(float),
            )

        raise ValueError(
            f"y must be a structured array with fields ['evento', 't_stop'] "
            f"or a 2D array of shape (n_samples, 2). Got shape {y.shape}."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object,
    ) -> 'SurvivalTreeModel':
        """Fits the Survival Tree on PCA-reduced window features.

        Args:
            X: Feature matrix of shape (n_windows, n_components). Output
                of DimReducer.transform() after flatten_windows().
            y: Survival target. Either:
                - Structured array with dtype [('evento', bool), ('t_stop', float)]
                - 2D array of shape (n_windows, 2) with columns [evento, t_stop]
            **kwargs: Accepts but ignores 'groups' for GGS loop compatibility.

        Returns:
            Self.
        """
        try:
            X_checked = check_array(X)
            y_surv = self._build_y_surv(y)

            self.model_: _SurvivalTree = _SurvivalTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
            )
            self.model_.fit(X_checked, y_surv)
            self.is_fitted_ = True

        except Exception as e:
            self.is_fitted_ = False
            warnings.warn(
                f"Fit failed: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

        return self

    def predict_survival_function(
        self,
        X: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns the survival function S(t) for each window in X.

        Args:
            X: Feature matrix of shape (n_windows, n_components).

        Returns:
            List of (times, probs) tuples — one per window. Each tuple
            contains the time axis and S(t) values for that window.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        check_is_fitted(self, ['model_'])
        X_checked = check_array(X)

        step_functions = self.model_.predict_survival_function(X_checked)

        result: list[tuple[np.ndarray, np.ndarray]] = []
        for fn in step_functions:
            times = fn.x
            probs = fn.y
            result.append((times, probs))

        return result

    def _survival_to_rul(
        self,
        survival_functions: list[tuple[np.ndarray, np.ndarray]],
        t_current: np.ndarray,
    ) -> np.ndarray:
        """Converts survival functions to clipped RUL predictions.

        For each window, finds the first time t where F(t) >= confidence_threshold
        and computes RUL = t_failure - t_current.

        Args:
            survival_functions: List of (times, probs) from predict_survival_function.
            t_current: Array of shape (n_windows,) with t_stop per window.

        Returns:
            RUL array of shape (n_windows,) clipped to clipping_threshold.
        """
        rul_list: list[float] = []

        for i, (times, probs) in enumerate(survival_functions):
            failure_probs = 1.0 - probs
            exceeds = np.where(failure_probs >= self.confidence_threshold)[0]

            t_failure = float(times[exceeds[0]]) if len(exceeds) > 0 else float(times[-1])
            rul = max(t_failure - float(t_current[i]), 0.0)
            rul_list.append(rul)

        raw_rul = np.array(rul_list)
        return np.minimum(raw_rul, self.clipping_threshold)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates clipped RUL predictions from the survival function.

        Requires t_stop values to compute RUL = t_failure - t_current.
        Since predict() only receives X, t_stop must be passed as the
        last column of X or separately via predict_with_time().

        Note: For the GGS loop, use predict_with_time() directly to pass
        t_stop explicitly. This method assumes t_stop is the last column
        of X for BaseRULModel compatibility.

        Args:
            X: Feature matrix of shape (n_windows, n_components + 1) where
                the last column is t_stop. Or (n_windows, n_components) if
                t_stop is not available — in that case t_current=0 and RUL
                equals the predicted failure time directly.

        Returns:
            Predicted RUL array of shape (n_windows,).
        """
        if not self.is_fitted_:
            return np.full(X.shape[0], np.nan)

        survival_functions = self.predict_survival_function(X)
        t_current = np.zeros(X.shape[0])
        return self._survival_to_rul(survival_functions, t_current)

    def predict_with_time(
        self,
        X: np.ndarray,
        t_stop: np.ndarray,
    ) -> np.ndarray:
        """Generates RUL predictions using explicit t_stop values.

        Preferred method for the GGS loop where t_stop is available
        from flatten_windows() output.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            t_stop: Array of shape (n_windows,) with the last observed
                cycle for each window.

        Returns:
            Predicted RUL array of shape (n_windows,) clipped to
            clipping_threshold.
        """
        if not self.is_fitted_:
            return np.full(X.shape[0], np.nan)

        survival_functions = self.predict_survival_function(X)
        return self._survival_to_rul(survival_functions, t_stop)