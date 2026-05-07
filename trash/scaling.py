"""Feature scaling for the RUL estimation pipeline (Nodo 1).

This module implements the first transformation stage of the pipeline —
robust scaling of sensor and setting columns while preserving metadata
columns (unit_number, time_in_cycles) intact for downstream windowing.

RobustScaler is used instead of StandardScaler because C-MAPSS sensor data
contains outliers from degradation events. RobustScaler centers by median
and scales by IQR, making it robust to extreme values in late-life cycles.

Design decisions:
    Selective scaling:
        Only feature columns (sensors and settings) are scaled. unit_number
        and time_in_cycles are metadata required by build_windows to construct
        sliding windows and counting process intervals — scaling them would
        corrupt the window structure and survival targets.

    DataFrame preservation:
        Input and output are both DataFrames with identical column structure.
        This ensures compatibility with build_windows which expects named
        columns to identify metadata vs feature columns.

    sklearn compatibility:
        Implements BaseEstimator and TransformerMixin so FeatureScaler can
        be used standalone or composed with other transformers if needed.
        fit() stores the scaler parameters from training data only, and
        transform() applies them to any subsequent data (validation, test,
        production) without refitting — preventing data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted


# Columns that must never be scaled — they are structural metadata
# required by build_windows for window construction and survival targets.
# evento is included because it is a binary indicator known at runtime
# (0 in production, 1 at the last cycle of train motors) and must pass
# through the pipeline without scaling distortion.
_PROTECTED_COLS: set[str] = {'unit_number', 'time_in_cycles', 'evento'}


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Robust scaler applied exclusively to sensor and setting columns.

    Wraps sklearn's RobustScaler to apply scaling only to feature columns
    (sensors and operational settings), leaving unit_number and time_in_cycles
    unchanged. The scaler is fitted on training data and applied to validation
    and test data using the same parameters to prevent leakage.

    Args:
        quantile_range: Tuple (q_min, q_max) defining the IQR range used
            by RobustScaler for scaling. Defaults to (25.0, 75.0) which
            corresponds to the standard IQR.

    Attributes:
        scaler_: Fitted RobustScaler instance. Available after fit().
        feature_cols_: List of column names that were scaled. Available
            after fit(). Used to guarantee consistent column alignment
            between fit() and transform() calls.
    """

    def __init__(self, quantile_range: tuple[float, float] = (25.0, 75.0)) -> None:
        self.quantile_range = quantile_range

    def fit(self, X: pd.DataFrame, y: object = None) -> 'FeatureScaler':
        """Fits the RobustScaler on the feature columns of X.

        Identifies feature columns as all columns not in _PROTECTED_COLS
        and fits a RobustScaler on those columns using training data only.

        Args:
            X: Input DataFrame containing metadata and feature columns.
                Must include time_in_cycles. unit_number is optional.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            Self.

        Raises:
            ValueError: If X contains no feature columns to scale (i.e.
                all columns are protected metadata columns).
        """
        self.feature_cols_: list[str] = [
            c for c in X.columns if c not in _PROTECTED_COLS
        ]

        if not self.feature_cols_:
            raise ValueError(
                "No feature columns found to scale. All columns are protected "
                f"metadata: {sorted(X.columns.tolist())}. "
                f"Protected columns: {_PROTECTED_COLS}."
            )

        self.scaler_: RobustScaler = RobustScaler(
            quantile_range=self.quantile_range
        )
        self.scaler_.fit(X[self.feature_cols_].to_numpy())

        return self

    def transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Scales feature columns using the fitted RobustScaler.

        Applies the scaling parameters from fit() to the feature columns
        of X. Protected columns (unit_number, time_in_cycles) are copied
        unchanged into the output DataFrame.

        Args:
            X: Input DataFrame. Must contain the same feature columns as
                the DataFrame used in fit(). Column order may differ —
                alignment is handled by name.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            DataFrame with the same columns and index as X, where feature
            columns are scaled and metadata columns are unchanged.

        Raises:
            NotFittedError: If fit() has not been called.
            ValueError: If X is missing any feature columns seen during fit().
        """
        check_is_fitted(self, ['scaler_', 'feature_cols_'])

        missing = [c for c in self.feature_cols_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"Columns present during fit but missing in transform: {missing}"
            )

        X_out = X.copy()
        X_out[self.feature_cols_] = self.scaler_.transform(
            X[self.feature_cols_].to_numpy()
        )

        return X_out

    def get_feature_names_out(
        self, input_features: np.ndarray | None = None
    ) -> np.ndarray:
        """Returns the column names of the output DataFrame.

        Args:
            input_features: Ignored. Column names are taken from fit().

        Returns:
            Array of all column names in the same order as the input DataFrame.
        """
        check_is_fitted(self, ['feature_cols_'])
        return np.array(self.feature_cols_)