"""MAD-based robust scaler compatible with sklearn pipelines.

Implements a robust scaler using median and MAD (Median Absolute Deviation)
for centering and scaling, equivalent to R's scale() with mad(). This is
required for frailtyPenal convergence — RobustScaler uses IQR which produces
different scaling factors that cause istop=2 in frailtyPenal.

Mathematical equivalence with R:
    center: median(x)
    scale:  mad(x) = median(|x - median(x)|)

Note: RobustScaler uses IQR (Q3 - Q1) for scaling, not MAD. The two
produce different results and frailtyPenal is sensitive to this difference.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MADScaler(BaseEstimator, TransformerMixin):
    """Scales features using median and MAD, equivalent to R's mad() scaling.

    Centers each feature by its median and scales by its MAD (median absolute
    deviation). Features with MAD=0 are set to zero. This is the scaling
    method required for frailtyPenal convergence in R.

    Attributes:
        medians_: Array of medians computed during fit, shape (n_features,).
        mads_: Array of MADs computed during fit, shape (n_features,).
        feature_names_in_: Column names if X was a DataFrame during fit.
    """

    def fit(self, X: np.ndarray, y: object = None) -> 'MADScaler':
        """Computes median and MAD for each feature.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            Self.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
            X_arr = X.to_numpy().astype(float)
        else:
            X_arr = np.array(X, dtype=float)

        self.medians_ = np.median(X_arr, axis=0)
        self.mads_ = np.median(np.abs(X_arr - self.medians_), axis=0)

        return self

    def transform(self, X: np.ndarray, y: object = None) -> pd.DataFrame:
        """Scales X using the median and MAD computed during fit.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            Scaled DataFrame with the same column names as the input
            (or generic names if X was an ndarray).
        """
        check_is_fitted(self, ['medians_', 'mads_'])

        if isinstance(X, pd.DataFrame):
            col_names = list(X.columns)
            X_arr = X.to_numpy().astype(float)
        else:
            X_arr = np.array(X, dtype=float)
            col_names = (
                list(self.feature_names_in_)
                if hasattr(self, 'feature_names_in_')
                else [f'col_{i}' for i in range(X_arr.shape[1])]
            )

        X_scaled = np.where(
            self.mads_ > 0,
            (X_arr - self.medians_) / self.mads_,
            0.0
        )

        return pd.DataFrame(X_scaled, columns=col_names)

    def get_feature_names_out(
        self, input_features: np.ndarray | None = None
    ) -> np.ndarray:
        """Returns feature names for the scaled output.

        Args:
            input_features: Ignored. Feature names are taken from fit.

        Returns:
            Array of feature name strings.
        """
        check_is_fitted(self, ['medians_'])
        if hasattr(self, 'feature_names_in_'):
            return self.feature_names_in_
        return np.array([f'col_{i}' for i in range(len(self.medians_))])