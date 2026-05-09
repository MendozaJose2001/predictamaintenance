#./src/pipeline/dim_reduction.py

"""Dimensionality reduction for the RUL estimation pipeline (Nodo 4).

This module implements PCA-based dimensionality reduction as the fourth
transformation stage of the pipeline. The input is a MotorWindows dictionary
with 2D feature matrices (n_windows, n_features) from Nodo 3. The output
preserves the same MotorWindows hierarchy with X_windows reduced to
(n_windows, n_components).

Design decisions:
    Internal RobustScaler before PCA:
        Features extracted by Nodo 3 (median, abs_energy, slope, etc.) have
        very different magnitudes. abs_energy = sum(x²) over window_size steps
        produces values ~100x larger than other features like slope (~0.5).
        Without scaling, PCA is biased toward high-magnitude features and
        ignores subtle but informative ones. A RobustScaler is applied
        internally before PCA to equalize feature contributions.
        RobustScaler is preferred over StandardScaler because abs_energy
        and autocorrelation features can have heavy-tailed distributions.
        This scaler is fitted exclusively on training data (fit() call)
        and applied to validation data (transform() call) — no leakage.

    Global PCA:
        PCA is fitted on the concatenation of all motor feature matrices
        in the input MotorWindows after internal scaling. This produces a
        single global projection shared across all motors — consistent with
        the standard approach in the PHM literature (Alomari et al. 2023).

    Hierarchy preservation:
        After global fitting and transformation, the reduced feature matrix
        is redistributed back to each motor using the original window counts.
        The MotorWindows structure is preserved intact — t_start, t_stop,
        evento, and y_rul pass through unchanged.

    Feature name update:
        After PCA, the original sensor feature names no longer apply.
        feature_names is updated to ['PC_1', 'PC_2', ..., 'PC_n_components']
        to reflect the new column semantics.

    sklearn backend:
        Uses sklearn.decomposition.PCA internally. If a different backend
        is needed in the future (e.g. IncrementalPCA for large datasets,
        KernelPCA for non-linear reduction, or UMAP), only this file needs
        to be modified — the MotorWindows interface remains unchanged.

    n_components as GGS hyperparameter:
        n_components controls the trade-off between information retention
        and model complexity. It is expected to be explored in the GGS
        grid with a range of 5-30 components as suggested by the advisor.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted

from src.pipeline.windowing import MotorData, MotorWindows


class DimReducer(BaseEstimator):
    """PCA-based dimensionality reducer for the RUL pipeline (Nodo 4).

    Applies RobustScaler followed by PCA on the concatenated feature
    matrices of all motors in the input MotorWindows. The RobustScaler
    equalizes feature magnitudes before PCA to prevent high-magnitude
    features (e.g. abs_energy) from dominating the principal components.

    Both the scaler and PCA are fitted exclusively on training data and
    applied to validation/test data without refitting — no data leakage.

    Args:
        n_components: Number of principal components to retain. Acts as a
            GGS hyperparameter — typical range is 5 to 30. Defaults to 10.

    Attributes:
        scaler_: Fitted RobustScaler instance. Available after fit().
        pca_: Fitted sklearn PCA instance. Available after fit().
        pc_names_: List of PC column names ['PC_1', ..., 'PC_n']. Available
            after fit().
    """

    def __init__(self, n_components: int = 10) -> None:
        self.n_components = n_components

    def fit(
        self,
        motor_windows: MotorWindows,
        y: object = None,
    ) -> 'DimReducer':
        """Fits RobustScaler and PCA on the concatenated feature matrices.

        Concatenates X_windows from all motors, fits a RobustScaler to
        equalize feature magnitudes, then fits a global PCA on the scaled
        features. Both transformers are stored for use in transform().

        Args:
            motor_windows: MotorWindows from Nodo 3. Each motor entry must
                have X_windows of shape (n_windows, n_features) — 2D, as
                produced by extract_window_features.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            Self.

        Raises:
            ValueError: If any motor's X_windows is not 2-dimensional.
            ValueError: If n_components exceeds the number of features.
        """
        for motor_id, data in motor_windows.items():
            if data['X_windows'].ndim != 2:
                raise ValueError(
                    f"Motor {motor_id}: X_windows must be 2D "
                    f"(n_windows, n_features), got shape {data['X_windows'].shape}. "
                    f"Nodo 3 must be applied before Nodo 4."
                )

        # Concatenate all motors for global fitting
        X_all = np.concatenate(
            [data['X_windows'] for data in motor_windows.values()],
            axis=0
        )

        n_features = X_all.shape[1]
        if self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} exceeds the number of "
                f"features n_features={n_features}."
            )

        # Internal RobustScaler — equalizes feature magnitudes before PCA.
        # Fitted on training data only to prevent leakage.
        self.scaler_: RobustScaler = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X_all)

        # Global PCA on scaled features
        self.pca_: PCA = PCA(n_components=self.n_components)
        self.pca_.fit(X_scaled)

        self.pc_names_: list[str] = [
            f'PC_{i+1}' for i in range(self.n_components)
        ]

        return self

    def transform(
        self,
        motor_windows: MotorWindows,
        y: object = None,
    ) -> MotorWindows:
        """Transforms each motor's feature matrix using scaler and PCA.

        Applies the RobustScaler fitted in fit() followed by the global PCA
        to each motor's X_windows independently, then reconstructs the
        MotorWindows dictionary with reduced feature matrices. All other
        fields (t_start, t_stop, evento, y_rul) pass through unchanged.
        feature_names is updated to PC names.

        Args:
            motor_windows: MotorWindows from Nodo 3. Must contain the same
                features as the MotorWindows used in fit().
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            MotorWindows with X_windows of shape (n_windows, n_components)
            and feature_names updated to ['PC_1', ..., 'PC_n_components'].

        Raises:
            NotFittedError: If fit() has not been called.
            ValueError: If any motor's X_windows is not 2-dimensional.
        """
        check_is_fitted(self, ['scaler_', 'pca_', 'pc_names_'])

        result: MotorWindows = {}

        for motor_id, data in motor_windows.items():
            if data['X_windows'].ndim != 2:
                raise ValueError(
                    f"Motor {motor_id}: X_windows must be 2D "
                    f"(n_windows, n_features), got shape {data['X_windows'].shape}."
                )

            # Apply scaler then PCA — same order as fit()
            X_scaled = self.scaler_.transform(data['X_windows'])
            X_reduced = self.pca_.transform(X_scaled)

            result[motor_id] = MotorData(
                X_windows=X_reduced,
                t_start=data['t_start'],
                t_stop=data['t_stop'],
                evento=data['evento'],
                y_rul=data['y_rul'],
                feature_names=self.pc_names_,
            )

        return result

    def explained_variance_ratio(self) -> np.ndarray:
        """Returns the explained variance ratio of each principal component.

        Useful for selecting n_components — the cumulative sum indicates
        how much total variance is retained with the current n_components.

        Returns:
            Array of shape (n_components,) with the fraction of variance
            explained by each PC in descending order.

        Raises:
            NotFittedError: If fit() has not been called.
        """
        check_is_fitted(self, ['scaler_', 'pca_'])
        return self.pca_.explained_variance_ratio_