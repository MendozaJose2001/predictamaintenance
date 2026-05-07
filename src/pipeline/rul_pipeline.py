"""RUL pipeline orchestrator — Nodos 2 through 4.

This module implements RULPipeline, the single entry point for transforming
raw multi-motor DataFrames into feature matrices ready for survival or
regression model training and prediction.

The pipeline encapsulates three sequential transformation stages:

    Nodo 2 — build_windows:
        Sliding window construction. Produces MotorWindows with 3D
        feature tensors and counting process survival targets.

    Nodo 3 — extract_window_features:
        Numpy-native feature extraction (statistical + trend).
        Transforms 3D tensors to 2D feature matrices per motor.

    Nodo 4 — DimReducer:
        Internal RobustScaler (on extracted features) followed by PCA.
        Fitted on training data only to prevent leakage.

Design decision — Nodo 1 (FeatureScaler) removed:
    Empirical verification showed that a RobustScaler on raw sensors before
    windowing is redundant when DimReducer already applies RobustScaler on
    the extracted features before PCA. Without Nodo 1:
    - No NaN or Inf in PCA output
    - Better variance distribution across PCA components (PC1=0.51 vs 0.70)
    - Higher cumulative explained variance (0.91 vs 0.95 is comparable)
    - Simpler pipeline with one fewer stateful transformer
    The DimReducer's internal RobustScaler handles all scaling needs before
    the PCA decomposition.

Usage pattern in GGS loop:
    # Training fold
    X_tr, y_rul_tr, t_stop_tr, evento_tr, groups_tr = (
        pipeline.fit_transform(X_df_train, y_df_train)
    )

    # Validation fold
    X_val, y_rul_val, t_stop_val, evento_val, groups_val = (
        pipeline.transform(X_df_val, y_df_val)
    )

Data separation contract:
    X_df must contain: unit_number, time_in_cycles, evento, sensors/settings
    X_df must NOT contain: RUL (separated into y_df before calling pipeline)
    y_df must contain: unit_number, time_in_cycles, RUL

    The pipeline reconstructs y_rul by merging t_stop (from windowing)
    with y_df on (unit_number, time_in_cycles).
"""

import time

import numpy as np
import pandas as pd

from src.pipeline.feature_extraction import ALL_FEATURES, extract_window_features
from src.pipeline.windowing import MotorWindows, build_windows, flatten_windows
from src.pipeline.dim_reduction import DimReducer


# Type alias for the pipeline output tuple
PipelineOutput = tuple[
    np.ndarray,  # X          (n_windows, n_components)
    np.ndarray,  # y_rul      (n_windows,) clipped
    np.ndarray,  # t_stop     (n_windows,)
    np.ndarray,  # evento     (n_windows,)
    np.ndarray,  # groups     (n_windows,) motor_id per window
]


class RULPipeline:
    """Orchestrates Nodos 2-4 of the RUL feature engineering pipeline.

    Transforms a multi-motor time-series DataFrame into a flat feature
    matrix ready for survival or regression model training. Encapsulates
    build_windows (Nodo 2), extract_window_features (Nodo 3), and
    DimReducer (Nodo 4).

    Only DimReducer is stateful — it is fitted on training data in
    fit_transform() and applied without refitting in transform(), preventing
    data leakage from validation data into the PCA parameters.

    Args:
        window_size: Number of consecutive cycles per sliding window.
            GGS hyperparameter. Defaults to 30.
        clipping_threshold: Maximum RUL value for y_rul clipping.
            GGS hyperparameter. Defaults to 125.
        n_components: Number of PCA components to retain.
            GGS hyperparameter. Defaults to 10.
        verbose: If True, prints timing for each pipeline stage.
            Defaults to False.

    Attributes:
        reducer_: Fitted DimReducer. Available after fit_transform().
        is_fitted_: True after fit_transform() completes successfully.
    """

    def __init__(
        self,
        window_size: int = 30,
        clipping_threshold: int = 125,
        n_components: int = 10,
        verbose: bool = False,
    ) -> None:
        self.window_size = window_size
        self.clipping_threshold = clipping_threshold
        self.n_components = n_components
        self.verbose = verbose
        self.is_fitted_: bool = False

    def _log(self, message: str, elapsed: float) -> None:
        """Prints a timing message when verbose=True."""
        if self.verbose:
            print(f"  {message}: {elapsed:.2f}s")

    def _build_and_extract(self, X_df: pd.DataFrame) -> MotorWindows:
        """Applies Nodos 2 and 3 to the input DataFrame.

        Args:
            X_df: Input DataFrame with sensors/settings, unit_number,
                time_in_cycles, and evento columns.

        Returns:
            MotorWindows with 2D feature matrices after extraction.
        """
        t0 = time.perf_counter()
        motor_windows = build_windows(
            df=X_df,
            window_size=self.window_size,
            clipping_threshold=self.clipping_threshold,
        )
        self._log("Nodo 2 (windowing)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        motor_features = extract_window_features(
            motor_windows=motor_windows,
            features=ALL_FEATURES,
        )
        self._log("Nodo 3 (features)", time.perf_counter() - t0)

        return motor_features

    def _reconstruct_y_rul(
        self,
        groups: np.ndarray,
        t_stop: np.ndarray,
        y_df: pd.DataFrame | None,
    ) -> np.ndarray:
        """Reconstructs y_rul by merging t_stop with RUL from y_df.

        Args:
            groups: Motor ID per window, shape (n_windows,).
            t_stop: Last cycle of each window, shape (n_windows,).
            y_df: DataFrame with columns [unit_number, time_in_cycles, RUL].
                If None, returns NaN array (production mode).

        Returns:
            y_rul array of shape (n_windows,), clipped to
            clipping_threshold. NaN if y_df is None.
        """
        if y_df is None:
            return np.full(len(groups), np.nan)

        df_merge = pd.DataFrame({
            'unit_number': groups,
            'time_in_cycles': t_stop.astype(int),
        })
        merged = df_merge.merge(
            y_df[['unit_number', 'time_in_cycles', 'RUL']],
            on=['unit_number', 'time_in_cycles'],
            how='left',
        )
        return np.minimum(
            merged['RUL'].to_numpy(dtype=float),
            float(self.clipping_threshold),
        )

    def fit_transform(
        self,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame | None = None,
    ) -> PipelineOutput:
        """Fits the pipeline on training data and returns transformed output.

        Applies Nodos 2 and 3 (stateless) then fits and applies DimReducer
        (Nodo 4) on training data to produce the flat feature matrix.

        Args:
            X_df: Training DataFrame without RUL column. Must contain
                unit_number, time_in_cycles, evento, and sensor/setting
                columns.
            y_df: DataFrame with columns [unit_number, time_in_cycles, RUL].
                Used to reconstruct y_rul after windowing. If None,
                y_rul is NaN (production mode without ground truth).

        Returns:
            PipelineOutput tuple of:
                X:       (n_windows, n_components) float array
                y_rul:   (n_windows,) clipped RUL, NaN if y_df is None
                t_stop:  (n_windows,) last cycle of each window
                evento:  (n_windows,) event indicator
                groups:  (n_windows,) motor_id per window
        """
        # Nodos 2 + 3 — stateless: windowing and feature extraction
        motor_features = self._build_and_extract(X_df)

        # Nodo 4 — fit DimReducer on training features then transform
        t0 = time.perf_counter()
        self.reducer_: DimReducer = DimReducer(n_components=self.n_components)
        self.reducer_.fit(motor_features)
        motor_reduced = self.reducer_.transform(motor_features)
        self._log("Nodo 4 (PCA)", time.perf_counter() - t0)

        self.is_fitted_ = True

        # Flatten to 2D arrays
        X, t_start, t_stop, evento, _, groups = flatten_windows(motor_reduced)
        y_rul = self._reconstruct_y_rul(groups, t_stop, y_df)

        return X, y_rul, t_stop, evento, groups

    def transform(
        self,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame | None = None,
    ) -> PipelineOutput:
        """Transforms new data using the fitted DimReducer.

        Applies Nodos 2 and 3 (stateless) then applies the DimReducer
        fitted in fit_transform() without refitting.

        Args:
            X_df: DataFrame without RUL column. Same structure as the
                DataFrame used in fit_transform().
            y_df: DataFrame with columns [unit_number, time_in_cycles, RUL].
                If None, y_rul is NaN (production or test mode).

        Returns:
            PipelineOutput tuple — same structure as fit_transform().

        Raises:
            RuntimeError: If fit_transform() has not been called.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "RULPipeline is not fitted. Call fit_transform() before transform()."
            )

        # Nodos 2 + 3 — stateless
        motor_features = self._build_and_extract(X_df)

        # Nodo 4 — transform only (reducer already fitted)
        t0 = time.perf_counter()
        motor_reduced = self.reducer_.transform(motor_features)
        self._log("Nodo 4 (PCA)", time.perf_counter() - t0)

        # Flatten to 2D arrays
        X, t_start, t_stop, evento, _, groups = flatten_windows(motor_reduced)
        y_rul = self._reconstruct_y_rul(groups, t_stop, y_df)

        return X, y_rul, t_stop, evento, groups

    def explained_variance_ratio(self) -> np.ndarray:
        """Returns the PCA explained variance ratio from DimReducer.

        Returns:
            Array of shape (n_components,).

        Raises:
            RuntimeError: If fit_transform() has not been called.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "RULPipeline is not fitted. Call fit_transform() first."
            )
        return self.reducer_.explained_variance_ratio()