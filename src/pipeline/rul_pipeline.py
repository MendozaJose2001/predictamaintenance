"""RUL estimation pipeline orchestrator.

This module implements the RULPipeline class — the single entry point for
transforming a raw multi-motor DataFrame into a MotorWindows dictionary
ready for survival model training or prediction.

The pipeline orchestrates Nodo 2 (windowing) and Nodo 3 (feature extraction)
in sequence, preserving the MotorWindows format throughout so that downstream
components (PCA, survival models, GGS loop) can consume it directly without
format conversions.

Pipeline stages:
    Nodo 2 — build_windows:
        Transforms the flat time-series DataFrame into per-motor sliding
        windows with counting process survival targets and clipped RUL labels.

    Nodo 3 — extract_window_features:
        Replaces the 3D window tensor with a 2D tsfresh feature matrix.
        All survival targets and RUL labels pass through unchanged.

Nodo 0 (column removal) is handled externally by DatasetManager before
calling this pipeline. Nodo 1 (RobustScaler) and Nodo 4 (PCA) are applied
inside the GGS fold loop, not here, to prevent data leakage between folds.

Usage:
    pipeline = RULPipeline(window_size=30, clipping_threshold=125)
    motor_windows = pipeline.transform(df)

    # Flatten for GGS
    X, t_start, t_stop, evento, y_rul, groups = flatten_windows(motor_windows)
"""

import time

import numpy as np
import pandas as pd

from src.pipeline.feature_extraction import ALL_FEATURES, extract_window_features
from src.pipeline.windowing import MotorWindows, build_windows


class RULPipeline:
    """Orchestrates the RUL feature engineering pipeline (Nodos 2 and 3).

    Transforms a multi-motor time-series DataFrame into a MotorWindows
    dictionary containing tsfresh feature matrices and survival targets,
    ready for PCA and survival model fitting.

    Nodo 1 (scaling) and Nodo 4 (PCA) are intentionally excluded from this
    class — they must be fitted inside each GGS fold using only training
    data to prevent leakage from validation data into the transformers.

    Args:
        window_size: Number of consecutive cycles per sliding window.
            Must be >= 1. Early cycles (fewer than window_size) are
            discarded per motor. Larger values capture longer degradation
            history at the cost of fewer windows per motor.
        clipping_threshold: Maximum RUL value applied to y_rul targets.
            Consistent with the piecewise linear RUL convention used
            across all models in this project.
        n_jobs: Number of parallel jobs for tsfresh feature extraction.
            Defaults to 1 (serial). Increase on machines with many cores
            if feature extraction is a bottleneck.
        verbose: If True, prints timing information for each pipeline
            stage. Useful for profiling during development.
            Defaults to False.
    """

    def __init__(
        self,
        window_size: int,
        clipping_threshold: int,
        verbose: bool = False,
    ) -> None:
        self.window_size = window_size
        self.clipping_threshold = clipping_threshold
        self.verbose = verbose

    def transform(self, df: pd.DataFrame) -> MotorWindows:
        """Transforms a multi-motor DataFrame into a MotorWindows dictionary.

        Applies Nodo 2 (windowing) and Nodo 3 (feature extraction) in
        sequence. The output MotorWindows contains one entry per motor with
        a 2D tsfresh feature matrix and the associated survival targets.

        The DataFrame must contain at minimum:
            - time_in_cycles: cycle index, >= 1, strictly increasing per motor
            - Feature columns: sensor and setting values to be windowed
            Optional columns (used if present, filled with defaults if absent):
            - unit_number: motor identifier (default: 0 for single-motor)
            - RUL: remaining useful life per row (default: NaN)
            - evento: binary event indicator (default: 0, all censored)

        Args:
            df: Input DataFrame. May contain one or multiple motors.
                Column structure must match the C-MAPSS clean dataset format
                produced by DatasetManager.

        Returns:
            MotorWindows dictionary keyed by motor_id. Each entry contains:
                X_windows: 2D array (n_windows, n_features) of tsfresh features.
                t_start:   counting process interval start per window.
                t_stop:    counting process interval end per window.
                evento:    event indicator per window.
                y_rul:     clipped RUL per window.
                feature_names: tsfresh feature column names.
        """
        # Nodo 2 — sliding windows
        t0 = time.perf_counter()
        motor_windows = build_windows(
            df=df,
            window_size=self.window_size,
            clipping_threshold=self.clipping_threshold,
        )
        t1 = time.perf_counter()
        if self.verbose:
            n_motors = len(motor_windows)
            n_windows = sum(
                d['X_windows'].shape[0] for d in motor_windows.values()
            )
            print(
                f"[Nodo 2] {n_motors} motors, {n_windows} windows "
                f"— {t1 - t0:.2f}s"
            )

        # Nodo 3 — numpy feature extraction
        motor_features = extract_window_features(
            motor_windows=motor_windows,
            features=ALL_FEATURES,
        )
        t2 = time.perf_counter()
        if self.verbose:
            n_features = next(iter(motor_features.values()))['X_windows'].shape[1]
            print(
                f"[Nodo 3] {n_features} features per window "
                f"— {t2 - t1:.2f}s"
            )
            print(f"[Total]  {t2 - t0:.2f}s")

        return motor_features