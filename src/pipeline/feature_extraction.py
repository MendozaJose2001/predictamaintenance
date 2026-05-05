"""Temporal feature extraction for the RUL estimation pipeline.

This module implements Nodo 3 of the pipeline — transforming the 3D window
tensor (n_windows, window_size, n_sensors) produced by Nodo 2 into a 2D
feature matrix (n_windows, n_features_total) suitable for PCA in Nodo 4.

Feature extraction is implemented directly in numpy rather than tsfresh.
Benchmarking showed that tsfresh overhead (DataFrame construction, internal
dispatching) dominated computation time (~15 min for 140 motors), making it
unsuitable for repeated GGS iterations. The numpy implementation computes
the same feature set vectorized over the window axis with negligible overhead.

Features are divided into two categories justified by the literature
(Vollert & Theissler 2021):

    Statistical descriptors (STATISTICAL_FEATURES):
        Capture the distribution of sensor values within each window.
        - median:      central tendency, robust to outliers
        - abs_energy:  total signal energy (sum of squares)
        - q25:         25th percentile
        - q75:         75th percentile (together with q25 gives implicit IQR)

        Note: minimum, maximum and standard_deviation were removed —
        min/max are redundant after RobustScaler normalization, and std
        is redundant given the implicit IQR from the two quantiles.
        Q25 and Q75 are kept separate rather than computing IQR = Q75 - Q25
        because PCA can derive the IQR direction implicitly while retaining
        more distributional information from both quantiles individually.

    Trend and correlation (TREND_FEATURES):
        Capture the temporal dynamics of sensor degradation within each window.
        - slope:              linear trend slope (rate of degradation)
        - rvalue:             Pearson r with time (proxy for linear monotonicity)
        - autocorr_lag_{1,2,3}: short-term temporal dependency
        - partial_autocorr_lag_{1,2,3}: direct lag dependency (indirect removed)

TODO: Evaluate adding Spearman correlation with time as a custom feature
    for monotonicity detection. rvalue (Pearson) is the current proxy.
    Pending discussion with advisor on whether linear monotonicity is
    sufficient or explicit rank-based monotonicity (Spearman) is required.
"""

import numpy as np
import pandas as pd

from src.pipeline.windowing import MotorData, MotorWindows


# ---------------------------------------------------------------------------
# Feature name constants
# ---------------------------------------------------------------------------

STATISTICAL_FEATURES: list[str] = [
    'median',
    'abs_energy',
    'q25',
    'q75',
]

TREND_FEATURES: list[str] = [
    'slope',
    'rvalue',
    'autocorr_lag_1',
    'autocorr_lag_2',
    'autocorr_lag_3',
    'partial_autocorr_lag_1',
    'partial_autocorr_lag_2',
    'partial_autocorr_lag_3',
]

ALL_FEATURES: list[str] = STATISTICAL_FEATURES + TREND_FEATURES


# ---------------------------------------------------------------------------
# Numpy feature computation
# ---------------------------------------------------------------------------

def _compute_features(
    X: np.ndarray,
    features: list[str],
) -> np.ndarray:
    """Computes the requested features over the window axis of X.

    All computations are fully vectorized over n_windows and n_sensors —
    no Python loops over windows or sensors.

    Args:
        X: Window tensor of shape (n_windows, window_size, n_sensors).
        features: List of feature names to compute. Must be a subset of
            ALL_FEATURES. Order determines the column order in the output.

    Returns:
        Feature matrix of shape (n_windows, len(features) * n_sensors).
        Columns are ordered as: feature_1_sensor_1, feature_1_sensor_2, ...,
        feature_2_sensor_1, feature_2_sensor_2, ..., etc.

    Raises:
        ValueError: If any feature name is not recognized.
    """
    n_windows, window_size, n_sensors = X.shape
    valid = set(ALL_FEATURES)
    unknown = [f for f in features if f not in valid]
    if unknown:
        raise ValueError(
            f"Unknown features: {unknown}. Valid options: {sorted(valid)}"
        )

    # Time axis for trend computations: 0, 1, ..., window_size-1
    # Shape: (window_size,) broadcast to (n_windows, window_size, n_sensors)
    t = np.arange(window_size, dtype=float)
    t_mean = t.mean()
    t_centered = t - t_mean
    t_var = np.sum(t_centered ** 2)  # scalar

    feature_blocks: list[np.ndarray] = []

    for feat in features:

        if feat == 'median':
            # (n_windows, n_sensors)
            block = np.median(X, axis=1)

        elif feat == 'abs_energy':
            # sum of squares along window axis
            block = np.sum(X ** 2, axis=1)

        elif feat == 'q25':
            block = np.quantile(X, 0.25, axis=1)

        elif feat == 'q75':
            block = np.quantile(X, 0.75, axis=1)

        elif feat == 'slope':
            # OLS slope = cov(t, x) / var(t) — vectorized
            x_mean = X.mean(axis=1, keepdims=True)  # (n_windows, 1, n_sensors)
            x_centered = X - x_mean                  # (n_windows, window_size, n_sensors)
            cov = np.sum(t_centered[:, np.newaxis] * x_centered, axis=1) / window_size
            block = cov / (t_var / window_size)

        elif feat == 'rvalue':
            # Pearson r = cov(t, x) / (std(t) * std(x))
            x_mean = X.mean(axis=1, keepdims=True)
            x_centered = X - x_mean
            x_std = X.std(axis=1)  # (n_windows, n_sensors)
            t_std = t_centered.std()  # scalar
            cov = np.sum(t_centered[:, np.newaxis] * x_centered, axis=1) / window_size
            # Avoid division by zero for constant windows — np.where evaluates
            # both branches before selecting, so we suppress the warning explicitly.
            denom = t_std * x_std
            with np.errstate(invalid='ignore', divide='ignore'):
                block = np.where(denom > 0, cov / denom, 0.0)

        elif feat.startswith('autocorr_lag_'):
            lag = int(feat.split('_')[-1])
            # Pearson correlation between x[t] and x[t+lag]
            if lag >= window_size:
                block = np.zeros((n_windows, n_sensors))
            else:
                x1 = X[:, :-lag, :]   # (n_windows, window_size-lag, n_sensors)
                x2 = X[:, lag:, :]
                x1_mean = x1.mean(axis=1, keepdims=True)
                x2_mean = x2.mean(axis=1, keepdims=True)
                x1_c = x1 - x1_mean
                x2_c = x2 - x2_mean
                n = window_size - lag
                cov = np.sum(x1_c * x2_c, axis=1) / n
                std1 = x1.std(axis=1)
                std2 = x2.std(axis=1)
                denom = std1 * std2
                with np.errstate(invalid='ignore', divide='ignore'):
                    block = np.where(denom > 0, cov / denom, 0.0)

        elif feat.startswith('partial_autocorr_lag_'):
            lag = int(feat.split('_')[-1])
            # Partial autocorrelation via Yule-Walker equations.
            # Computed sensor by sensor — Yule-Walker requires 1D input.
            # This is the only non-fully-vectorized computation.
            block = np.zeros((n_windows, n_sensors))
            for s in range(n_sensors):
                for w in range(n_windows):
                    series = X[w, :, s]
                    series_std = series.std()
                    if series_std < 1e-10:
                        # Constant series — partial autocorr undefined, set to 0
                        block[w, s] = 0.0
                        continue
                    # Build autocorrelation vector r[1..lag]
                    r = np.array([
                        np.corrcoef(series[:-k], series[k:])[0, 1]
                        if k < window_size else 0.0
                        for k in range(1, lag + 1)
                    ])
                    # Toeplitz matrix R of autocorrelations r[0..lag-1]
                    # Note: k=0 case handled separately to avoid series[:-0]
                    # which returns an empty array instead of the full series.
                    r0 = np.zeros(lag)
                    r0[0] = 1.0  # autocorrelation at lag 0 is always 1
                    for k in range(1, lag):
                        if k < window_size:
                            r0[k] = np.corrcoef(series[:-k], series[k:])[0, 1]
                    R = np.zeros((lag, lag))
                    for i in range(lag):
                        for j in range(lag):
                            idx = abs(i - j)
                            R[i, j] = r0[idx] if idx < lag else 0.0
                    R[np.diag_indices(lag)] = 1.0
                    try:
                        phi = np.linalg.solve(R, r)
                        block[w, s] = phi[-1]
                    except np.linalg.LinAlgError:
                        block[w, s] = 0.0
        else:
            # Should never reach here due to validation above
            raise ValueError(f"Unhandled feature: {feat}")

        feature_blocks.append(block)

    # Concatenate all feature blocks: (n_windows, len(features) * n_sensors)
    return np.concatenate(feature_blocks, axis=1)


def _build_feature_names(
    features: list[str],
    sensor_names: list[str],
) -> list[str]:
    """Builds the column names for the feature matrix.

    Column order matches the output of _compute_features:
        feature_1__sensor_1, feature_1__sensor_2, ...,
        feature_2__sensor_1, feature_2__sensor_2, ..., etc.

    Uses the double-underscore separator convention from tsfresh for
    compatibility with downstream tooling.

    Args:
        features: List of feature names in the same order as _compute_features.
        sensor_names: List of sensor column names.

    Returns:
        List of strings of length len(features) * len(sensor_names).
    """
    return [
        f'{sensor}__{feat}'
        for feat in features
        for sensor in sensor_names
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_window_features(
    motor_windows: MotorWindows,
    features: list[str] | None = None,
) -> MotorWindows:
    """Extracts temporal features from sliding windows using numpy.

    Transforms the 3D window tensor in each motor's MotorData into a 2D
    feature matrix by computing the requested features over the window axis.
    All other fields (t_start, t_stop, evento, y_rul) are preserved unchanged.

    All feature computations are vectorized over n_windows and n_sensors.
    Partial autocorrelation requires a Yule-Walker solve per window per sensor
    and is the only non-fully-vectorized computation.

    Args:
        motor_windows: Output of build_windows (Nodo 2). Each motor entry
            must have X_windows of shape (n_windows, window_size, n_sensors).
        features: List of feature names to compute. Must be a subset of
            ALL_FEATURES. Defaults to ALL_FEATURES (statistical + trend).
            Pass STATISTICAL_FEATURES or TREND_FEATURES to use a subset.

    Returns:
        MotorWindows with the same structure as the input, but X_windows
        replaced by a 2D array of shape (n_windows, n_features_total) and
        feature_names updated to reflect the computed feature × sensor names.

    Raises:
        ValueError: If any motor's X_windows is not 3-dimensional.
        ValueError: If any motor's windows are not in chronological order.
        ValueError: If any feature name is not in ALL_FEATURES.
    """
    if features is None:
        features = ALL_FEATURES

    result: MotorWindows = {}

    for motor_id, data in motor_windows.items():
        X_windows = data['X_windows']

        if X_windows.ndim != 3:
            raise ValueError(
                f"Motor {motor_id}: X_windows must be 3D "
                f"(n_windows, window_size, n_sensors), got shape {X_windows.shape}."
            )

        if not np.all(np.diff(data['t_stop']) >= 0):
            raise ValueError(
                f"Motor {motor_id}: windows are not in chronological order. "
                f"t_stop must be non-decreasing. Check build_windows output."
            )

        sensor_names: list[str] = list(data['feature_names'])
        X_features = _compute_features(X_windows, features)
        new_feature_names = _build_feature_names(features, sensor_names)

        result[motor_id] = MotorData(
            X_windows=X_features,
            t_start=data['t_start'],
            t_stop=data['t_stop'],
            evento=data['evento'],
            y_rul=data['y_rul'],
            feature_names=new_feature_names,
        )

    return result