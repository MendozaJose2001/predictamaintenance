#./src/pipeline/feature_extraction.py

"""Temporal feature extraction for the RUL estimation pipeline.

This module implements Nodo 3 of the pipeline — transforming the 3D window
tensor (n_windows, window_size, n_sensors) produced by Nodo 2 into a 2D
feature matrix (n_windows, n_features_total) suitable for PCA in Nodo 4.

Feature extraction is implemented directly in numpy rather than tsfresh.
Benchmarking showed that tsfresh overhead (DataFrame construction, internal
dispatching) dominated computation time (~15 min for 140 motors), making it
unsuitable for repeated GGS iterations. The numpy implementation computes
the same feature set vectorized over the window axis with negligible overhead.

Features are organized into five domain-specific lists, enabling systematic
evaluation of feature set candidates (Fase 1 of the feature selection
methodology). Each domain captures a qualitatively different aspect of
sensor degradation:

    STATISTICAL_FEATURES:
        Distribution of sensor values within each window.
        - median:   central tendency, robust to outliers
        - rms:      root mean square — normalized energy (replaces abs_energy)
        - q25:      25th percentile
        - q75:      75th percentile

        Note: rms replaces abs_energy. Both capture signal energy but rms
        is normalized by window_size (rms = sqrt(mean(x²))), avoiding the
        numerical explosion of abs_energy (sum(x²) ~ 10⁹ for raw sensors).
        Empirically confirmed: same PCA cumvar, dramatically better scale.

    TREND_FEATURES:
        Temporal dynamics and monotonicity of sensor degradation.
        - slope:            OLS linear trend slope (rate of degradation)
        - rvalue:           Pearson r with time (linear monotonicity proxy)
        - autocorr_lag_1:   short-term temporal dependency (lag 1)
        - autocorr_lag_2:   short-term temporal dependency (lag 2)

    MEMORY_FEATURES:
        Long-range temporal persistence — alternatives to partial_autocorr.
        - runs_ratio:  proportion of sign-change runs in first differences.
                       Low → persistent direction; High → chaotic/noisy.
                       O(n) per window per sensor.
        - hurst_rs:    Hurst exponent via rescaled range R/S.
                       H > 0.5 → persistence; H < 0.5 → antipersistence.
                       O(n) per window per sensor.

        Note: partial_autocorr (Yule-Walker) was evaluated and discarded
        due to O(n²) cost per window per sensor — prohibitive for 200
        bootstrap iterations × 140 motors × 15,814 windows.

    FREQUENCY_FEATURES:
        Spectral information within each window.
        - fft_coef_1:  absolute FFT coefficient at frequency 1 cycle/window
        - fft_coef_2:  absolute FFT coefficient at frequency 2 cycles/window
        - fft_coef_3:  absolute FFT coefficient at frequency 3 cycles/window

        Note: fft_coef_0 is excluded — it equals window_size × mean(x),
        which is redundant with median/mean already in STATISTICAL_FEATURES.
        Frequencies > 3 are excluded for window_size=20 — dominated by noise.

    COMPLEXITY_FEATURES:
        Signal irregularity and unpredictability.
        - permutation_entropy:  Shannon entropy of ordinal patterns (tau=1, dim=3).
                                Low → regular/predictable; High → complex/disordered.
                                Alomari et al. (2023) identify this as dominant
                                in PC2 for speed and pressure sensors.

Pre-defined feature set combinations for Fase 1 analysis:
    SET_A: mean, std, rms, slope, rvalue                    (baseline minimal)
    SET_B: STATISTICAL_FEATURES + TREND_FEATURES            (pipeline reference)
    SET_C: SET_B + MEMORY_FEATURES                          (+memory)
    SET_D: SET_B + FREQUENCY_FEATURES                       (+frequency)
    SET_E: SET_B + COMPLEXITY_FEATURES                      (+complexity)
"""

import numpy as np

from src.pipeline.windowing import MotorData, MotorWindows


# ---------------------------------------------------------------------------
# Feature name constants — organized by domain
# ---------------------------------------------------------------------------

STATISTICAL_FEATURES: list[str] = [
    'median',
    'rms',
    'q25',
    'q75',
]

TREND_FEATURES: list[str] = [
    'slope',
    'rvalue',
    'autocorr_lag_1',
    'autocorr_lag_2',
]

MEMORY_FEATURES: list[str] = [
    'runs_ratio',
    'hurst_rs',
]

FREQUENCY_FEATURES: list[str] = [
    'fft_coef_1',
    'fft_coef_2',
    'fft_coef_3',
]

COMPLEXITY_FEATURES: list[str] = [
    'permutation_entropy',
]

# Legacy alias — all original features for backward compatibility
# (includes partial_autocorr and abs_energy from original implementation)
LEGACY_FEATURES: list[str] = [
    'median', 'abs_energy', 'q25', 'q75',
    'slope', 'rvalue',
    'autocorr_lag_1', 'autocorr_lag_2', 'autocorr_lag_3',
    'partial_autocorr_lag_1', 'partial_autocorr_lag_2', 'partial_autocorr_lag_3',
]

# All new features combined
ALL_FEATURES: list[str] = (
    STATISTICAL_FEATURES
    + TREND_FEATURES
    + MEMORY_FEATURES
    + FREQUENCY_FEATURES
    + COMPLEXITY_FEATURES
)

# ---------------------------------------------------------------------------
# Pre-defined feature set candidates for Fase 1 analysis
# ---------------------------------------------------------------------------

SET_A: list[str] = ['mean', 'std', 'rms', 'slope', 'rvalue']

SET_B: list[str] = STATISTICAL_FEATURES + TREND_FEATURES

SET_C: list[str] = SET_B + MEMORY_FEATURES

SET_D: list[str] = SET_B + FREQUENCY_FEATURES

SET_E: list[str] = SET_B + COMPLEXITY_FEATURES

FEATURE_SETS: dict[str, list[str]] = {
    'A': SET_A,
    'B': SET_B,
    'C': SET_C,
    'D': SET_D,
    'E': SET_E,
}


# ---------------------------------------------------------------------------
# Numpy feature computation
# ---------------------------------------------------------------------------

def _compute_features(
    X: np.ndarray,
    features: list[str],
) -> np.ndarray:
    """Computes the requested features over the window axis of X.

    Fully vectorized over n_windows and n_sensors for all features except
    permutation_entropy (which requires Python loops over windows and sensors
    for ordinal pattern construction).

    Args:
        X: Window tensor of shape (n_windows, window_size, n_sensors).
        features: List of feature names to compute. Each must be a known
            feature name (see module-level constants for valid names).
            Order determines the column order in the output.

    Returns:
        Feature matrix of shape (n_windows, len(features) * n_sensors).
        Columns ordered as: feat_1_sensor_1, feat_1_sensor_2, ...,
        feat_2_sensor_1, feat_2_sensor_2, ..., etc.

    Raises:
        ValueError: If any feature name is not recognized.
    """
    n_windows, window_size, n_sensors = X.shape

    # Collect all known feature names across domains
    _ALL_KNOWN = set(
        STATISTICAL_FEATURES + ['mean', 'std', 'abs_energy']
        + TREND_FEATURES + ['autocorr_lag_3',
                            'partial_autocorr_lag_1',
                            'partial_autocorr_lag_2',
                            'partial_autocorr_lag_3']
        + MEMORY_FEATURES
        + FREQUENCY_FEATURES
        + COMPLEXITY_FEATURES
    )
    unknown = [f for f in features if f not in _ALL_KNOWN]
    if unknown:
        raise ValueError(
            f"Unknown features: {unknown}. "
            f"Valid options: {sorted(_ALL_KNOWN)}"
        )

    # Time axis for trend computations
    t = np.arange(window_size, dtype=float)
    t_mean = t.mean()
    t_centered = t - t_mean
    t_var = np.sum(t_centered ** 2)

    feature_blocks: list[np.ndarray] = []

    for feat in features:

        # ----------------------------------------------------------------
        # Statistical features
        # ----------------------------------------------------------------
        if feat == 'median':
            block = np.median(X, axis=1)

        elif feat == 'mean':
            block = X.mean(axis=1)

        elif feat == 'std':
            block = X.std(axis=1)

        elif feat == 'rms':
            # Root mean square — normalized energy, O(n)
            block = np.sqrt(np.mean(X ** 2, axis=1))

        elif feat == 'abs_energy':
            # Legacy: sum of squares — kept for backward compatibility
            block = np.sum(X ** 2, axis=1)

        elif feat == 'q25':
            block = np.quantile(X, 0.25, axis=1)

        elif feat == 'q75':
            block = np.quantile(X, 0.75, axis=1)

        # ----------------------------------------------------------------
        # Trend features
        # ----------------------------------------------------------------
        elif feat == 'slope':
            x_mean = X.mean(axis=1, keepdims=True)
            x_centered = X - x_mean
            cov = np.sum(
                t_centered[:, np.newaxis] * x_centered, axis=1
            ) / window_size
            block = cov / (t_var / window_size)

        elif feat == 'rvalue':
            x_mean = X.mean(axis=1, keepdims=True)
            x_centered = X - x_mean
            x_std = X.std(axis=1)
            t_std = t_centered.std()
            cov = np.sum(
                t_centered[:, np.newaxis] * x_centered, axis=1
            ) / window_size
            denom = t_std * x_std
            with np.errstate(invalid='ignore', divide='ignore'):
                block = np.where(denom > 0, cov / denom, 0.0)

        elif feat.startswith('autocorr_lag_'):
            lag = int(feat.split('_')[-1])
            if lag >= window_size:
                block = np.zeros((n_windows, n_sensors))
            else:
                x1 = X[:, :-lag, :]
                x2 = X[:, lag:, :]
                x1_c = x1 - x1.mean(axis=1, keepdims=True)
                x2_c = x2 - x2.mean(axis=1, keepdims=True)
                n = window_size - lag
                cov = np.sum(x1_c * x2_c, axis=1) / n
                denom = x1.std(axis=1) * x2.std(axis=1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    block = np.where(denom > 0, cov / denom, 0.0)

        elif feat.startswith('partial_autocorr_lag_'):
            # Legacy: Yule-Walker — O(n²) per window per sensor
            lag = int(feat.split('_')[-1])
            block = np.zeros((n_windows, n_sensors))
            for s in range(n_sensors):
                for w in range(n_windows):
                    series = X[w, :, s]
                    if series.std() < 1e-10:
                        continue
                    r = np.array([
                        np.corrcoef(series[:-k], series[k:])[0, 1]
                        if k < window_size else 0.0
                        for k in range(1, lag + 1)
                    ])
                    r0 = np.zeros(lag)
                    r0[0] = 1.0
                    for k in range(1, lag):
                        if k < window_size:
                            r0[k] = np.corrcoef(
                                series[:-k], series[k:]
                            )[0, 1]
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

        # ----------------------------------------------------------------
        # Memory features — O(n) alternatives to partial_autocorr
        # ----------------------------------------------------------------
        elif feat == 'runs_ratio':
            # Proportion of sign-change runs in first differences.
            # d_t = sign(x_t - x_{t-1}), count runs of equal signs.
            # runs_ratio = n_runs / (window_size - 1)
            # Low → persistent direction; High → chaotic alternation.
            diffs = np.diff(X, axis=1)          # (n_windows, ws-1, n_sensors)
            signs = np.sign(diffs)
            # Count sign changes (transitions between consecutive diffs)
            changes = np.diff(signs, axis=1)    # (n_windows, ws-2, n_sensors)
            n_changes = (changes != 0).sum(axis=1)  # (n_windows, n_sensors)
            # n_runs = n_changes + 1 (at least one run always exists)
            n_runs = n_changes + 1
            block = n_runs / (window_size - 1)

        elif feat == 'hurst_rs':
            # Hurst exponent via simplified rescaled range R/S.
            # Series: cumulative deviations from mean.
            # R = max(cumsum) - min(cumsum)
            # S = std(x)
            # H = log(R/S) / log(n)
            # H > 0.5 → persistent; H < 0.5 → antipersistent.
            x_mean = X.mean(axis=1, keepdims=True)
            x_dev = X - x_mean                           # centered
            cumsum = np.cumsum(x_dev, axis=1)            # cumulative sum
            R = (cumsum.max(axis=1)
                 - cumsum.min(axis=1))                   # range
            S = X.std(axis=1)                            # std
            with np.errstate(invalid='ignore', divide='ignore'):
                rs = np.where(S > 1e-10, R / S, 0.0)
                block = np.where(
                    rs > 0,
                    np.log(rs) / np.log(window_size),
                    0.0
                )

        # ----------------------------------------------------------------
        # Frequency features
        # ----------------------------------------------------------------
        elif feat.startswith('fft_coef_'):
            k = int(feat.split('_')[-1])
            # Absolute value of k-th FFT coefficient over window axis
            # rfft shape: (n_windows, window_size//2+1, n_sensors)
            fft_vals = np.fft.rfft(X, axis=1)
            if k < fft_vals.shape[1]:
                block = np.abs(fft_vals[:, k, :])
            else:
                block = np.zeros((n_windows, n_sensors))

        # ----------------------------------------------------------------
        # Complexity features
        # ----------------------------------------------------------------
        elif feat == 'permutation_entropy':
            # Shannon entropy of ordinal patterns — tau=1, dim=3.
            # PE = -sum(p(π) * log2(p(π))) for all ordinal patterns π.
            # O(n) per window per sensor — Python loop required.
            dim = 3
            block = np.zeros((n_windows, n_sensors))
            for s in range(n_sensors):
                for w in range(n_windows):
                    series = X[w, :, s]
                    n = window_size - dim + 1
                    if n <= 0:
                        continue
                    # Build ordinal patterns
                    patterns: list[tuple] = []
                    for i in range(n):
                        sub = series[i:i + dim]
                        patterns.append(tuple(np.argsort(sub)))
                    # Count frequencies
                    from collections import Counter
                    counts = Counter(patterns)
                    total = sum(counts.values())
                    probs = np.array(
                        [v / total for v in counts.values()],
                        dtype=float
                    )
                    # Shannon entropy base 2
                    with np.errstate(divide='ignore'):
                        block[w, s] = -np.sum(
                            probs * np.log2(probs + 1e-12)
                        )

        else:
            raise ValueError(f"Unhandled feature: {feat}")

        feature_blocks.append(block)

    return np.concatenate(feature_blocks, axis=1)


def _build_feature_names(
    features: list[str],
    sensor_names: list[str],
) -> list[str]:
    """Builds the column names for the feature matrix.

    Column order matches the output of _compute_features:
        sensor_1__feat_1, sensor_2__feat_1, ...,
        sensor_1__feat_2, sensor_2__feat_2, ..., etc.

    Args:
        features: Feature names in the same order as _compute_features.
        sensor_names: Sensor column names.

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

    Args:
        motor_windows: Output of build_windows (Nodo 2). Each motor entry
            must have X_windows of shape (n_windows, window_size, n_sensors).
        features: List of feature names to compute. Defaults to SET_B
            (STATISTICAL_FEATURES + TREND_FEATURES — pipeline reference).
            Use FEATURE_SETS['A'..'E'] for candidate evaluation.

    Returns:
        MotorWindows with X_windows replaced by a 2D array of shape
        (n_windows, n_features_total) and feature_names updated.

    Raises:
        ValueError: If any motor's X_windows is not 3-dimensional.
        ValueError: If any motor's windows are not in chronological order.
        ValueError: If any feature name is not recognized.
    """
    if features is None:
        features = SET_B

    result: MotorWindows = {}

    for motor_id, data in motor_windows.items():
        X_windows = data['X_windows']

        if X_windows.ndim != 3:
            raise ValueError(
                f"Motor {motor_id}: X_windows must be 3D "
                f"(n_windows, window_size, n_sensors), "
                f"got shape {X_windows.shape}."
            )

        if not np.all(np.diff(data['t_stop']) >= 0):
            raise ValueError(
                f"Motor {motor_id}: windows not in chronological order. "
                f"t_stop must be non-decreasing."
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