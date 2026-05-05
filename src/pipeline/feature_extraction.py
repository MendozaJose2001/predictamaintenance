"""Temporal feature extraction for the RUL estimation pipeline.

This module implements Nodo 3 of the pipeline — transforming the 3D window
tensor (n_ventanas, window_size, n_sensores) produced by Nodo 2 into a 2D
feature matrix (n_ventanas, n_features_total) suitable for PCA in Nodo 4.

Feature extraction is performed using tsfresh with a manually curated
configuration divided into two categories justified by the literature
(Vollert & Theissler 2021):

    Statistical descriptors:
        Capture the distribution of sensor values within each window —
        central tendency, spread, extremes, and energy.

    Trend and correlation:
        Capture the temporal dynamics of sensor degradation within each
        window — linear trend, autocorrelation, and partial autocorrelation.

Design decisions:
    Conjoint extraction:
        All sensors are processed in a single tsfresh call using a wide
        DataFrame format. This is more efficient than per-sensor calls and
        produces the output directly in the format expected by PCA (one row
        per window, one column per feature × sensor combination).

    Motor-level processing:
        Feature extraction is applied motor by motor to avoid constructing
        a single massive DataFrame for the entire dataset. This keeps memory
        usage predictable and makes the per-fold extraction in the GGS loop
        straightforward.

    Passthrough of survival targets:
        The output MotorWindows preserves all fields from Nodo 2 (t_start,
        t_stop, evento, y_rul) unchanged. Only X_windows is replaced — from
        a 3D tensor to a 2D feature matrix. feature_names is updated to
        reflect the tsfresh output column names.

    NaN handling:
        tsfresh may produce NaN for some features (e.g. partial_autocorrelation
        when the window is too short). NaN values are filled with 0 after
        extraction. This is safe because the subsequent RobustScaler and PCA
        are not NaN-tolerant.

TODO: Evaluate adding Spearman correlation with time as a custom feature
    for monotonicity detection. tsfresh does not provide Spearman natively.
    linear_trend attr="rvalue" is the closest proxy currently included.
    Pending discussion with advisor on whether monotonicity is sufficiently
    captured by linear_trend or requires explicit Spearman computation via
    scipy.stats.spearmanr(series, range(window_size)).
"""

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from src.pipeline.windowing import MotorData, MotorWindows


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

# Statistical descriptors — capture distribution within each window.
STATISTICAL_FEATURES: dict = {
    "median":             None,
    "minimum":            None,
    "maximum":            None,
    "abs_energy":         None,
    # standard_deviation used as proxy for MAD — tsfresh has no native MAD.
    # Both measure spread; std is more sensitive to outliers than MAD but
    # is universally supported and sufficient for a first pipeline iteration.
    "standard_deviation": None,
    # quantile at 0.25 and 0.75 — together they define the IQR
    "quantile":           [{"q": 0.25}, {"q": 0.75}],
}

# Trend and correlation — capture temporal dynamics within each window.
# TODO: Add Spearman correlation for monotonicity — see module docstring.
TREND_FEATURES: dict = {
    # slope: rate of change over the window
    # rvalue: Pearson correlation with time — proxy for linear monotonicity
    "linear_trend":            [{"attr": "slope"}, {"attr": "rvalue"}],
    # autocorrelation at lag 1, 2, 3 — short-term temporal dependency
    "autocorrelation":         [{"lag": 1}, {"lag": 2}, {"lag": 3}],
    # partial autocorrelation — removes indirect lag effects
    "partial_autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
}

# Full feature configuration — union of both categories.
# Pass STATISTICAL_FEATURES or TREND_FEATURES individually to ablate
# one category during experimentation.
TSFRESH_FEATURES: dict = {**STATISTICAL_FEATURES, **TREND_FEATURES}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_window_features(
    motor_windows: MotorWindows,
    feature_config: dict | None = None,
    n_jobs: int = 1,
) -> MotorWindows:
    """Extracts temporal features from sliding windows using tsfresh.

    Transforms the 3D window tensor in each motor's MotorData into a 2D
    feature matrix by applying tsfresh feature calculators to each sensor
    series within each window. All other fields (t_start, t_stop, evento,
    y_rul) are preserved unchanged.

    Feature extraction is performed motor by motor to keep memory usage
    predictable. Within each motor, all windows are processed in a single
    tsfresh call using the wide multi-column DataFrame format.

    Note on quantiles and IQR:
        STATISTICAL_FEATURES includes quantile at q=0.25 and q=0.75 rather
        than computing IQR directly. IQR = Q75 - Q25 would introduce perfect
        multicollinearity with the two quantiles, which can destabilize PCA.
        PCA can derive the IQR direction implicitly as a linear combination
        of the two quantile components without explicit computation.

    Args:
        motor_windows: Output of build_windows (Nodo 2). Each motor entry
            must have X_windows of shape (n_windows, window_size, n_sensors).
        feature_config: tsfresh feature calculator configuration dictionary.
            Keys are calculator names, values are parameter lists or None.
            Defaults to TSFRESH_FEATURES (both statistical and trend groups).
            Pass STATISTICAL_FEATURES or TREND_FEATURES to use a subset.
        n_jobs: Number of parallel jobs for tsfresh feature extraction.
            Defaults to 1 (serial) — safe for all hardware configurations
            and avoids parallelization overhead when processing small per-motor
            DataFrames. Increase on machines with many cores if profiling
            shows feature extraction is a bottleneck.

    Returns:
        MotorWindows with the same structure as the input, but X_windows
        replaced by a 2D array of shape (n_windows, n_features_total) and
        feature_names updated to the tsfresh output column names.

    Raises:
        ValueError: If any motor's X_windows is not 3-dimensional.
        ValueError: If any motor's windows are not in chronological order
            (t_stop must be non-decreasing across windows).
    """
    if feature_config is None:
        feature_config = TSFRESH_FEATURES

    result: MotorWindows = {}

    for motor_id, data in motor_windows.items():
        X_windows = data['X_windows']

        if X_windows.ndim != 3:
            raise ValueError(
                f"Motor {motor_id}: X_windows must be 3D "
                f"(n_windows, window_size, n_sensors), got shape {X_windows.shape}."
            )

        # Defensive check — build_windows guarantees chronological order but
        # this catches accidental reordering introduced between Nodo 2 and 3.
        if not np.all(np.diff(data['t_stop']) >= 0):
            raise ValueError(
                f"Motor {motor_id}: windows are not in chronological order. "
                f"t_stop must be non-decreasing. Check build_windows output."
            )

        n_windows, window_size, n_sensors = X_windows.shape
        sensor_names: list[str] = list(data['feature_names'])

        # Build tsfresh long-format DataFrame using vectorized operations.
        # Equivalent to the row-by-row loop but orders of magnitude faster:
        #   window_id: each window repeated window_size times
        #   time:      0..window_size-1 tiled n_windows times
        #   sensors:   X_windows reshaped from (n_windows, window_size, n_sensors)
        #              to (n_windows * window_size, n_sensors)
        window_ids = np.repeat(np.arange(n_windows), window_size)
        time_idx = np.tile(np.arange(window_size), n_windows)
        sensor_vals = X_windows.reshape(n_windows * window_size, n_sensors)

        df_long = pd.DataFrame(sensor_vals, columns=sensor_names)
        df_long.insert(0, 'window_id', window_ids)
        df_long.insert(1, 'time', time_idx)

        # Extract features — one row per window_id in the output.
        # pd.DataFrame() wrapper forces the correct type since tsfresh's
        # return type annotation is ambiguous (list | Unknown | DataFrame).
        features_df: pd.DataFrame = pd.DataFrame(extract_features(
            df_long,
            column_id='window_id',
            column_sort='time',
            default_fc_parameters=feature_config,
            impute_function=impute,
            disable_progressbar=True,
            n_jobs=n_jobs,
        ))

        # Sort by window_id to guarantee alignment with t_start, t_stop etc.
        features_df = features_df.sort_index()

        # Fill any remaining NaN (e.g. partial_autocorrelation edge cases)
        features_df = features_df.fillna(0.0)

        X_features = features_df.to_numpy(dtype=float)
        new_feature_names: list[str] = list(features_df.columns)

        result[motor_id] = MotorData(
            X_windows=X_features,
            t_start=data['t_start'],
            t_stop=data['t_stop'],
            evento=data['evento'],
            y_rul=data['y_rul'],
            feature_names=new_feature_names,
        )

    return result