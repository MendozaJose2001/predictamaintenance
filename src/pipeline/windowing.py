"""Sliding window construction for the RUL estimation pipeline.

This module implements Nodo 2 of the pipeline — transforming a flat
time-series DataFrame into a dictionary of sliding windows per motor.
Each window captures the degradation history of a motor over the last
window_size cycles, enabling temporal feature extraction in subsequent
nodes (tsfresh, PCA).

Design decisions:
    Input flexibility:
        The function accepts a DataFrame containing one or more motors.
        Motor identity is tracked via the 'unit_number' column (or a
        groups array) so that the same function works in both training
        (all motors together) and production (single motor) contexts.

    Output structure:
        A dictionary keyed by motor_id. Each value contains the window
        tensor and the associated survival targets and RUL labels. This
        avoids forcing 3D arrays into a 2D DataFrame and preserves the
        motor identity needed for GroupKFold splitting.

    Window indexing:
        For motor with cycles t=1..T, windows are generated from
        t=window_size to t=T. The first window_size-1 cycles are
        discarded — with a piecewise linear RUL + clipping, those early
        cycles fall in the flat region and carry little predictive signal.

    Survival targets:
        Each window maps to one counting process interval:
            t_start = time_in_cycles[first row of window] - 1
            t_stop  = time_in_cycles[last row of window]
            evento  = 1 only if t_stop is the last observed cycle of a
                      train motor (evento=1 in the original dataset)

    RUL target:
        RUL at t_stop, clipped to clipping_threshold. Used for metric
        computation — not passed to survival model fit.
"""

import numpy as np
import pandas as pd
from typing import TypedDict


class MotorData(TypedDict):
    """Data structure for a single motor's sliding windows."""
    X_windows: np.ndarray
    t_start: np.ndarray
    t_stop: np.ndarray
    evento: np.ndarray
    y_rul: np.ndarray
    feature_names: list[str]


# Type alias for the full output dictionary keyed by motor_id
MotorWindows = dict[int, MotorData]


def build_windows(
    df: pd.DataFrame,
    window_size: int,
    clipping_threshold: int,
    unit_col: str = 'unit_number',
    time_col: str = 'time_in_cycles',
    rul_col: str = 'RUL',
    evento_col: str = 'evento',
) -> MotorWindows:
    """Constructs sliding windows over a multi-motor time-series DataFrame.

    For each motor in df, generates one window per cycle from cycle
    window_size onward. Each window contains the last window_size rows
    of sensor/setting data ending at that cycle, together with the
    corresponding survival interval and clipped RUL.

    The function is designed to handle both training datasets (multiple
    motors, RUL and evento columns present) and production datasets
    (single motor, RUL and evento may be absent — see notes below).

    Assumptions:
        Column order: feature_cols is derived from the input DataFrame column
            order after removing metadata columns. The same column order must
            be preserved across all calls (train, validation, production) to
            guarantee consistent feature alignment. Consider sorting columns
            externally before calling this function if order consistency is
            a concern.

        Cycle normalization: time_in_cycles is NOT normalized. The absolute
            cycle count is preserved so that Cox and AFT models can estimate
            a shared baseline hazard H0(t) on a common time axis across
            motors. This assumes that all motors begin at time_in_cycles >= 1,
            which holds for C-MAPSS FD001 by construction. A ValueError is
            raised if any motor starts below cycle 1 to prevent silent
            negative t_start values. If motors have been filtered externally
            and cycles no longer start at 1, re-index before calling:
                df[time_col] = df.groupby(unit_col)[time_col].transform(
                    lambda x: x - x.min() + 1
                )

    Args:
        df: Input DataFrame containing at minimum:
            - unit_col: Motor identifier (int). If absent, all rows are
              treated as a single motor with id=0.
            - time_col: Cycle index, strictly increasing per motor.
            - Feature columns: all remaining columns except unit_col,
              time_col, rul_col, and evento_col are treated as features.
            - rul_col: Scalar RUL per row. Optional — if absent, y_rul
              is filled with NaN.
            - evento_col: Binary event indicator per row. Optional — if
              absent, evento is filled with 0 (all censored).
        window_size: Number of consecutive cycles per window. Must be
            >= 1. Windows shorter than window_size (early cycles of each
            motor) are discarded.
        clipping_threshold: Maximum RUL value. Applied to y_rul before
            storing — consistent with the piecewise linear RUL convention.
        unit_col: Name of the motor identifier column. Defaults to
            'unit_number'.
        time_col: Name of the cycle index column. Defaults to
            'time_in_cycles'.
        rul_col: Name of the RUL column. Defaults to 'RUL'.
        evento_col: Name of the event indicator column. Defaults to
            'evento'.

    Returns:
        Dictionary mapping motor_id (int) to a sub-dictionary with:
            'X_windows': np.ndarray of shape
                (n_windows, window_size, n_features).
                Sensor and setting values for each window. Feature order
                matches the order of columns in df after removing
                unit_col, time_col, rul_col, and evento_col.
            't_start': np.ndarray of shape (n_windows,).
                Start of the counting process interval for each window.
                Equals time_in_cycles[first row] - 1.
            't_stop': np.ndarray of shape (n_windows,).
                End of the counting process interval for each window.
                Equals time_in_cycles[last row].
            'evento': np.ndarray of shape (n_windows,) dtype int.
                1 if the window ends at the observed failure cycle of a
                train motor, 0 otherwise.
            'y_rul': np.ndarray of shape (n_windows,).
                RUL at t_stop clipped to clipping_threshold. NaN if
                rul_col is absent from df.
            'feature_names': list[str].
                Names of the feature columns in X_windows column order.

    Raises:
        ValueError: If window_size < 1.
        ValueError: If any motor has fewer rows than window_size (no
            windows can be generated for that motor).
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    # Determine feature columns — everything except metadata and targets
    meta_cols = {unit_col, time_col, rul_col, evento_col}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Handle missing unit_number — treat entire df as one motor
    if unit_col not in df.columns:
        df = df.copy()
        df[unit_col] = 0

    # Handle missing RUL — fill with NaN
    has_rul = rul_col in df.columns
    if not has_rul:
        df = df.copy()
        df[rul_col] = np.nan

    # Handle missing evento — fill with 0 (all censored, production mode)
    has_evento = evento_col in df.columns
    if not has_evento:
        df = df.copy()
        df[evento_col] = 0

    result: MotorWindows = {}

    for motor_id, motor_df in df.groupby(unit_col, sort=False):
        # Sort by time to guarantee chronological window order
        motor_df = motor_df.sort_values(time_col).reset_index(drop=True)

        # Validate that time_in_cycles starts at >= 1 so that
        # t_start = time_in_cycles - 1 is never negative. Cox and AFT models
        # require a shared absolute time axis across motors — normalizing each
        # motor independently would distort the baseline hazard estimate. If
        # motors have been filtered and cycles no longer start at 1, the
        # caller must re-index cycles before passing df to build_windows.
        first_cycle = float(motor_df[time_col].iloc[0])
        if first_cycle < 1:
            raise ValueError(
                f"Motor {motor_id} has time_in_cycles starting at {first_cycle}. "
                f"Expected >= 1 so that t_start = time_in_cycles - 1 >= 0. "
                f"Re-index cycles before calling build_windows."
            )

        n_cycles = len(motor_df)

        if n_cycles < window_size:
            raise ValueError(
                f"Motor {motor_id} has {n_cycles} cycles but window_size={window_size}. "
                f"All motors must have at least window_size cycles."
            )

        feature_vals = motor_df[feature_cols].to_numpy(dtype=float)
        time_vals = motor_df[time_col].to_numpy(dtype=float)
        rul_vals = motor_df[rul_col].to_numpy(dtype=float)
        evento_vals = motor_df[evento_col].to_numpy(dtype=int)

        # Number of complete windows for this motor
        n_windows = n_cycles - window_size + 1

        X_windows = np.empty((n_windows, window_size, len(feature_cols)), dtype=float)
        t_start = np.empty(n_windows, dtype=float)
        t_stop = np.empty(n_windows, dtype=float)
        evento_out = np.zeros(n_windows, dtype=int)
        y_rul = np.empty(n_windows, dtype=float)

        for i in range(n_windows):
            # Window spans cycles [i, i + window_size)
            window_slice = slice(i, i + window_size)
            last_idx = i + window_size - 1

            X_windows[i] = feature_vals[window_slice]
            t_start[i] = time_vals[i] - 1.0
            t_stop[i] = time_vals[last_idx]

            # evento=1 only if the last cycle of this window is the
            # observed failure cycle of a train motor
            evento_out[i] = evento_vals[last_idx]

            # RUL at t_stop, clipped to clipping_threshold
            y_rul[i] = min(rul_vals[last_idx], clipping_threshold)

        motor_id_int: int = int(str(motor_id))
        result[motor_id_int] = MotorData(
            X_windows=X_windows,
            t_start=t_start,
            t_stop=t_stop,
            evento=evento_out,
            y_rul=y_rul,
            feature_names=feature_cols,
        )

    return result


def flatten_windows(
    motor_windows: MotorWindows,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flattens a MotorWindows dictionary into concatenated arrays.

    Convenience function for assembling the full training dataset after
    windowing. Concatenates all motor windows in motor_id order and
    builds a groups array for GroupKFold splitting.

    Args:
        motor_windows: Output of build_windows.

    Returns:
        Tuple of (X_windows, t_start, t_stop, evento, y_rul, groups) where:
            X_windows: np.ndarray of shape (total_windows, window_size, n_features).
            t_start: np.ndarray of shape (total_windows,).
            t_stop: np.ndarray of shape (total_windows,).
            evento: np.ndarray of shape (total_windows,).
            y_rul: np.ndarray of shape (total_windows,).
            groups: np.ndarray of shape (total_windows,) with motor_id
                repeated for each window of that motor.
    """
    X_list, ts_list, te_list, ev_list, rul_list, grp_list = [], [], [], [], [], []

    for motor_id, data in motor_windows.items():
        n = len(data['y_rul'])
        X_list.append(data['X_windows'])
        ts_list.append(data['t_start'])
        te_list.append(data['t_stop'])
        ev_list.append(data['evento'])
        rul_list.append(data['y_rul'])
        grp_list.append(np.full(n, motor_id, dtype=int))

    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(ts_list),
        np.concatenate(te_list),
        np.concatenate(ev_list),
        np.concatenate(rul_list),
        np.concatenate(grp_list),
    )