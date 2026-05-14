"""Test set analysis functions for PredictaMaintenance.

Provides data loading, prediction generation, and metric computation
utilities for evaluating production models on the held-out test set.
All prediction functions return plain dicts of numpy arrays, keeping
the interface free of matplotlib dependencies.

Three levels of analysis are supported:

    - Trajectory level: all windows of all 60 test motors.
    - Last window level: zoom into the risk zone — one window per motor.
    - Per-motor level: metric distribution across the 60 test motors.

Typical usage::

    from src.analysis.test_analysis import (
        get_trajectory_predict,
        test_trajectory,
        test_last_window,
        test_per_motor,
    )

    results, predictors, m_test = get_trajectory_predict()
    test_trajectory()
    test_last_window()
    test_per_motor()
"""

import warnings
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.analysis.config import (
    ELIGIBLE_MODELS,
    MODEL_NAMES,
    PREDICTOR_PATHS,
)

# ---------------------------------------------------------------------------
# Type aliases and protocols
# ---------------------------------------------------------------------------

#: Per-model prediction dict: y_pred, y_true, t_stop, groups as 1-D arrays.
ModelResults = dict[str, dict[str, np.ndarray]]

#: Per-motor metric lists keyed by model identifier.
PerMotorMetrics = dict[str, list[float]]

#: Return type of test_per_motor.
PerMotorResult = tuple[pd.DataFrame, PerMotorMetrics, PerMotorMetrics]


class _Predictor(Protocol):
    """Structural protocol for production pipeline predictors.

    Defines the minimal interface expected from any loaded .pkl pipeline
    object. Using a Protocol avoids casting to Any while giving Pyright
    full knowledge of the attributes and methods used downstream.
    """

    window_size:          int
    feature_set:          str
    n_components:         int
    clipping_threshold:   int

    def predict(self, X: pd.DataFrame, return_mode: str = 'last') -> np.ndarray:
        """Generates RUL predictions for all windows in X."""
        ...


#: Loaded production predictors keyed by model identifier.
Predictors = dict[str, _Predictor]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_models(debug: bool = True) -> Predictors:
    """Loads all eligible production models from disk.

    Args:
        debug: If True, prints model configuration for each loaded predictor.

    Returns:
        Dict mapping model key to the loaded pipeline object.
    """
    import joblib  # noqa: PLC0415

    predictors: Predictors = {}
    for key in ELIGIBLE_MODELS:
        predictor: _Predictor = joblib.load(PREDICTOR_PATHS[key])
        predictors[key] = predictor
        if debug:
            print(
                f"[OK] {MODEL_NAMES[key]:20s} — "
                f"window_size={predictor.window_size}, "
                f"feature_set={predictor.feature_set}, "
                f"n_components={predictor.n_components}, "
                f"clipping_threshold={predictor.clipping_threshold}"
            )
    return predictors


def get_test_motors(debug: bool = True) -> tuple[pd.DataFrame, list[int]]:
    """Loads all test motor CSV files and concatenates them into one DataFrame.

    Args:
        debug: If True, prints dataset shape and event count.

    Returns:
        Tuple of (df_test, m_test) where df_test is the concatenated
        DataFrame and m_test is the ordered list of test motor IDs.
    """
    from src.dataset_manager import DatasetManager  # noqa: PLC0415

    _, m_test_raw = DatasetManager.split_dataset()
    m_test: list[int] = [int(x) for x in m_test_raw]

    dfs: list[pd.DataFrame] = []
    for idx in m_test:
        df = pd.read_csv(f'data/clean/data_motor_{idx}.csv')
        df.insert(0, 'unit_number', idx)
        dfs.append(df)

    df_test = pd.concat(dfs, ignore_index=True)

    if debug:
        print(f"Motores de test: {df_test['unit_number'].nunique()}")
        print(f"Total filas:     {len(df_test)}")
        print(f"Eventos:         {df_test['evento'].sum()}")

    return df_test, m_test


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------

def get_trajectory_predict(
    debug: bool = True,
) -> tuple[ModelResults, Predictors, list[int]]:
    """Generates full trajectory predictions for all eligible models.

    Each motor contributes all its sliding windows. The y_true array is
    clipped to each model's clipping_threshold to match the training target.

    Args:
        debug: If True, prints per-model window counts.

    Returns:
        Tuple of (results, predictors, m_test) where results maps each
        model key to a dict with keys: y_pred, y_true, t_stop, groups.
    """
    warnings.filterwarnings('ignore')

    predictors = get_models(debug)
    df_test, m_test = get_test_motors(debug)

    results: ModelResults = {}

    for key, predictor in predictors.items():
        y_pred_list: list[np.ndarray] = []
        y_true_list: list[np.ndarray] = []
        t_stop_list: list[np.ndarray] = []
        groups_list: list[np.ndarray] = []

        for motor_id in m_test:
            df_motor = df_test[df_test['unit_number'] == motor_id].copy()
            X_motor  = df_motor.drop(columns=['RUL'])

            y_pred = predictor.predict(X_motor, return_mode='all')

            clip    = predictor.clipping_threshold
            rul_raw = np.asarray(df_motor['RUL'].values, dtype=float)
            y_true  = np.minimum(rul_raw[-len(y_pred):], float(clip))
            t_stop  = np.asarray(
                df_motor['time_in_cycles'].values, dtype=float
            )[-len(y_pred):]

            y_pred_list.append(np.asarray(y_pred, dtype=float))
            y_true_list.append(y_true)
            t_stop_list.append(t_stop)
            groups_list.append(np.full(len(y_pred), motor_id, dtype=int))

        results[key] = {
            'y_pred': np.concatenate(y_pred_list),
            'y_true': np.concatenate(y_true_list),
            't_stop': np.concatenate(t_stop_list),
            'groups': np.concatenate(groups_list),
        }

        if debug:
            print(
                f"[OK] {MODEL_NAMES[key]:20s} — "
                f"{len(results[key]['y_pred'])} ventanas"
            )

    return results, predictors, m_test


def get_last_window_predict(
    debug: bool = True,
) -> tuple[ModelResults, Predictors, list[int]]:
    """Extracts the last sliding window per motor from full trajectories.

    The last window corresponds to the highest-degradation observation
    available — the point closest to failure in the test cluster. This
    represents the risk zone relevant to actual maintenance decisions.

    Args:
        debug: If True, prints the number of extracted windows.

    Returns:
        Tuple of (results_last, predictors, m_test) with the same
        structure as get_trajectory_predict but with one row per motor.
    """
    results, predictors, m_test = get_trajectory_predict(debug=False)

    results_last: ModelResults = {}

    for key, res in results.items():
        groups   = res['groups']
        idx_last = np.array([
            int(np.where(groups == motor_id)[0][-1])
            for motor_id in m_test
        ], dtype=int)
        results_last[key] = {
            'y_pred': res['y_pred'][idx_last],
            'y_true': res['y_true'][idx_last],
            't_stop': res['t_stop'][idx_last],
            'groups': groups[idx_last],
        }

    if debug:
        key0 = list(results_last.keys())[0]
        print(
            f"Ventanas extraídas por modelo: "
            f"{len(results_last[key0]['y_pred'])} (una por motor)"
        )

    return results_last, predictors, m_test


# ---------------------------------------------------------------------------
# Metric tables
# ---------------------------------------------------------------------------

def _build_metrics_row(
    key: str,
    res: dict[str, np.ndarray],
    clip: int,
) -> dict[str, Any]:
    """Computes the four evaluation metrics for a single model result dict.

    Args:
        key: Model identifier key.
        res: Dict with y_pred and y_true arrays.
        clip: Clipping threshold used for metric computation.

    Returns:
        Dict with Modelo, S-Score, C-Index, MAE, RMSE.
    """
    from src.metrics_manager import Metrics  # noqa: PLC0415

    return {
        'Modelo' : MODEL_NAMES[key],
        'S-Score': Metrics.s_score(res['y_pred'], res['y_true'], clip),
        'C-Index': Metrics.c_index(res['y_pred'], res['y_true'], clip),
        'MAE'    : Metrics.mae(res['y_pred'],     res['y_true'], clip),
        'RMSE'   : Metrics.rmse(res['y_pred'],    res['y_true'], clip),
    }


def test_trajectory(debug: bool = False) -> None:
    """Prints the metric table for Subanálisis 1 — full trajectory evaluation.

    Evaluates all eligible models over the complete set of sliding windows
    from the 60 test motors. S-Score and C-Index are the primary metrics;
    MAE and RMSE are reported for reference.

    Args:
        debug: Passed to get_trajectory_predict for verbose loading output.
    """
    results, predictors, m_test = get_trajectory_predict(debug)

    rows = [
        _build_metrics_row(key, res, predictors[key].clipping_threshold)
        for key, res in results.items()
    ]
    df_metrics = (
        pd.DataFrame(rows)
        .sort_values('S-Score', ascending=True)
        .reset_index(drop=True)
    )

    print("=" * 65)
    print("  Full Trajectory Evaluation")
    print("=" * 65)
    print(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"  Ventanas totales evaluadas: {len(results['nb']['y_pred'])}")
    print(f"  Motores de test:            {len(m_test)}")
    print("=" * 65)


def test_last_window(debug: bool = False) -> None:
    """Prints the metric table for Subanálisis 2 — risk zone evaluation.

    Evaluates all eligible models using only the last sliding window per
    motor (N=60). This reflects the maintenance decision point where the
    motor is in its most advanced degradation state.

    Args:
        debug: Passed to get_last_window_predict for verbose loading output.
    """
    results_last, predictors, m_test = get_last_window_predict(debug)

    rows = [
        _build_metrics_row(key, res, predictors[key].clipping_threshold)
        for key, res in results_last.items()
    ]
    df_metrics = (
        pd.DataFrame(rows)
        .sort_values('S-Score', ascending=True)
        .reset_index(drop=True)
    )

    print("=" * 75)
    print("  Risk Zone Evaluation")
    print("=" * 65)
    print(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"  Motores evaluados: {len(m_test)}")
    print("=" * 75)


def test_per_motor(debug: bool = False) -> PerMotorResult:
    """Computes per-motor metric distributions for Subanálisis 3.

    Calculates S-Score and C-Index for each of the 60 test motors
    independently, then reports median, IQR, and maximum across motors.
    Motors where C-Index is undefined (constant y_true due to clipping)
    are excluded from C-Index statistics via np.nan.

    Args:
        debug: Passed to get_trajectory_predict for verbose loading output.

    Returns:
        Tuple of (df_per_motor, ss_per_motor, ci_per_motor) where
        df_per_motor contains summary statistics and ss_per_motor /
        ci_per_motor are the raw per-motor lists for downstream plotting.
    """
    from src.metrics_manager import Metrics  # noqa: PLC0415

    results, predictors, m_test = get_trajectory_predict(debug)

    ss_per_motor: PerMotorMetrics = {key: [] for key in results}
    ci_per_motor: PerMotorMetrics = {key: [] for key in results}

    for motor_id in m_test:
        for key, res in results.items():
            clip = predictors[key].clipping_threshold
            mask = res['groups'] == motor_id

            if int(mask.sum()) < 2:
                ss_per_motor[key].append(np.nan)
                ci_per_motor[key].append(np.nan)
                continue

            y_pred_m = res['y_pred'][mask]
            y_true_m = res['y_true'][mask]

            ss_per_motor[key].append(
                float(Metrics.s_score(y_pred_m, y_true_m, clip))
            )
            try:
                ci_per_motor[key].append(
                    float(Metrics.c_index(y_pred_m, y_true_m, clip))
                )
            except ZeroDivisionError:
                # Motor with all windows in the flat RUL zone — y_true
                # constant at clipping_threshold, no rankable pairs.
                ci_per_motor[key].append(np.nan)

    rows: list[dict[str, Any]] = []
    for key in results:
        ss_arr = np.array(ss_per_motor[key], dtype=float)
        ci_arr = np.array(ci_per_motor[key], dtype=float)
        rows.append({
            'Modelo'      : MODEL_NAMES[key],
            'S-Score med' : float(np.nanmedian(ss_arr)),
            'S-Score IQR' : float(
                np.nanpercentile(ss_arr, 75) - np.nanpercentile(ss_arr, 25)
            ),
            'S-Score max' : float(np.nanmax(ss_arr)),
            'C-Index med' : float(np.nanmedian(ci_arr)),
            'C-Index IQR' : float(
                np.nanpercentile(ci_arr, 75) - np.nanpercentile(ci_arr, 25)
            ),
        })

    df_per_motor = (
        pd.DataFrame(rows)
        .sort_values('S-Score med', ascending=True)
        .reset_index(drop=True)
    )

    print("=" * 75)
    print("  Per-motor Metric Distributions")
    print("=" * 75)
    print(df_per_motor.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))
    print()
    print("  S-Score med: mediana sobre 60 motores (↓ mejor)")
    print("  S-Score IQR: varianza entre motores   (↓ más consistente)")
    print("  S-Score max: peor motor individual    (↓ menos picos)")
    print("  C-Index med: ranking medio            (↑ mejor)")
    print("=" * 75)

    return df_per_motor, ss_per_motor, ci_per_motor