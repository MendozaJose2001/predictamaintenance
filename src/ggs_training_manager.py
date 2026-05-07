"""Group Grid Search training manager for the new sliding window pipeline.

This module implements GGSTrainingManager — the training orchestrator for
hyperparameter optimization of RUL estimation models using the new Nodo 1-4
pipeline. It replaces the legacy training_manager.py which was designed for
the old per-motor CSV loading approach.

Key design differences from legacy GGSTrainingManager:
    - No prepare_training_data() — receives X_df and y_df directly
    - No sklearn Pipeline — uses RULPipeline orchestrator (Nodos 1-4)
    - param_grid covers both pipeline and model hyperparameters
    - Pipeline params (window_size, n_components, clipping_threshold) are
      separated internally from model params

Pipeline hyperparameters in param_grid:
    window_size:         sliding window size (Nodo 2)
    n_components:        PCA components (Nodo 4)
    clipping_threshold:  RUL clipping threshold (Nodo 2 + model)

Model hyperparameters in param_grid:
    Any parameter accepted by model_class.__init__() except the three
    pipeline params above. Examples:
        NegativeBinomialPiecewise: alpha, alpha_reg, l1_ratio, link_type
        SVRModel:                  C, epsilon, kernel, gamma, degree

Warning suppression:
    Statistical model warnings (statsmodels convergence, domain warnings)
    are suppressed during grid search to avoid terminal flooding.
    RuntimeWarnings from failed model fits are also suppressed — failed
    configurations are handled via NaN metrics.
"""

import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from statsmodels.tools.sm_exceptions import ConvergenceWarning, DomainWarning, ValueWarning
from tqdm.auto import tqdm

from src.metrics_manager import Metrics
from src.models.base_model import BaseRULModel
from src.pipeline.rul_pipeline import RULPipeline


# Pipeline-level hyperparameters — separated from model params internally
_PIPELINE_PARAMS: frozenset[str] = frozenset({
    'window_size',
    'n_components',
    'clipping_threshold',
})


def _suppress_warnings() -> None:
    """Suppresses expected warnings during grid search."""
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', DomainWarning)
    warnings.simplefilter('ignore', ValueWarning)
    os.environ['PYTHONWARNINGS'] = 'ignore'


def _split_params(params: dict) -> tuple[dict, dict]:
    """Splits a flat param dict into pipeline and model param dicts.

    Args:
        params: Flat dictionary with all hyperparameters.

    Returns:
        Tuple of (pipeline_params, model_params).
    """
    pipeline_params = {k: v for k, v in params.items() if k in _PIPELINE_PARAMS}
    model_params    = {k: v for k, v in params.items() if k not in _PIPELINE_PARAMS}
    return pipeline_params, model_params


def _evaluate_fold(
    model: BaseRULModel,
    X_val: np.ndarray,
    y_rul_val: np.ndarray,
    t_stop_val: np.ndarray,
    evento_val: np.ndarray,
) -> dict[str, float]:
    """Evaluates a fitted model on validation fold using all four metrics.

    Args:
        model: Fitted BaseRULModel instance.
        X_val: Validation feature matrix (n_windows, n_components).
        y_rul_val: True clipped RUL for validation windows.
        t_stop_val: Last cycle of each validation window.
        evento_val: Event indicator for each validation window.

    Returns:
        Dictionary with keys S_score, C_index, MAE, RMSE.
    """
    y_pred = model.predict(X_val)
    clipping = int(getattr(model, 'clipping_threshold', 125))
    results: dict[str, float] = {}

    for name, func in Metrics.get_metrics().items():
        try:
            results[name] = float(func(y_pred, y_rul_val, clipping))
        except Exception:
            results[name] = np.nan

    return results


def _run_single_config(
    params: dict,
    model_class: type,
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    groups_df: np.ndarray,
    n_folds: int,
) -> dict:
    """Evaluates one hyperparameter configuration via GroupKFold CV.

    For each fold:
        1. Splits X_df and y_df into train/val using GroupKFold
        2. Instantiates RULPipeline with pipeline_params
        3. Calls fit_transform on training fold
        4. Calls transform on validation fold
        5. Instantiates and fits model with model_params
        6. Evaluates predictions with four metrics

    Args:
        params: Flat dict of all hyperparameters (pipeline + model).
        model_class: BaseRULModel subclass to instantiate.
        X_df: Full feature DataFrame (without RUL).
        y_df: Full target DataFrame (unit_number, time_in_cycles, RUL).
        groups_df: Motor ID per row for GroupKFold splitting.
        n_folds: Number of GroupKFold folds.

    Returns:
        Dict with hyperparameter values and mean metrics across folds.
        NaN metrics if all folds failed.
    """
    pipeline_params, model_params = _split_params(params)
    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, val_idx in gkf.split(X_df, groups=groups_df):
        try:
            X_train_fold = X_df.iloc[train_idx].reset_index(drop=True)
            X_val_fold   = X_df.iloc[val_idx].reset_index(drop=True)
            y_train_fold = y_df.iloc[train_idx].reset_index(drop=True)
            y_val_fold   = y_df.iloc[val_idx].reset_index(drop=True)

            # Nodos 1-4 — fit on training, transform both
            pipeline = RULPipeline(**pipeline_params)
            X_tr, y_rul_tr, t_stop_tr, evento_tr, groups_tr = (
                pipeline.fit_transform(X_train_fold, y_train_fold)
            )
            X_val, y_rul_val, t_stop_val, evento_val, _ = (
                pipeline.transform(X_val_fold, y_val_fold)
            )

            # Nodo 5 — model fit and predict
            model = model_class(**model_params)
            model.fit(X_tr, y_rul_tr)

            fold_metrics.append(
                _evaluate_fold(model, X_val, y_rul_val, t_stop_val, evento_val)
            )

        except Exception:
            fold_metrics.append({
                'S_score': np.nan,
                'C_index': np.nan,
                'MAE':     np.nan,
                'RMSE':    np.nan,
            })

    # Average metrics across successful folds
    mean_metrics: dict[str, float] = {}
    for metric in ['S_score', 'C_index', 'MAE', 'RMSE']:
        values = [f[metric] for f in fold_metrics if not np.isnan(f[metric])]
        mean_metrics[f'mean_{metric}'] = float(np.mean(values)) if values else np.nan

    return {**params, **mean_metrics}


class GGSTrainingManager:
    """Manages group grid search training for the sliding window pipeline.

    Orchestrates hyperparameter optimization for RUL estimation models
    using the new Nodo 1-4 pipeline. Both pipeline and model hyperparameters
    are explored jointly in the grid search.

    Args:
        model_class: BaseRULModel subclass to evaluate (not an instance).
        X_df: Full feature DataFrame without RUL column. Must contain
            unit_number, time_in_cycles, evento, and sensor/setting columns.
        y_df: Full target DataFrame with columns:
            unit_number, time_in_cycles, RUL.
        groups: Motor ID per row. Used for GroupKFold splitting to prevent
            temporal leakage between motors.

    Example:
        manager = GGSTrainingManager(
            model_class=NegativeBinomialPiecewise,
            X_df=X_df_full,
            y_df=y_df_full,
            groups=groups_full,
        )
        results = manager.group_grid_search(
            param_grid={
                'window_size':        [20, 30],
                'n_components':       [5, 10],
                'clipping_threshold': [125],
                'alpha':              [0.5, 1.0],
                'link_type':          ['log'],
            },
            n_folds=5,
        )
        display(manager.get_ggs_results(top_n=10))
    """

    def __init__(
        self,
        model_class: type,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame,
        groups: np.ndarray,
    ) -> None:
        self.model_class = model_class
        self.X_df = X_df
        self.y_df = y_df
        self.groups = groups
        self.ggs_results_: list[dict] | None = None

    def group_grid_search(
        self,
        param_grid: dict,
        n_folds: int = 5,
        silence: bool = True,
    ) -> list[dict]:
        """Runs a grouped grid search over the provided hyperparameter grid.

        Iterates over all combinations of pipeline and model hyperparameters,
        evaluating each via GroupKFold cross-validation. For each fold,
        RULPipeline is fitted on training data and applied to validation data
        without leakage.

        Args:
            param_grid: Dictionary mapping parameter names to candidate value
                lists. Pipeline params (window_size, n_components,
                clipping_threshold) and model params are mixed freely.
            n_folds: Number of GroupKFold folds. Defaults to 5.
            silence: If True, suppresses convergence and domain warnings.
                Defaults to True.

        Returns:
            List of dicts, one per configuration, with hyperparameter values
            and mean cross-validated metrics (mean_S_score, mean_C_index,
            mean_MAE, mean_RMSE).
        """
        if silence:
            _suppress_warnings()

        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        configs = [dict(zip(keys, combo)) for combo in product(*values)]
        total_tasks = len(configs) * n_folds

        print(f"Starting GGS: {len(configs)} configs × {n_folds} folds "
              f"= {total_tasks} tasks")

        results = []
        with tqdm(total=len(configs), desc="GGS progress") as pbar:
            for params in configs:
                result = _run_single_config(
                    params=params,
                    model_class=self.model_class,
                    X_df=self.X_df,
                    y_df=self.y_df,
                    groups_df=self.groups,
                    n_folds=n_folds,
                )
                results.append(result)
                pbar.update(1)

        self.ggs_results_ = results
        return self.ggs_results_

    def get_ggs_results(self, top_n: int = 15) -> pd.DataFrame:
        """Returns ranked grid search results sorted by ascending S-Score.

        Failed configurations (NaN metrics) are ranked last with a
        placeholder S-Score of 999999 and Success=0.

        Args:
            top_n: Number of top configurations to return. Defaults to 15.

        Returns:
            DataFrame with hyperparameter columns plus mean_S_score,
            mean_C_index, mean_MAE, mean_RMSE, and Success columns.

        Raises:
            ValueError: If group_grid_search() has not been run yet.
        """
        if self.ggs_results_ is None:
            raise ValueError(
                "No results available. Run group_grid_search() first."
            )

        df = pd.DataFrame(self.ggs_results_)
        df['Success'] = df['mean_S_score'].apply(
            lambda x: 0 if np.isnan(x) else 1
        )
        df['mean_S_score'] = df['mean_S_score'].fillna(999999)
        df['mean_MAE']     = df['mean_MAE'].fillna(999999)

        df = df.sort_values(
            by=['Success', 'mean_S_score'],
            ascending=[False, True],
        ).head(top_n)

        return df.reset_index(drop=True)