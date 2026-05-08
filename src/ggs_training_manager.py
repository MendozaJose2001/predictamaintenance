"""Group Grid Search training manager for the new sliding window pipeline.

This module implements GGSTrainingManager — the training orchestrator for
hyperparameter optimization of RUL estimation models using the Nodo 2-4
pipeline. It replaces the legacy training_manager.py which was designed for
the old per-motor CSV loading approach.

Key design differences from legacy GGSTrainingManager:
    - No prepare_training_data() — receives X_df and y_df directly
    - No sklearn Pipeline — uses RULPipeline orchestrator (Nodos 2-4)
    - param_grid covers both pipeline and model hyperparameters
    - Pipeline params are separated internally from model params via
      _PIPELINE_PARAMS frozenset

Pipeline hyperparameters in param_grid:
    window_size:         sliding window size (Nodo 2)
    n_components:        PCA components (Nodo 4)
    clipping_threshold:  RUL clipping — affects y_rul only, not X features
    feature_set:         feature combination key for Nodo 3
                         Valid values: 'A', 'B', 'C', 'D'
                         (SET_E discarded — computationally prohibitive)

                         Preliminary ratio-adjusted ranking (ws=30):
                             D (p=176): ratio=7.80 — candidate winner
                             C (p=160): ratio=7.14 — strong candidate
                             B (p=128): ratio=6.51 — pipeline reference
                             A (p=80):  ratio=4.45 — baseline minimal
                         Definitive selection performed by GGS via MAE/RMSE.

Model hyperparameters in param_grid:
    Any parameter accepted by model_class.__init__() except the pipeline
    params above. Examples:
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
from pathlib import Path
from threading import Lock
from typing import cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from statsmodels.tools.sm_exceptions import ConvergenceWarning, DomainWarning, ValueWarning
from tqdm.auto import tqdm

from src.metrics_manager import Metrics
from src.models.base_model import BaseRULModel
from src.pipeline.rul_pipeline import RULPipeline
from src.utils.ggs_io import (
    GGSSession,
    config_key,
    resolve_ggs_session,
    save_checkpoint,
    save_metadata,
    save_results,
)


# Pipeline-level hyperparameters — separated from model params internally.
# Any key in param_grid that appears here is passed to RULPipeline(**),
# all other keys are passed to model_class(**).
_PIPELINE_PARAMS: frozenset[str] = frozenset({
    'window_size',
    'n_components',
    'clipping_threshold',
    'feature_set',
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

            pipeline = RULPipeline(**pipeline_params)
            X_tr, y_rul_tr, t_stop_tr, evento_tr, groups_tr = (
                pipeline.fit_transform(X_train_fold, y_train_fold)
            )
            X_val, y_rul_val, t_stop_val, evento_val, _ = (
                pipeline.transform(X_val_fold, y_val_fold)
            )

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

    mean_metrics: dict[str, float] = {}
    for metric in ['S_score', 'C_index', 'MAE', 'RMSE']:
        values = [f[metric] for f in fold_metrics if not np.isnan(f[metric])]
        mean_metrics[f'mean_{metric}'] = float(np.mean(values)) if values else np.nan

    return {**params, **mean_metrics}


class GGSTrainingManager:
    """Manages group grid search training for the sliding window pipeline.

    Orchestrates hyperparameter optimization for RUL estimation models
    using the Nodo 2-4 pipeline. Both pipeline and model hyperparameters
    are explored jointly in the grid search, including feature_set which
    controls which feature combination is extracted in Nodo 3.

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
                'feature_set':        ['A', 'B', 'C', 'D'],
                'window_size':        [15, 20, 25, 30],
                'n_components':       [10, 15, 20],
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
        checkpoint_every: int = 1,
        base_dir: Path = Path('outputs/ggs'),
        n_jobs: int = 1,
    ) -> list[dict]:
        """Runs a grouped grid search over the provided hyperparameter grid.

        Iterates over all combinations of pipeline and model hyperparameters,
        evaluating each via GroupKFold cross-validation. Supports automatic
        checkpointing and resumption from interrupted runs.

        Parallelism:
            n_jobs=1 (default): sequential evaluation with fold-level progress
                bar and checkpoint after every config.
            n_jobs>1: parallel evaluation across configs using joblib.
                The progress bar advances per config (not per fold).
                Checkpointing uses a thread lock to avoid concurrent writes.
                Resume is fully supported in both modes.

        Session management:
            Each run is identified by (model_class.__name__, param_grid_hash).
            If a checkpoint exists for this pair, the run resumes automatically
            inheriting the previous timestamp. If the param_grid changes, a new
            session is created — the previous checkpoint is not affected.

        Args:
            param_grid: Dictionary mapping parameter names to candidate value
                lists. Pipeline params (window_size, n_components,
                clipping_threshold, feature_set) and model params are mixed
                freely — separated internally via _PIPELINE_PARAMS.
            n_folds: Number of GroupKFold folds. Defaults to 5.
            silence: If True, suppresses convergence and domain warnings.
                Defaults to True.
            checkpoint_every: Save checkpoint every N completed configurations.
                Defaults to 1. Only used in sequential mode (n_jobs=1).
                In parallel mode, checkpoint is saved after each batch.
            base_dir: Root directory for GGS outputs. Defaults to outputs/ggs/.
            n_jobs: Number of parallel workers. 1 = sequential (default).
                -1 = all available cores. Recommended: 2-3 for this pipeline.

        Returns:
            List of dicts, one per configuration, with hyperparameter values
            and mean cross-validated metrics (mean_S_score, mean_C_index,
            mean_MAE, mean_RMSE).
        """
        if silence:
            _suppress_warnings()

        # Build full config list
        keys        = list(param_grid.keys())
        values      = list(param_grid.values())
        all_configs = [dict(zip(keys, combo)) for combo in product(*values)]
        total_configs = len(all_configs)

        # Resolve session — detects resume automatically
        session = resolve_ggs_session(
            model_class=self.model_class,
            param_grid=param_grid,
            base_dir=base_dir,
        )

        # Save metadata only for new sessions
        if not session.is_resume:
            save_metadata(session, param_grid, n_folds, total_configs)

        # Filter out already completed configs
        pending_configs = [
            c for c in all_configs
            if config_key(c) not in session.completed_keys
        ]
        n_completed = total_configs - len(pending_configs)

        print(f"GGS: {total_configs} total configs × {n_folds} folds "
              f"| n_jobs={n_jobs}")
        if session.is_resume:
            print(f"     Resuming — {n_completed} done, "
                  f"{len(pending_configs)} pending")

        # ------------------------------------------------------------------
        # Sequential mode — config-level progress + fine-grained checkpoint
        # ------------------------------------------------------------------
        if n_jobs == 1:
            new_results: list[dict] = []

            for i, params in enumerate(
                tqdm(
                    pending_configs,
                    desc=f"GGS {self.model_class.__name__}",
                    unit='config',
                ),
                start=1,
            ):
                result = _run_single_config(
                    params=params,
                    model_class=self.model_class,
                    X_df=self.X_df,
                    y_df=self.y_df,
                    groups_df=self.groups,
                    n_folds=n_folds,
                )
                new_results.append(result)

                if i % checkpoint_every == 0:
                    save_checkpoint(session, new_results)

        # ------------------------------------------------------------------
        # Parallel mode — config-level progress + lock-protected checkpoint
        # ------------------------------------------------------------------
        else:
            checkpoint_lock = Lock()
            new_results_parallel: list[dict] = []

            def _run_and_checkpoint(params: dict, idx: int) -> dict:
                result = _run_single_config(
                    params=params,
                    model_class=self.model_class,
                    X_df=self.X_df,
                    y_df=self.y_df,
                    groups_df=self.groups,
                    n_folds=n_folds,
                )
                with checkpoint_lock:
                    new_results_parallel.append(result)
                    if idx % checkpoint_every == 0:
                        save_checkpoint(session, new_results_parallel)
                return result

            new_results = cast(list[dict], list(Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(_run_and_checkpoint)(params, i)
                for i, params in tqdm(
                    enumerate(pending_configs, start=1),
                    total=len(pending_configs),
                    desc=f"GGS {self.model_class.__name__}",
                    unit='config',
                )
            )))

        # Save final results and remove checkpoint
        save_results(session, new_results)

        # Store all results (prior + new) in manager
        self.ggs_results_ = session.prior_results + new_results
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