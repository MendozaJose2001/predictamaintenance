import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from src.mad_scaler import MADScaler

"""MADScaler uses median + MAD scaling, equivalent to R's mad() function.
This is required for frailtyPenal convergence (CoxFrailty model). If you
want to switch back to sklearn's RobustScaler (IQR-based) for non-frailty
models, replace the import and pipeline step as follows:

from sklearn.preprocessing import RobustScaler
pipeline = Pipeline([('scaler', RobustScaler()), ('model', model)])

Note that RobustScaler will cause istop=2 failures in CoxFrailty.
"""
from statsmodels.tools.sm_exceptions import ConvergenceWarning, DomainWarning, ValueWarning
from tqdm.auto import tqdm

from src.metrics_manager import Metrics
from src.models.base_model import BaseRULModel


def _suppress_ggs_warnings() -> None:
    """Suppresses common convergence and domain warnings during grid search.

    Silences warnings that are expected during hyperparameter exploration,
    including statsmodels ConvergenceWarning, DomainWarning, and ValueWarning,
    as well as any RuntimeWarning emitted by model fit failures. This prevents
    the terminal from being flooded during large grid searches while preserving
    the ability to detect genuine errors via error_score in the manual loop.
    """
    warnings.filterwarnings("ignore")
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', DomainWarning)
    warnings.simplefilter('ignore', ValueWarning)
    os.environ['PYTHONWARNINGS'] = 'ignore'


def _evaluate_fold(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_metrics_val: np.ndarray
) -> dict[str, float]:
    """Evaluates a fitted pipeline on a validation fold using all four metrics.

    Args:
        pipeline: Fitted sklearn Pipeline containing scaler and model steps.
        X_val: Validation feature matrix.
        y_metrics_val: Scalar RUL array for the validation fold.

    Returns:
        Dictionary with keys S_score, C_index, MAE, RMSE.
    """
    metrics_funcs = Metrics.get_metrics()
    return {
        name: func(pipeline, X_val, y_metrics_val)
        for name, func in metrics_funcs.items()
    }


def _run_single_config(
    params: dict,
    model_class: type,
    X: pd.DataFrame,
    y_fit: np.ndarray,
    y_metrics: np.ndarray,
    groups: np.ndarray,
    n_folds: int
) -> dict:
    """Evaluates a single hyperparameter configuration via GroupKFold CV.

    Builds a fresh pipeline for each fold, fits it on the training split
    using y_fit, and evaluates it on the validation split using y_metrics.
    Averages metrics across all folds.

    Args:
        params: Dictionary of hyperparameter values for this configuration.
            Keys must not include the 'model__' prefix.
        model_class: The BaseRULModel subclass to instantiate.
        X: Full feature matrix.
        y_fit: Target array passed to pipeline.fit().
        y_metrics: Scalar RUL array passed to scoring functions.
        groups: Motor unit identifiers for GroupKFold splitting.
        n_folds: Number of GroupKFold folds.

    Returns:
        Dictionary containing the hyperparameter values and mean metrics
        across all folds, with NaN values if all folds failed.
    """
    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, val_idx in gkf.split(X, y_fit, groups):
        try:
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_fit_train = y_fit[train_idx]
            y_metrics_val = y_metrics[val_idx]
            groups_train_fold = groups[train_idx]

            model = model_class(**params)
            pipeline = Pipeline([
                ('scaler', MADScaler()),
                ('model', model)
            ])
            pipeline.set_output(transform='pandas')
            pipeline.fit(
                X_train_fold,
                y_fit_train,
                model__groups=groups_train_fold
            )
            fold_metrics.append(_evaluate_fold(pipeline, X_val_fold, y_metrics_val))

        except Exception:
            fold_metrics.append({
                'S_score': np.nan,
                'C_index': np.nan,
                'MAE': np.nan,
                'RMSE': np.nan
            })

    # Average metrics across folds
    mean_metrics: dict[str, float] = {}
    for metric in ['S_score', 'C_index', 'MAE', 'RMSE']:
        values = [f[metric] for f in fold_metrics if not np.isnan(f[metric])]
        mean_metrics[f'mean_{metric}'] = float(np.mean(values)) if values else np.nan

    return {**params, **mean_metrics}


class GGSTrainingManager:
    """Manages group grid search training for RUL estimation models.

    Implements a manual GroupKFold cross-validation loop that decouples the
    target used for fitting (y_fit) from the target used for metric evaluation
    (y_metrics). This design supports both regression models (where y_fit ==
    y_metrics) and survival models (where y_fit is a structured array and
    y_metrics is scalar RUL), without relying on sklearn's GridSearchCV single-y
    constraint.

    Args:
        model: An instance of a BaseRULModel subclass. Used to prepare
            training data via prepare_training_data.
        list_ids: Array of motor unit identifiers whose CSV files will be
            loaded from data/clean/ via the model's prepare_training_data.
    """

    def __init__(self, model: BaseRULModel, list_ids: np.ndarray) -> None:
        X_train, y_fit_train, y_metrics_train, groups_train = (
            model.prepare_training_data(list_ids)
        )

        self.model = model
        self.X_train = X_train
        self.y_fit_train = y_fit_train
        self.y_metrics_train = y_metrics_train
        self.groups_train = groups_train
        self.ggs_results_: list[dict] | None = None

    def get_training_data(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the prepared training data tuple.

        Provides access to the feature matrix, fit target, metrics target,
        and groups array assembled during initialization.

        Returns:
            Tuple of (X_train, y_fit_train, y_metrics_train, groups_train).
        """
        return (
            self.X_train,
            self.y_fit_train,
            self.y_metrics_train,
            self.groups_train
        )

    def group_grid_search(
        self,
        param_grid: dict,
        n_folds: int = 5,
        silence: bool = True
    ) -> list[dict]:
        """Runs a grouped grid search over the provided hyperparameter grid.

        Iterates over all hyperparameter combinations and evaluates each using
        GroupKFold cross-validation. Groups are defined by motor unit identifiers
        to prevent temporal leakage between folds. For each configuration, y_fit
        is passed to pipeline.fit() and y_metrics is passed to scoring functions,
        enabling correct evaluation for both regression and survival model families.

        Args:
            param_grid: Dictionary mapping model parameter names to lists of
                candidate values. Keys must NOT include the 'model__' prefix
                (e.g. {'alpha': [0.5, 1.0], 'clipping_threshold': [110, 125]}).
            n_folds: Number of folds for GroupKFold cross-validation.
                Defaults to 5.
            silence: If True, suppresses convergence and domain warnings
                during the search to avoid terminal flooding. Defaults to True.

        Returns:
            List of dictionaries, one per configuration, containing
            hyperparameter values and mean cross-validated metrics.
        """
        if silence:
            _suppress_ggs_warnings()

        model_class = self.model.__class__
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        configs = [dict(zip(keys, combo)) for combo in product(*values)]
        total_tasks = len(configs) * n_folds

        print(f"Starting Grid Search: {len(configs)} configs × {n_folds} folds "
              f"= {total_tasks} tasks")

        results = []
        with tqdm(total=len(configs), desc="GGS progress") as pbar:
            for params in configs:
                result = _run_single_config(
                    params=params,
                    model_class=model_class,
                    X=self.X_train,
                    y_fit=self.y_fit_train,
                    y_metrics=self.y_metrics_train,
                    groups=self.groups_train,
                    n_folds=n_folds
                )
                results.append(result)
                pbar.update(1)

        self.ggs_results_ = results
        return self.ggs_results_

    def get_ggs_results(self, top_n: int = 15) -> pd.DataFrame:
        """Extracts and ranks grid search results by ascending S-Score.

        Parses the results list from group_grid_search, flags failed
        configurations, and returns the top_n best configurations sorted
        by S-Score. Failed configurations are ranked last.

        Args:
            top_n: Number of top configurations to return. Defaults to 15.

        Returns:
            DataFrame with hyperparameter columns plus mean_S_score,
            mean_C_index, mean_MAE, mean_RMSE, and Success columns,
            sorted by ascending S-Score with failed configurations last.

        Raises:
            ValueError: If group_grid_search has not been run yet.
        """
        if self.ggs_results_ is None:
            raise ValueError(
                "No grid search results available. Run group_grid_search() first."
            )

        df = pd.DataFrame(self.ggs_results_)

        df['Success'] = df['mean_S_score'].apply(lambda x: 0 if np.isnan(x) else 1)
        df['mean_S_score'] = df['mean_S_score'].fillna(999999)
        df['mean_MAE'] = df['mean_MAE'].fillna(999999)

        df = df.sort_values(
            by=['Success', 'mean_S_score'],
            ascending=[False, True]
        ).head(top_n)

        return df.reset_index(drop=True)