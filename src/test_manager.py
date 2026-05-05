import numpy as np
import pandas as pd
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

from src.metrics_manager import Metrics
from src.models.base_model import BaseRULModel


class TestManager:
    """Manages model evaluation across all four assessment scenarios.

    Evaluates a trained model on both training and test partitions, in both
    trajectory mode (full degradation history) and deployment mode (last
    observed cycle per motor). Results are returned as a single DataFrame
    matching the reporting format used in the paper.

    Decouples the fitting target (y_fit) from the metrics target (y_metrics),
    enabling correct evaluation for both regression models (where y_fit ==
    y_metrics) and survival models (where y_fit is structured and y_metrics
    is scalar RUL).

    Args:
        model: An instance of a BaseRULModel subclass. Used to prepare
            both training and test data via prepare_training_data.
        train_ids: Array of motor unit identifiers for the training partition.
        test_ids: Array of motor unit identifiers for the test partition.
    """

    def __init__(
        self,
        model: BaseRULModel,
        train_ids: np.ndarray,
        test_ids: np.ndarray
    ) -> None:
        X_train, y_fit_train, y_metrics_train, groups_train = (
            model.prepare_training_data(train_ids)
        )
        X_test, y_fit_test, y_metrics_test, groups_test = (
            model.prepare_training_data(test_ids)
        )

        self.model = model
        self.X_train = X_train
        self.y_fit_train = y_fit_train
        self.y_metrics_train = y_metrics_train
        self.groups_train = groups_train
        self.X_test = X_test
        self.y_fit_test = y_fit_test
        self.y_metrics_test = y_metrics_test
        self.groups_test = groups_test

    def _get_last_indices(
        self,
        groups: np.ndarray
    ) -> np.ndarray:
        """Returns the indices of the last observed cycle per motor unit.

        Args:
            groups: Integer array mapping each row to its motor unit identifier.

        Returns:
            Array of integer indices corresponding to the last row of each
            motor unit in the dataset.
        """
        return (
            pd.Series(groups)
            .groupby(groups)
            .tail(1)
            .index
            .to_numpy()
        )

    def _evaluate_scenario(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y_metrics: np.ndarray,
        groups: np.ndarray,
        only_last: bool
    ) -> dict:
        """Evaluates a fitted pipeline on a single scenario.

        Args:
            pipeline: Fitted sklearn Pipeline containing scaler and model steps.
                The scaler must have been fitted on the training partition.
            X: Feature matrix for evaluation.
            y_metrics: Scalar RUL target array for metric computation.
            groups: Integer array mapping each row to its motor unit identifier.
            only_last: If True, evaluates only the last observed cycle per
                motor (deployment mode). If False, evaluates all cycles
                (trajectory mode).

        Returns:
            Dictionary with keys N, S-Score, C-Index, MAE, RMSE.
        """
        if only_last:
            last_idx = self._get_last_indices(groups)
            X_eval = X.iloc[last_idx].reset_index(drop=True)
            y_eval = y_metrics[last_idx]
        else:
            X_eval = X
            y_eval = y_metrics

        metrics_funcs = Metrics.get_metrics()
        results = {
            name: func(pipeline, X_eval, y_eval)
            for name, func in metrics_funcs.items()
        }

        return {
            'N': len(X_eval),
            'S-Score': results['S_score'],
            'C-Index': results['C_index'],
            'MAE': results['MAE'],
            'RMSE': results['RMSE']
        }

    def evaluate_best_model(
        self,
        param_grid: dict
    ) -> pd.DataFrame:
        """Evaluates the best model configuration across all four scenarios.

        Retrains the model with the provided hyperparameters on the full
        training partition using y_fit_train, with the MADScaler fitted
        exclusively on training data. Evaluates the fitted pipeline in four
        scenarios using y_metrics for consistent metric computation:
            - Train | Trajectory: all cycles of training motors
            - Train | Deployment: last cycle of training motors
            - Test  | Trajectory: all cycles of test motors
            - Test  | Deployment: last cycle of test motors

        Args:
            param_grid: Dictionary of hyperparameters for the winning
                configuration. Keys must NOT include the 'model__' prefix
                (e.g. {'alpha': 1.5, 'link_type': 'log'}).

        Returns:
            DataFrame with columns Partition, Mode, N, S-Score, C-Index,
            MAE, RMSE. Rows are ordered as Train/Trajectory, Train/Deployment,
            Test/Trajectory, Test/Deployment.
        """
        # Instantiate model with winning hyperparameters
        model = self.model.__class__(**param_grid)

        # Fit pipeline exclusively on training data using y_fit
        pipeline = Pipeline([
            ('scaler', MADScaler()),
            ('model', model)
        ])
        pipeline.set_output(transform='pandas')
        pipeline.fit(
            self.X_train,
            self.y_fit_train,
            model__groups=self.groups_train
        )

        # Evaluate all four scenarios using y_metrics
        scenarios = [
            ('Train', 'Trajectory', self.X_train, self.y_metrics_train, self.groups_train, False),
            ('Train', 'Deployment', self.X_train, self.y_metrics_train, self.groups_train, True),
            ('Test',  'Trajectory', self.X_test,  self.y_metrics_test,  self.groups_test,  False),
            ('Test',  'Deployment', self.X_test,  self.y_metrics_test,  self.groups_test,  True),
        ]

        rows = []
        for partition, mode, X, y_metrics, groups, only_last in scenarios:
            metrics = self._evaluate_scenario(pipeline, X, y_metrics, groups, only_last)
            rows.append({
                'Partition': partition,
                'Mode': mode,
                **metrics
            })

        return pd.DataFrame(rows)