# ./src/metrics_manager.py

from lifelines.utils import concordance_index
from sklearn.metrics import make_scorer
import numpy as np
from typing import Tuple, Dict, Any, Callable

class Metrics:
    """
    A collection of specialized scoring metrics for RUL (Remaining Useful Life) estimation.
    
    This class implements a 'Honest Evaluation' framework where metrics are 
    dynamically synchronized with the model's internal clipping threshold during 
    cross-validation or grid search.
    """

    @staticmethod
    def _comun_values(estimator: Any, X: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the fundamental components for metric calculation.

        This helper synchronizes the evaluation by applying the current model's 
        clipping threshold to the ground truth, ensuring a piecewise RUL comparison.

        Args:
            estimator: The fitted model or pipeline containing the 'model' step.
            X: Input features for prediction.
            y_true: Original linear ground truth targets.

        Returns:
            A tuple containing (prediction_error, piecewise_target, predictions).
        """
        # Retrieve the dynamic threshold from the internal model inside the pipeline
        threshold = estimator.named_steps['model'].clipping_threshold
        
        # Perform prediction
        y_pred = estimator.predict(X)
        
        # Apply piecewise clipping to the ground truth to match the model's design
        y_true_piecewise = np.minimum(y_true, threshold)
        
        # Error defined as (Predicted - Actual)
        diff = y_pred - y_true_piecewise
        
        return diff, y_true_piecewise, y_pred 

    @classmethod
    def s_score_metric(cls, estimator: Any, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the NASA S-score with piecewise honesty.

        The S-score is an asymmetric penalty function. It penalizes late 
        predictions (overestimation) more severely than early ones.

        Penalty terms:
            - If diff < 0 (Early): exp(-diff / 13) - 1
            - If diff > 0 (Late):  exp(diff / 10) - 1

        Returns:
            Negative mean S-score (negated for Scikit-Learn maximization).
        """
        diff, _, _ = cls._comun_values(estimator, X, y_true)
        
        s_score = np.mean(
            np.where(
                diff < 0, 
                np.exp(-diff / 13.0) - 1, 
                np.exp(diff / 10.0) - 1
            )
        )
        
        # Return negative for maximization in GridSearchCV
        return float(s_score)
        
    @classmethod
    def c_index_metric(cls, estimator: Any, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the Concordance Index (C-index).

        Measures the model's ability to correctly rank the relative risk or 
        remaining life of different units. A value of 1.0 represents perfect ranking.

        Returns:
            The concordance index as a float.
        """
        _, y_true_piecewise, y_pred = cls._comun_values(estimator, X, y_true)
        
        c_index = concordance_index(y_true_piecewise, y_pred)
        
        return float(c_index)
    
    @classmethod
    def mae_metric(cls, estimator: Any, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the Piecewise Mean Absolute Error (MAE).

        Measures the average magnitude of error relative to the clipped target.

        Returns:
            Negative MAE (negated for Scikit-Learn maximization).
        """
        diff, _, _ = cls._comun_values(estimator, X, y_true)
        mae = np.mean(np.abs(diff))
        
        return float(mae)
    
    @classmethod
    def rmse_metric(cls, estimator: Any, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the Piecewise Root Mean Squared Error (RMSE).

        Measures the square root of the average of squared errors. 
        Highly sensitive to outliers.

        Returns:
            Negative RMSE (negated for Scikit-Learn maximization).
        """
        diff, _, _ = cls._comun_values(estimator, X, y_true)
        rmse = np.sqrt(np.mean(diff**2))
        
        return float(rmse)
    
    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        """
        Returns a dictionary of scorers compatible with Scikit-Learn.
        Note: We use the functions directly because they follow the 
        (estimator, X, y_true) signature, which is valid for 'scoring' in GridSearch.
        """
        # NO usamos make_scorer aquí si queremos mantener la firma (estimator, X, y)
        # GridSearchCV acepta funciones con firma (estimator, X, y) directamente.
        
        return {
            'S_score': cls.s_score_metric,
            'C_index': cls.c_index_metric,
            'MAE': cls.mae_metric,
            'RMSE': cls.rmse_metric
        }