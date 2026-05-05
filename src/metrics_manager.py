from lifelines.utils import concordance_index
import numpy as np
from typing import Any


class Metrics:
    """A collection of specialized scoring metrics for RUL estimation.

    Implements an 'Honest Evaluation' framework where metrics are dynamically
    synchronized with the model's internal clipping threshold during cross-validation
    or grid search. Both predictions and ground truth are clipped to the same space
    before metric computation.
    """

    @staticmethod
    def _comun_values(
        estimator: Any,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the fundamental components for metric calculation.

        Synchronizes evaluation by clipping both predictions and ground truth to
        the model's threshold, ensuring a consistent piecewise RUL comparison.
        The threshold is retrieved dynamically from the pipeline's model step.

        Args:
            estimator: Fitted sklearn Pipeline containing a 'model' step with a
                clipping_threshold attribute.
            X: Input feature matrix of shape (n_samples, n_features).
            y_true: Original linear RUL ground truth of shape (n_samples,).
                Values are not pre-clipped.

        Returns:
            Tuple of (diff, y_true_piecewise, y_pred) where:
                diff: Prediction error (y_pred - y_true_piecewise), shape (n_samples,).
                y_true_piecewise: Clipped ground truth, shape (n_samples,).
                y_pred: Clipped predictions, shape (n_samples,).
        """
        threshold: int = estimator.named_steps['model'].clipping_threshold

        y_pred: np.ndarray = np.minimum(estimator.predict(X), threshold)
        y_true_piecewise: np.ndarray = np.minimum(y_true, threshold)

        diff: np.ndarray = y_pred - y_true_piecewise

        return diff, y_true_piecewise, y_pred

    @classmethod
    def s_score_metric(
        cls,
        estimator: Any,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """Calculates the NASA S-score with piecewise honest evaluation.

        Asymmetric penalty function that penalizes late predictions (overestimation)
        more severely than early ones, reflecting the higher operational risk of
        unexpected failure versus premature intervention.

        Penalty terms:
            Early prediction (diff < 0): exp(-diff / 13) - 1
            Late prediction  (diff >= 0): exp(diff / 10) - 1

        Args:
            estimator: Fitted sklearn Pipeline containing a 'model' step.
            X: Input feature matrix of shape (n_samples, n_features).
            y_true: Original linear RUL ground truth of shape (n_samples,).

        Returns:
            Mean S-score as a positive float. Lower values indicate better performance.
        """
        diff, _, _ = cls._comun_values(estimator, X, y_true)

        s_score: float = float(np.mean(
            np.where(
                diff < 0,
                np.exp(-diff / 13.0) - 1,
                np.exp(diff / 10.0) - 1
            )
        ))

        return s_score

    @classmethod
    def c_index_metric(
        cls,
        estimator: Any,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """Calculates the Concordance Index (C-index).

        Measures the model's ability to correctly rank the relative remaining life
        of different units. A value of 0.5 corresponds to random ordering and 1.0
        to perfect discrimination.

        Args:
            estimator: Fitted sklearn Pipeline containing a 'model' step.
            X: Input feature matrix of shape (n_samples, n_features).
            y_true: Original linear RUL ground truth of shape (n_samples,).

        Returns:
            C-index as a float in [0.5, 1.0]. Higher values indicate better ranking.
        """
        _, y_true_piecewise, y_pred = cls._comun_values(estimator, X, y_true)

        return float(concordance_index(y_true_piecewise, y_pred))

    @classmethod
    def mae_metric(
        cls,
        estimator: Any,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """Calculates the Piecewise Mean Absolute Error (MAE).

        Measures the average magnitude of prediction error relative to the clipped
        ground truth. Less sensitive to outliers than RMSE.

        Args:
            estimator: Fitted sklearn Pipeline containing a 'model' step.
            X: Input feature matrix of shape (n_samples, n_features).
            y_true: Original linear RUL ground truth of shape (n_samples,).

        Returns:
            MAE as a positive float. Lower values indicate better performance.
        """
        diff, _, _ = cls._comun_values(estimator, X, y_true)

        return float(np.mean(np.abs(diff)))

    @classmethod
    def rmse_metric(
        cls,
        estimator: Any,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """Calculates the Piecewise Root Mean Squared Error (RMSE).

        Measures the square root of the average squared prediction error relative
        to the clipped ground truth. More sensitive to outliers than MAE due to
        the quadratic nature of the penalty.

        Args:
            estimator: Fitted sklearn Pipeline containing a 'model' step.
            X: Input feature matrix of shape (n_samples, n_features).
            y_true: Original linear RUL ground truth of shape (n_samples,).

        Returns:
            RMSE as a positive float. Lower values indicate better performance.
        """
        diff, _, _ = cls._comun_values(estimator, X, y_true)

        return float(np.sqrt(np.mean(diff ** 2)))

    @classmethod
    def get_metrics(cls) -> dict[str, Any]:
        """Returns a dictionary of scorers compatible with scikit-learn.

        Each scorer follows the (estimator, X, y_true) signature accepted directly
        by GridSearchCV without requiring make_scorer wrapping.

        Returns:
            Dictionary mapping metric names to callable scorer functions.
        """
        return {
            'S_score': cls.s_score_metric,
            'C_index': cls.c_index_metric,
            'MAE': cls.mae_metric,
            'RMSE': cls.rmse_metric
        }