#./src/metrics_manager.py

"""Metrics for RUL estimation evaluation.

This module implements the four metrics used to evaluate all RUL models
in PredictaMaintenance. All metrics operate directly on prediction and
ground truth arrays — no sklearn Pipeline dependency.

Metrics:
    S-Score:  NASA asymmetric penalty. Penalizes late predictions more
              severely than early ones. Lower is better.
    C-Index:  Concordance index. Measures ranking quality. Higher is better.
    MAE:      Mean absolute error on piecewise-clipped RUL. Lower is better.
    RMSE:     Root mean squared error on piecewise-clipped RUL. Lower is better.

Honest evaluation:
    All metrics clip both predictions and ground truth to clipping_threshold
    before computation. This ensures evaluation in the same piecewise RUL
    space used during training — preventing artificially low errors from
    the flat region where all motors are healthy.
"""

from typing import Callable
import numpy as np
from lifelines.utils import concordance_index


class Metrics:
    """Collection of RUL evaluation metrics.

    All methods receive predictions and ground truth as numpy arrays
    directly, along with the clipping threshold for honest evaluation.
    No sklearn Pipeline dependency.
    """

    @staticmethod
    def _clip_both(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        clipping_threshold: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Clips both predictions and ground truth to clipping_threshold.

        Args:
            y_pred: Predicted RUL array of shape (n_samples,).
            y_true: True RUL array of shape (n_samples,).
            clipping_threshold: Maximum RUL value for clipping.

        Returns:
            Tuple of (diff, y_true_clipped, y_pred_clipped) where:
                diff:           y_pred_clipped - y_true_clipped
                y_true_clipped: ground truth clipped to threshold
                y_pred_clipped: predictions clipped to threshold
        """
        y_pred_clipped = np.minimum(y_pred, clipping_threshold)
        y_true_clipped = np.minimum(y_true, clipping_threshold)
        diff = y_pred_clipped - y_true_clipped
        return diff, y_true_clipped, y_pred_clipped

    @classmethod
    def s_score(
        cls,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        clipping_threshold: int = 125,
    ) -> float:
        """NASA S-Score — asymmetric penalty for late vs early predictions.

        Penalizes late predictions (overestimation) more severely than
        early ones, reflecting the higher operational risk of unexpected
        failure versus premature maintenance intervention.

        Penalty:
            diff < 0  (early): exp(-diff / 13) - 1
            diff >= 0 (late):  exp( diff / 10) - 1

        Args:
            y_pred: Predicted RUL array.
            y_true: True RUL array.
            clipping_threshold: RUL clipping value. Defaults to 125.

        Returns:
            Mean S-Score as float. Lower is better.
        """
        diff, _, _ = cls._clip_both(y_pred, y_true, clipping_threshold)
        return float(np.mean(
            np.where(
                diff < 0,
                np.exp(-diff / 13.0) - 1,
                np.exp(diff  / 10.0) - 1,
            )
        ))

    @classmethod
    def c_index(
        cls,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        clipping_threshold: int = 125,
    ) -> float:
        """Concordance Index — ranking quality of RUL predictions.

        Measures the proportion of pairs correctly ranked by the model.
        0.5 = random, 1.0 = perfect discrimination.

        Args:
            y_pred: Predicted RUL array.
            y_true: True RUL array.
            clipping_threshold: RUL clipping value. Defaults to 125.

        Returns:
            C-Index as float in [0, 1]. Higher is better.
        """
        _, y_true_clipped, y_pred_clipped = cls._clip_both(
            y_pred, y_true, clipping_threshold
        )
        return float(concordance_index(y_true_clipped, y_pred_clipped))

    @classmethod
    def mae(
        cls,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        clipping_threshold: int = 125,
    ) -> float:
        """Mean Absolute Error on piecewise-clipped RUL.

        Args:
            y_pred: Predicted RUL array.
            y_true: True RUL array.
            clipping_threshold: RUL clipping value. Defaults to 125.

        Returns:
            MAE as float. Lower is better.
        """
        diff, _, _ = cls._clip_both(y_pred, y_true, clipping_threshold)
        return float(np.mean(np.abs(diff)))

    @classmethod
    def rmse(
        cls,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        clipping_threshold: int = 125,
    ) -> float:
        """Root Mean Squared Error on piecewise-clipped RUL.

        Args:
            y_pred: Predicted RUL array.
            y_true: True RUL array.
            clipping_threshold: RUL clipping value. Defaults to 125.

        Returns:
            RMSE as float. Lower is better.
        """
        diff, _, _ = cls._clip_both(y_pred, y_true, clipping_threshold)
        return float(np.sqrt(np.mean(diff ** 2)))

    @classmethod
    def get_metrics(cls) -> dict[str, Callable[..., float]]:
        """Returns all metrics as a dict of callables.

        Each callable has signature:
            func(y_pred, y_true, clipping_threshold=125) -> float

        Returns:
            Dict mapping metric names to callable functions.
        """
        return {
            'S_score': cls.s_score,
            'C_index': cls.c_index,
            'MAE':     cls.mae,
            'RMSE':    cls.rmse,
        }