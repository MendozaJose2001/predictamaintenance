"""Weibull Accelerated Failure Time model for RUL estimation.

This module implements WeibullAFTModel — a survival analysis approach to
RUL prediction using the Weibull Accelerated Failure Time (AFT) regression
model from the lifelines library.

Motivation:
    Cox Proportional Hazards with frailty was evaluated and discarded due
    to non-identifiability of the frailty parameter θ with the low event
    rate (0.36%) produced by the sliding window pipeline on C-MAPSS FD001.
    Weibull AFT is evaluated as an alternative because:
    - It models survival time directly (log(T) = Xβ + σε) rather than
      the hazard ratio, requiring fewer events for parameter estimation.
    - The parametric Weibull baseline reduces degrees of freedom compared
      to Cox's non-parametric baseline.
    - lifelines provides a robust implementation with built-in convergence
      handling.

Duration variable:
    The AFT model uses t_stop as duration_col — the absolute cycle at which
    each window ends. This is the correct choice because:
    - AFT requires time elapsed from a fixed origin (cycle 0) to event/censure
    - t_stop represents accumulated operating time — the standard duration
      variable in survival analysis
    - y_rul is NOT used as duration because it is a decreasing variable
      (reaching 0 at failure), which is inadmissible in Weibull AFT

RUL prediction:
    The AFT model estimates the total lifetime distribution of the motor.
    RUL at window w is obtained by:

        RUL_pred = predict_percentile(X_w, p) - t_stop_w

    Where p = 1 - confidence_threshold. This subtracts the current cycle
    from the predicted total lifetime to obtain remaining cycles.

Interface contract:
    fit(X, y_rul, t_stop, evento, groups=None):
        Fits the Weibull AFT model using t_stop as duration and evento
        as event indicator.

    predict_with_time(X, t_stop):
        Primary prediction method. Returns RUL = percentile(X) - t_stop.

    predict(X):
        Fallback — raises RuntimeError. t_stop is required for prediction.
        Present for BaseRULModel interface compatibility only.

Known limitations:
    The 0.36% event rate (57 events / 15,814 windows) may cause
    convergence issues or degenerate predictions (S(t)=1). If observed,
    this confirms that survival models are structurally insufficient for
    C-MAPSS under sliding windows — a documented negative result.
"""

import warnings

import numpy as np
import pandas as pd

from src.models.base_model import BaseRULModel


class WeibullAFTModel(BaseRULModel):
    """Weibull AFT regression model for RUL estimation via lifelines.

    Models the total lifetime distribution of motors using sliding window
    features as time-varying covariates. RUL is estimated as the difference
    between the predicted total lifetime percentile and the current cycle.

    Args:
        clipping_threshold: Maximum RUL value for prediction clipping.
            GGS hyperparameter. Defaults to 125.
        confidence_threshold: Confidence level for RUL prediction.
            Passed directly as p to lifelines.predict_percentile(p).
            lifelines convention: predict_percentile(p) returns t where S(t)=p.

            Interpretation:
                confidence_threshold=0.8 → p=0.8 → S(t)=0.8
                → "with 80% confidence the motor still survives at t"
                → 20% of similar motors have failed by t
                Higher values → more conservative estimates (motor still alive).
                Lower values → more aggressive (most motors have failed by t).
            GGS hyperparameter. Defaults to 0.8.
        penalizer: L2 regularization strength. Defaults to 0.0.
        l1_ratio: Elastic net mixing (0=L2, 1=L1). Defaults to 0.0.
        fit_intercept: Whether to fit an intercept. Defaults to True.

    Attributes:
        model_: Fitted WeibullAFTFitter. Available after fit().
        is_fitted_: True after fit() completes successfully.
        feature_names_: PC column names used during fit().
    """

    def __init__(
        self,
        clipping_threshold: int = 125,
        confidence_threshold: float = 0.8,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
    ) -> None:
        self.clipping_threshold = clipping_threshold
        self.confidence_threshold = confidence_threshold
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.is_fitted_: bool = False
        self.model_ = None
        self.feature_names_: list[str] = []

    def prepare_training_data(self, list_ids: np.ndarray) -> tuple:
        raise NotImplementedError(
            "WeibullAFTModel uses the sliding window pipeline directly. "
            "Call fit(X, y_rul, t_stop=t_stop, evento=evento) instead."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object,
    ) -> 'WeibullAFTModel':
        """Fits the Weibull AFT model on sliding window features.

        Uses t_stop as duration_col (absolute cycle — time from origin to
        event/censure) and evento as event_col. y (y_rul) is accepted for
        interface compatibility but not used in the AFT fit.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            y: y_rul array — accepted for API compatibility, not used.
            **kwargs:
                t_stop (np.ndarray): Absolute cycle of each window. Required.
                evento (np.ndarray): Event indicator. 1=failure, 0=censored.
                groups (np.ndarray): Motor ID per window — ignored.

        Returns:
            self
        """
        t_stop: np.ndarray | None = kwargs.get('t_stop', None)  # type: ignore
        evento: np.ndarray | None = kwargs.get('evento', None)  # type: ignore
        try:
            from lifelines import WeibullAFTFitter
        except ImportError:
            raise ImportError(
                "lifelines is required. Install with: pip install lifelines"
            )

        if t_stop is None:
            warnings.warn(
                "t_stop is required for WeibullAFTModel. "
                "Fit failed — t_stop=None.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.is_fitted_ = False
            return self

        # Build feature column names
        n_features = X.shape[1]
        self.feature_names_ = [f'PC_{i+1}' for i in range(n_features)]

        # Build survival DataFrame
        df_surv = pd.DataFrame(X, columns=self.feature_names_)
        df_surv['duration'] = t_stop.astype(float)
        df_surv['event'] = (
            evento.astype(int)
            if evento is not None
            else np.ones(len(t_stop), dtype=int)
        )

        # Remove rows with non-positive duration (Weibull requires duration > 0)
        valid_mask = df_surv['duration'] > 0
        n_removed = (~valid_mask).sum()
        if n_removed > 0:
            warnings.warn(
                f"Removed {n_removed} rows with duration <= 0.",
                RuntimeWarning,
                stacklevel=2,
            )
        df_surv = df_surv[valid_mask].reset_index(drop=True)

        if len(df_surv) == 0:
            warnings.warn(
                "No valid rows after filtering duration > 0. Fit failed.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.is_fitted_ = False
            return self

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.model_ = WeibullAFTFitter(
                    penalizer=self.penalizer,
                    l1_ratio=self.l1_ratio,
                    fit_intercept=self.fit_intercept,
                )
                self.model_.fit(
                    df_surv,
                    duration_col='duration',
                    event_col='event',
                )
            self.is_fitted_ = True

        except Exception as e:
            warnings.warn(
                f"WeibullAFTModel fit failed: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.is_fitted_ = False

        return self

    def predict_with_time(
        self,
        X: np.ndarray,
        t_stop: np.ndarray,
    ) -> np.ndarray:
        """Predicts RUL as predicted_lifetime_percentile - t_stop_current.

        The AFT model estimates the total lifetime T of the motor. RUL at
        the current window is obtained by subtracting the current cycle:

            RUL = predict_percentile(X, p=1-confidence_threshold) - t_stop

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            t_stop: Current cycle of each window (n_windows,).

        Returns:
            Predicted RUL array of shape (n_windows,), clipped to
            clipping_threshold and floored at 0. NaN if not fitted.
        """
        if not self.is_fitted_ or self.model_ is None:
            return np.full(len(X), np.nan)

        p = self.confidence_threshold

        try:
            df_pred = pd.DataFrame(X, columns=self.feature_names_)
            lifetime_pred = self.model_.predict_percentile(
                df_pred, p=p
            ).to_numpy()

            # RUL = predicted total lifetime - current cycle
            rul_pred = lifetime_pred - t_stop.astype(float)

            # Floor at 0 — negative RUL means the model predicts past failure
            rul_pred = np.where(
                np.isfinite(rul_pred),
                np.maximum(rul_pred, 0.0),
                np.nan,
            )
            return np.minimum(rul_pred, float(self.clipping_threshold))

        except Exception:
            return np.full(len(X), np.nan)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fallback predict — not usable without t_stop.

        WeibullAFTModel requires t_stop to compute RUL as:
            RUL = predict_percentile(X) - t_stop

        Use predict_with_time(X, t_stop) instead.

        Args:
            X: Feature matrix — accepted for interface compatibility.

        Returns:
            NaN array of shape (n_windows,) with a RuntimeWarning.
        """
        warnings.warn(
            "WeibullAFTModel.predict() cannot compute RUL without t_stop. "
            "Use predict_with_time(X, t_stop) instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.full(len(X), np.nan)

    def predict_percentile(
        self,
        X: np.ndarray,
        p: float | None = None,
    ) -> np.ndarray:
        """Returns raw predicted lifetime percentile (not RUL).

        Returns the predicted total lifetime T at percentile p — without
        subtracting t_stop. Useful for inspection and debugging.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            p: Percentile (0-1). Defaults to 1 - confidence_threshold.

        Returns:
            Predicted lifetime array. NaN if not fitted.
        """
        if not self.is_fitted_ or self.model_ is None:
            return np.full(len(X), np.nan)

        if p is None:
            p = self.confidence_threshold

        try:
            df_pred = pd.DataFrame(X, columns=self.feature_names_)
            return self.model_.predict_percentile(df_pred, p=p).to_numpy()
        except Exception:
            return np.full(len(X), np.nan)

    def print_summary(self) -> None:
        """Prints the lifelines model summary if fitted."""
        if self.is_fitted_ and self.model_ is not None:
            self.model_.print_summary()
        else:
            print("Model is not fitted.")