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

RUL prediction via death curve (Enfoque B):
    The standard survival analysis approach to RUL estimation:

        1. Fit WeibullAFT with t_stop (duration) and evento (event indicator)
        2. For each window, compute the survival function S(t)
        3. Derive the death curve F(t) = 1 - S(t)
        4. Find t* such that F(t*) = confidence_threshold
           → "with confidence_threshold probability the motor has failed by t*"
        5. RUL = t* - t_stop_current

    This is equivalent to:
        t* = predict_percentile(p = 1 - confidence_threshold)
        RUL = t* - t_stop

    But the explicit death curve approach is used here for transparency —
    it exposes S(t) and F(t) for visualization and diagnostic purposes.

Interface contract:
    fit(X, y, t_stop=t_stop, evento=evento):
        Fits the Weibull AFT model using t_stop as duration and evento
        as event indicator.

    predict_survival_function(X):
        Returns S(t) for each window — survival probability over time.

    predict_death_curve(X):
        Returns F(t) = 1 - S(t) for each window — death probability over time.

    predict_with_time(X, t_stop):
        Primary RUL prediction method. Reads t* from F(t) at confidence_threshold,
        then computes RUL = t* - t_stop.

    predict(X):
        Fallback — returns NaN with warning. t_stop is required for RUL.

Known limitations:
    The 0.36% event rate (57 events / 15,814 windows) may cause
    convergence issues or degenerate predictions (S(t)≈1 always). If observed,
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
    features as time-varying covariates. RUL is estimated via the death
    curve F(t) = 1 - S(t): find t* where F(t*) = confidence_threshold,
    then RUL = t* - t_stop_current.

    Args:
        clipping_threshold: Maximum RUL value for prediction clipping.
            GGS hyperparameter. Defaults to 125.
        confidence_threshold: Probability threshold on the death curve F(t).
            t* is the time where F(t*) = confidence_threshold, interpreted as:
            "with confidence_threshold probability the motor has failed by t*."

            Effect on RUL:
                Lower threshold → t* earlier → RUL smaller → alert sooner
                Higher threshold → t* later  → RUL larger  → alert later

            Maintenance interpretation:
                Low threshold (e.g. 0.2): conservative maintenance — alert
                when only 20% of similar motors have failed. Wide safety margin.
                High threshold (e.g. 0.9): aggressive maintenance — alert
                when 90% of similar motors have failed. Narrow safety margin.

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
            "Call fit(X, y, t_stop=t_stop, evento=evento) instead."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object,
    ) -> 'WeibullAFTModel':
        """Fits the Weibull AFT model on sliding window features.

        Uses t_stop as duration_col (absolute cycle from origin to
        event/censure) and evento as event_col. y (y_rul) is accepted
        for interface compatibility but not used in the AFT fit.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            y: y_rul — accepted for API compatibility, not used.
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

        # Build survival DataFrame with t_stop as duration
        df_surv = pd.DataFrame(X, columns=self.feature_names_)
        df_surv['duration'] = t_stop.astype(float)
        df_surv['event'] = (
            evento.astype(int)
            if evento is not None
            else np.ones(len(t_stop), dtype=int)
        )

        # Remove rows with non-positive duration (Weibull requires duration > 0)
        valid_mask = df_surv['duration'] > 0
        n_removed = int((~valid_mask).sum())
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

    def predict_survival_function(
        self,
        X: np.ndarray,
    ) -> pd.DataFrame | None:
        """Returns the survival function S(t) for each window.

        S(t) is the probability that the motor survives beyond time t,
        estimated from the fitted Weibull AFT model given the features X.

        Args:
            X: Feature matrix of shape (n_windows, n_components).

        Returns:
            DataFrame of shape (n_timepoints, n_windows) where:
                - index: time points t
                - columns: window indices 0..n_windows-1
                - values: S(t) ∈ [0, 1]
            None if model is not fitted.
        """
        if not self.is_fitted_ or self.model_ is None:
            return None
        try:
            df_pred = pd.DataFrame(X, columns=self.feature_names_)
            return self.model_.predict_survival_function(df_pred)
        except Exception:
            return None

    def predict_death_curve(
        self,
        X: np.ndarray,
    ) -> pd.DataFrame | None:
        """Returns the death curve F(t) = 1 - S(t) for each window.

        F(t) is the probability that the motor has failed by time t.
        This is the standard basis for RUL estimation in survival analysis:
        find t* such that F(t*) = confidence_threshold.

        Args:
            X: Feature matrix of shape (n_windows, n_components).

        Returns:
            DataFrame of shape (n_timepoints, n_windows) where:
                - index: time points t
                - columns: window indices 0..n_windows-1
                - values: F(t) = 1 - S(t) ∈ [0, 1]
            None if model is not fitted.
        """
        sf = self.predict_survival_function(X)
        if sf is None:
            return None
        return 1.0 - sf

    def predict_with_time(
        self,
        X: np.ndarray,
        t_stop: np.ndarray,
    ) -> np.ndarray:
        """Predicts RUL via the death curve at confidence_threshold.

        Procedure for each window:
            1. Compute death curve F(t) = 1 - S(t)
            2. Find t* = first t where F(t) >= confidence_threshold
               → "with confidence_threshold probability the motor has failed by t*"
            3. RUL = max(t* - t_stop_current, 0), clipped to clipping_threshold

        Effect of confidence_threshold on RUL:
            Low threshold (e.g. 0.2) → t* earlier → smaller RUL → alert sooner
                Conservative maintenance: few motors have failed by t*
            High threshold (e.g. 0.9) → t* later → larger RUL → alert later
                Aggressive maintenance: most motors have failed by t*

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            t_stop: Current absolute cycle of each window (n_windows,).

        Returns:
            Predicted RUL array of shape (n_windows,), clipped to
            clipping_threshold and floored at 0. NaN if not fitted or
            if F(t) never reaches confidence_threshold.
        """
        if not self.is_fitted_ or self.model_ is None:
            return np.full(len(X), np.nan)

        death_curve = self.predict_death_curve(X)
        if death_curve is None:
            return np.full(len(X), np.nan)

        # time points on the death curve axis
        time_points = death_curve.index.to_numpy(dtype=float)
        n_windows = X.shape[0]
        rul_pred = np.full(n_windows, np.nan)

        for w in range(n_windows):
            ft = death_curve.iloc[:, w].to_numpy()
            # Find first t where F(t) >= confidence_threshold
            idx = np.searchsorted(ft, self.confidence_threshold)
            if idx < len(time_points):
                t_star = time_points[idx]
                rul = t_star - float(t_stop[w])
                rul_pred[w] = float(np.clip(max(rul, 0.0),
                                            0.0, self.clipping_threshold))
            # else: F(t) never reaches threshold → remains NaN
            # (degenerate case — S(t)≈1 always, documented negative result)

        return rul_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fallback predict — not usable without t_stop.

        WeibullAFTModel requires t_stop to compute RUL. Use
        predict_with_time(X, t_stop) instead.

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

    def print_summary(self) -> None:
        """Prints the lifelines model summary if fitted."""
        if self.is_fitted_ and self.model_ is not None:
            self.model_.print_summary()
        else:
            print("Model is not fitted.")