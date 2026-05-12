#./src/models/cox_ph.py

"""Cox Proportional Hazards model for RUL estimation.

This module implements CoxPHModel — a survival analysis approach to RUL
prediction using the Cox Proportional Hazards model from the lifelines library.

Motivation:
    Weibull AFT was evaluated and documented as producing near-trivial predictions
    on C-MAPSS FD001 under the sliding window pipeline, due to the extremely low
    event rate (0.36% — 57 events / ~15,814 windows). Cox PH is evaluated as an
    alternative semi-parametric survival model because:
    - It makes no parametric assumption on the baseline hazard h₀(t), relying
      instead on Breslow's non-parametric estimator or parametric splines.
    - The partial likelihood estimation is more robust to low event rates than
      full likelihood methods (AFT), since it conditions on observed event times.
    - baseline_estimation_method='spline' provides a parametric alternative that
      may interpolate the baseline hazard more smoothly for sparse event data.

Duration variable:
    Uses t_stop as duration_col — the absolute cycle at which each window ends.
    This is consistent with WeibullAFTModel and correct because:
    - Cox requires time elapsed from a fixed origin (cycle 0) to event/censure.
    - t_stop represents accumulated operating time on a shared absolute axis
      across all motors, enabling Cox to estimate a meaningful baseline hazard.
    - window_size (t_stop - t_start) is constant across all windows for a given
      pipeline configuration, making it unsuitable as duration — the baseline
      hazard would collapse to a single time point.

RUL prediction via death curve:
    Identical procedure to WeibullAFTModel (Enfoque B):
        1. Fit CoxPH with t_stop (duration) and evento (event indicator)
        2. For each window, compute S(t|X) via predict_survival_function()
        3. Derive death curve F(t|X) = 1 - S(t|X)
        4. Find t* = first t where F(t*) >= confidence_threshold
        5. RUL = max(t* - t_stop_current, 0), clipped to clipping_threshold

    confidence_threshold is treated as a GGS hyperparameter — lower values
    trigger earlier maintenance alerts (conservative), higher values later
    alerts (aggressive). See WeibullAFTModel.predict_with_time() for rationale.

baseline_estimation_method:
    'breslow' (default): non-parametric baseline hazard via Breslow's method.
        Standard semi-parametric Cox model. Ties handled via Efron's method.
    'spline': parametric baseline via cubic splines, controlled by
        n_baseline_knots. More flexible interpolation, potentially better
        for sparse event data. n_baseline_knots is a GGS hyperparameter
        but is silently ignored by lifelines when method='breslow', following
        the same convention used by SVRModel where kernel-specific parameters
        (degree, gamma) are always passed regardless of the active kernel.

Known limitations:
    The 0.36% event rate may cause degenerate predictions (S(t)≈1 always),
    consistent with the negative result documented for WeibullAFTModel. If
    observed, this confirms that survival models under the sliding window
    pipeline are structurally insufficient for C-MAPSS FD001 — a documented
    negative result with methodological value.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.models.base_model import BaseRULModel


class CoxPHModel(BaseRULModel, BaseEstimator, RegressorMixin):
    """Cox Proportional Hazards model for RUL estimation via lifelines.

    Models the hazard function h(t|X) = h₀(t) · exp(β'X) using sliding
    window PCA features as covariates. RUL is estimated via the death
    curve F(t|X) = 1 - S(t|X): find t* where F(t*) = confidence_threshold,
    then RUL = t* - t_stop_current.

    The baseline hazard h₀(t) is estimated either non-parametrically
    (Breslow) or parametrically (cubic splines), controlled by
    baseline_estimation_method.

    Args:
        clipping_threshold: Maximum RUL value for prediction clipping.
            GGS hyperparameter. Defaults to 125.
        confidence_threshold: Probability threshold on the death curve F(t).
            t* is the first time where F(t*) >= confidence_threshold.
            Lower values trigger earlier alerts (conservative maintenance).
            Higher values trigger later alerts (aggressive maintenance).
            GGS hyperparameter. Defaults to 0.8.
        baseline_estimation_method: How to estimate the baseline hazard h₀(t).
            'breslow': non-parametric (standard semi-parametric Cox model).
            'spline': parametric cubic splines, requires n_baseline_knots.
            GGS hyperparameter. Defaults to 'breslow'.
        n_baseline_knots: Number of knots for spline baseline estimation.
            Only active when baseline_estimation_method='spline'. Silently
            ignored by lifelines when method='breslow', consistent with the
            SVRModel convention for kernel-specific parameters. GGS
            hyperparameter. Defaults to 3.
        penalizer: L2 regularization strength. Larger values shrink
            coefficients toward zero, reducing overfitting on PCA components.
            GGS hyperparameter. Defaults to 0.0.
        l1_ratio: Elastic net mixing parameter (0=L2, 1=L1). Only active
            when penalizer > 0. GGS hyperparameter. Defaults to 0.0.

    Attributes:
        model_: Fitted CoxPHFitter instance. Available after fit().
        is_fitted_: True after fit() completes successfully.
        feature_names_: PC column names used during fit(). Available
            after fit().
    """

    def __init__(
        self,
        clipping_threshold: int = 125,
        confidence_threshold: float = 0.8,
        baseline_estimation_method: str = 'breslow',
        n_baseline_knots: int = 3,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
    ) -> None:
        self.clipping_threshold = clipping_threshold
        self.confidence_threshold = confidence_threshold
        self.baseline_estimation_method = baseline_estimation_method
        self.n_baseline_knots = n_baseline_knots
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.is_fitted_: bool = False
        self.model_ = None
        self.feature_names_: list[str] = []

    def prepare_training_data(self, list_ids: np.ndarray) -> tuple:
        raise NotImplementedError(
            "CoxPHModel uses the sliding window pipeline directly. "
            "Call fit(X, y, t_stop=t_stop, evento=evento) instead."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object,
    ) -> 'CoxPHModel':
        """Fits the Cox PH model on sliding window features.

        Uses t_stop as duration_col (absolute cycle from origin to
        event/censure) and evento as event_col. y (y_rul) is accepted
        for interface compatibility but not used in the Cox fit.

        When baseline_estimation_method='spline', n_baseline_knots
        controls the flexibility of the parametric baseline hazard.
        When baseline_estimation_method='breslow', n_baseline_knots
        is passed to the constructor but silently ignored by lifelines —
        consistent with the SVRModel convention for kernel-specific
        parameters (degree, gamma) that are always passed regardless
        of the active kernel.

        On fitting failure, emits RuntimeWarning and sets is_fitted_=False,
        allowing the GGS loop to handle failed configurations gracefully.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            y: y_rul — accepted for API compatibility, not used.
            **kwargs:
                t_stop (np.ndarray): Absolute cycle of each window. Required.
                evento (np.ndarray): Event indicator. 1=failure, 0=censored.
                groups (np.ndarray): Motor ID per window — ignored.

        Returns:
            Self.
        """
        t_stop: np.ndarray | None = kwargs.get('t_stop', None)   # type: ignore
        evento: np.ndarray | None = kwargs.get('evento', None)   # type: ignore

        try:
            from lifelines import CoxPHFitter
        except ImportError:
            raise ImportError(
                "lifelines is required. Install with: pip install lifelines"
            )

        if t_stop is None:
            warnings.warn(
                "t_stop is required for CoxPHModel. "
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

        # Remove rows with non-positive duration (Cox requires duration > 0)
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
                self.model_ = CoxPHFitter(
                    baseline_estimation_method=self.baseline_estimation_method,
                    n_baseline_knots=self.n_baseline_knots,
                    penalizer=self.penalizer,
                    l1_ratio=self.l1_ratio,
                )
                self.model_.fit(
                    df_surv,
                    duration_col='duration',
                    event_col='event',
                )
            self.is_fitted_ = True

        except Exception as e:
            warnings.warn(
                f"CoxPHModel fit failed: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.is_fitted_ = False

        return self

    def predict_survival_function(
        self,
        X: np.ndarray,
    ) -> pd.DataFrame | None:
        """Returns the survival function S(t|X) for each window.

        S(t|X) = S₀(t)^exp(β'X) — the probability that the motor
        survives beyond time t given features X.

        Args:
            X: Feature matrix of shape (n_windows, n_components).

        Returns:
            DataFrame of shape (n_timepoints, n_windows) where:
                - index: time points t
                - columns: window indices 0..n_windows-1
                - values: S(t|X) ∈ [0, 1]
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
        """Returns the death curve F(t|X) = 1 - S(t|X) for each window.

        F(t|X) is the probability that the motor has failed by time t.
        Used to find t* such that F(t*) = confidence_threshold.

        Args:
            X: Feature matrix of shape (n_windows, n_components).

        Returns:
            DataFrame of shape (n_timepoints, n_windows) where:
                - index: time points t
                - columns: window indices 0..n_windows-1
                - values: F(t|X) = 1 - S(t|X) ∈ [0, 1]
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
            1. Compute death curve F(t|X) = 1 - S(t|X)
            2. Find t* = first t where F(t|X) >= confidence_threshold
            3. RUL = max(t* - t_stop_current, 0), clipped to clipping_threshold

        If F(t) never reaches confidence_threshold (degenerate case —
        S(t)≈1 always, consistent with the documented negative result for
        WeibullAFTModel on C-MAPSS FD001), RUL remains NaN for that window.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            t_stop: Current absolute cycle of each window, shape (n_windows,).

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

        time_points = death_curve.index.to_numpy(dtype=float)
        n_windows = X.shape[0]
        rul_pred = np.full(n_windows, np.nan)

        for w in range(n_windows):
            ft = death_curve.iloc[:, w].to_numpy()
            idx = np.searchsorted(ft, self.confidence_threshold)
            if idx < len(time_points):
                t_star = time_points[idx]
                rul = t_star - float(t_stop[w])
                rul_pred[w] = float(np.clip(max(rul, 0.0),
                                            0.0, self.clipping_threshold))

        return rul_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fallback predict — not usable without t_stop.

        CoxPHModel requires t_stop to compute RUL = t* - t_stop_current.
        Use predict_with_time(X, t_stop) instead.

        Returns:
            NaN array of shape (n_windows,) with a RuntimeWarning.
        """
        warnings.warn(
            "CoxPHModel.predict() cannot compute RUL without t_stop. "
            "Use predict_with_time(X, t_stop) instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.full(len(X), np.nan)

    def print_summary(self) -> None:
        """Prints the lifelines model summary if fitted."""
        if self.is_fitted_ and self.model_ is not None:
            try:
                self.model_.print_summary()
            except Exception as e:
                print(f"print_summary failed: {e}")
        else:
            print("Model is not fitted.")