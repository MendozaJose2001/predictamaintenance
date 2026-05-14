# ./src/models/cox_frailty.py

"""Cox proportional hazards model with shared frailty for RUL estimation.

This module implements CoxFrailty — a shared frailty Cox model as a
BaseRULModel subclass, using survival::coxph with frailty() via the
r_repository layer. The shared frailty term captures unobserved motor-level
heterogeneity, making this model methodologically correct for longitudinal
sensor data where multiple rows per motor violate the independence assumption
of standard Cox models.

Motivation:
    Standard Cox PH (CoxPHModel) treats all windows as independent observations,
    violating the independence assumption for multiple windows from the same motor.
    The shared frailty term ω_i addresses this by capturing motor-level
    heterogeneity:

        h(t|X, ω_i) = h_0(t) · ω_i · exp(β'X)

    where ω_i follows a gamma or gaussian distribution shared across all windows
    of motor i. This is statistically more correct than standard Cox PH for the
    sliding window pipeline.

Duration variable:
    Uses t_stop as duration_col — the absolute cycle at which each window ends.
    t_start = t_stop - 1 is constructed internally, consistent with the
    Andersen-Gill counting process format Surv(t_start, t_stop, evento).
    This is consistent with CoxPHModel and WeibullAFTModel.

RUL prediction via death curve:
    Identical procedure to CoxPHModel and WeibullAFTModel:
        1. Fit CoxFrailty with t_stop (duration) and evento (event indicator)
        2. Compute S(t|X) = S0(t)^exp(β'X) marginalizing over frailty
        3. Derive death curve F(t|X) = 1 - S(t|X)
        4. Find t* = first t where F(t*) >= confidence_threshold
        5. RUL = max(t* - t_stop_current, 0), clipped to clipping_threshold

    confidence_threshold is treated as a GGS hyperparameter.

R implementation note:
    frailtypack::frailtyPenal was the original implementation but proved
    numerically unstable on C-MAPSS cross-validation folds (istop=2 failures
    with insufficient events per fold). survival::coxph with frailty() is
    statistically equivalent and converges reliably across all fold sizes.

groups kwarg:
    Unlike CoxPHModel and WeibullAFTModel, CoxFrailty requires groups — the
    motor ID per window — to define the frailty clustering variable. groups
    is read from **kwargs in fit(), consistent with how the GGS passes it.
    When absent, each window is treated as an independent motor (frailty
    degenerates to standard Cox PH).

Known limitations:
    The 0.4% event rate produced by the sliding window pipeline on C-MAPSS
    FD001 may cause convergence failures or degenerate predictions (S(t)≈1),
    consistent with the documented negative results for CoxPHModel and
    WeibullAFTModel. The frailty term adds one additional parameter (θ) that
    requires sufficient events per group to identify — with few events per
    motor fold, identification may fail silently.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.models.base_model import BaseRULModel
from src.repository.frailty_cox import (
    fit_cox_frailty,
    predict_survival_functions,
    remove_model,
)


class CoxFrailty(BaseRULModel, BaseEstimator, RegressorMixin):
    """Cox proportional hazards model with shared gamma frailty for RUL estimation.

    Wraps survival::coxph with frailty() via the r_repository layer.
    The shared frailty term ω_i captures unobserved motor-level heterogeneity:

        h(t|X, ω_i) = h_0(t) · ω_i · exp(β'X)

    where ω_i follows a gamma or gaussian distribution shared across all
    windows of motor i. Predictions are generated via S(t|X) = S0(t)^exp(β'X)
    marginalizing over the frailty distribution for inference without requiring
    group membership at prediction time.

    RUL is estimated via the death curve F(t|X) = 1 - S(t|X): find t* where
    F(t*) = confidence_threshold, then RUL = t* - t_stop_current.

    Args:
        distribution: Distribution of the frailty term. One of 'gamma',
            'gaussian', or 't'. Defaults to 'gamma'.
        maxit: Maximum number of outer iterations for coxph. Fixed at 50
            based on Therneau's coxme default (iter.max=20) and empirical
            evidence that the inner EM loop for θ fails structurally with
            ~0.4% event rate regardless of iter.max — additional iterations
            beyond 50 yield no statistical benefit on C-MAPSS FD001.
            Not a GGS hyperparameter — controls the optimizer, not the model.
            Defaults to 50.
        method: Estimation method for frailty variance θ. One of 'em' or
            'aic'. 'em' uses the EM algorithm (default for gamma and t).
            'aic' minimizes AIC to select θ. When method='em' is passed
            with distribution='gaussian', frailty_cox maps it to 'reml'
            internally — frailty.gaussian does not accept 'em'.
            GGS hyperparameter. Defaults to 'em'.
        tdf: Degrees of freedom for the t frailty distribution. Only active
            when distribution='t'. Silently ignored by R for gamma and
            gaussian — consistent with the project convention (e.g. degree
            in SVR, n_baseline_knots in Cox PH).
            GGS hyperparameter. Defaults to 5.
        confidence_threshold: Cumulative failure probability F(t) = 1 - S(t)
            at which the predicted failure cycle is declared. Lower values
            trigger earlier alerts (conservative maintenance). Higher values
            trigger later alerts (aggressive maintenance).
            GGS hyperparameter. Defaults to 0.5.
        clipping_threshold: Maximum RUL value applied to predictions.
            GGS hyperparameter. Defaults to 125.

    Attributes:
        model_name_: Name of the fitted coxph model in R's global environment.
            Available after successful fit().
        is_fitted_: True after fit() completes successfully.
    """

    def __init__(
        self,
        distribution: str = 'gamma',
        maxit: int = 50,
        method: str = 'em',
        tdf: int = 5,
        confidence_threshold: float = 0.5,
        clipping_threshold: int = 125,
    ) -> None:
        self.distribution = distribution
        self.maxit = maxit
        self.method = method
        self.tdf = tdf
        self.confidence_threshold = confidence_threshold
        self.clipping_threshold = clipping_threshold
        self.is_fitted_: bool = False
        self.model_name_: str | None = None

    def prepare_training_data(self, list_ids: np.ndarray) -> tuple:
        """Not implemented — this model uses the sliding window pipeline.

        CoxFrailty is designed for the Nodo 2-4 pipeline. Call fit() directly
        with the pipeline output, passing t_stop, evento and groups via kwargs.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "CoxFrailty uses the sliding window pipeline directly. "
            "Call fit(X, y, t_stop=t_stop, evento=evento, groups=groups) "
            "instead."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object,
    ) -> 'CoxFrailty':
        """Fits the shared frailty Cox model via R's survival package.

        Reads t_stop, evento and groups from **kwargs — consistent with
        CoxPHModel and WeibullAFTModel. t_start is constructed internally
        as t_stop - 1 (Andersen-Gill counting process format).

        groups defines the frailty clustering variable — the motor ID per
        window. When absent, each window is treated as an independent motor,
        which degenerates the frailty term to standard Cox PH.

        On fitting failure, emits RuntimeWarning and sets is_fitted_=False,
        allowing the GGS loop to handle failed configurations gracefully.

        Args:
            X: Feature matrix of shape (n_windows, n_components). PCA-reduced
                output from the sliding window pipeline (Nodo 4).
            y: y_rul — accepted for API compatibility, not used in the fit.
            **kwargs:
                t_stop (np.ndarray): Absolute cycle of each window. Required.
                evento (np.ndarray): Event indicator. 1=failure, 0=censored.
                groups (np.ndarray): Motor ID per window. Used as the frailty
                    clustering variable. If absent, each window is independent.

        Returns:
            Self.
        """
        t_stop: np.ndarray | None  = kwargs.get('t_stop',  None)  # type: ignore
        evento: np.ndarray | None  = kwargs.get('evento',  None)  # type: ignore
        groups: np.ndarray | None  = kwargs.get('groups',  None)  # type: ignore

        if t_stop is None:
            warnings.warn(
                "t_stop is required for CoxFrailty. "
                "Fit failed — t_stop=None.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.is_fitted_ = False
            return self

        # Build feature column names from PCA output
        n_features = X.shape[1]
        feature_names = [f'PC_{i+1}' for i in range(n_features)]

        # Build survival DataFrame — Andersen-Gill counting process format
        df_surv = pd.DataFrame(X, columns=feature_names)
        t_stop_arr  = t_stop.astype(float)
        t_start_arr = t_stop_arr - 1.0   # t_start = t_stop - 1 per window

        evento_arr = (
            evento.astype(int)
            if evento is not None
            else np.ones(len(t_stop_arr), dtype=int)
        )
        motor_id_arr = (
            groups.astype(int)
            if groups is not None
            else np.arange(len(t_stop_arr), dtype=int)
        )

        # Remove rows with non-positive duration (R requires t_start < t_stop)
        valid_mask = t_stop_arr > 0
        n_removed = int((~valid_mask).sum())
        if n_removed > 0:
            warnings.warn(
                f"Removed {n_removed} rows with t_stop <= 0.",
                RuntimeWarning,
                stacklevel=2,
            )
            df_surv      = df_surv[valid_mask].reset_index(drop=True)
            t_start_arr  = t_start_arr[valid_mask]
            t_stop_arr   = t_stop_arr[valid_mask]
            evento_arr   = evento_arr[valid_mask]
            motor_id_arr = motor_id_arr[valid_mask]

        if len(df_surv) == 0:
            warnings.warn(
                "No valid rows after filtering t_stop <= 0. Fit failed.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.is_fitted_ = False
            return self

        try:
            # Remove previous R model if it exists — avoid memory leaks
            if self.model_name_ is not None:
                remove_model(self.model_name_)
                self.model_name_ = None

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.model_name_ = fit_cox_frailty(
                    X=df_surv,
                    t_start=t_start_arr,
                    t_stop=t_stop_arr,
                    evento=evento_arr,
                    motor_id=motor_id_arr,
                    distribution=self.distribution,
                    maxit=self.maxit,
                    method=self.method,
                    tdf=self.tdf,
                )
            self.is_fitted_ = True

        except Exception as e:
            warnings.warn(
                f"CoxFrailty fit failed for distribution='{self.distribution}': "
                f"{type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.is_fitted_ = False
            self.model_name_ = None

        return self

    def predict_survival_function(
        self,
        X: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Returns the survival function S(t|X) for each window.

        Computes S(t|X) = S0(t)^exp(β'X) marginalizing over the frailty
        distribution, via the r_repository layer.

        Args:
            X: Feature matrix of shape (n_windows, n_components).

        Returns:
            List of (times, probs) tuples — one per window — where:
                times: numpy array of time points.
                probs: numpy array of S(t|X) ∈ [0, 1].
            None if model is not fitted.
        """
        if not self.is_fitted_ or self.model_name_ is None:
            return None
        try:
            n_features = X.shape[1]
            feature_names = [f'PC_{i+1}' for i in range(n_features)]
            X_df = pd.DataFrame(X, columns=feature_names)
            return predict_survival_functions(
                model_name=self.model_name_,
                X_new=X_df,
            )
        except Exception:
            return None

    def predict_death_curve(
        self,
        X: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Returns the death curve F(t|X) = 1 - S(t|X) for each window.

        F(t|X) is the probability that the motor has failed by time t.
        Used to find t* such that F(t*) = confidence_threshold.

        Args:
            X: Feature matrix of shape (n_windows, n_components).

        Returns:
            List of (times, probs) tuples — one per window — where:
                times: numpy array of time points.
                probs: numpy array of F(t|X) = 1 - S(t|X) ∈ [0, 1].
            None if model is not fitted.

        Note:
            CoxFrailty returns a list of tuples (times, probs) rather than
            a pd.DataFrame — consistent with the r_repository output format
            from predict_survival_functions(). CoxPHModel and WeibullAFTModel
            return DataFrames because lifelines produces them natively.
        """
        sf = self.predict_survival_function(X)
        if sf is None:
            return None
        return [(times, 1.0 - probs) for times, probs in sf]

    def _survival_to_rul(
        self,
        survival_functions: list[tuple[np.ndarray, np.ndarray]],
        t_stop: np.ndarray,
    ) -> np.ndarray:
        """Converts survival functions to clipped RUL predictions.

        For each window, finds the first time t where F(t) >= confidence_threshold
        and computes RUL = t* - t_stop_current. If the threshold is never reached,
        uses the maximum observed time as fallback — consistent with the documented
        negative result behavior of CoxPHModel and WeibullAFTModel.

        Args:
            survival_functions: List of (times, probs) from predict_survival_function.
            t_stop: Current absolute cycle of each window, shape (n_windows,).

        Returns:
            RUL array of shape (n_windows,) clipped to clipping_threshold.
        """
        rul_list: list[float] = []

        for i, (times, probs) in enumerate(survival_functions):
            failure_probs = 1.0 - probs
            exceeds = np.where(failure_probs >= self.confidence_threshold)[0]

            if len(exceeds) > 0:
                t_failure = float(times[exceeds[0]])
            else:
                t_failure = float(times[-1])

            rul = max(t_failure - float(t_stop[i]), 0.0)
            rul_list.append(rul)

        raw_rul = np.array(rul_list)
        return np.minimum(raw_rul, self.clipping_threshold)

    def predict_with_time(
        self,
        X: np.ndarray,
        t_stop: np.ndarray,
    ) -> np.ndarray:
        """Predicts RUL via the death curve at confidence_threshold.

        Procedure for each window:
            1. Compute S(t|X) = S0(t)^exp(β'X) via r_repository
            2. Derive death curve F(t|X) = 1 - S(t|X)
            3. Find t* = first t where F(t*) >= confidence_threshold
            4. RUL = max(t* - t_stop_current, 0), clipped to clipping_threshold

        This is the primary prediction method used by GGSTrainingManager
        and RULProductionPredictor via the BaseRULModel.predict_with_time()
        interface.

        Args:
            X: Feature matrix of shape (n_windows, n_components).
            t_stop: Current absolute cycle of each window, shape (n_windows,).

        Returns:
            Predicted RUL array of shape (n_windows,), clipped to
            clipping_threshold and floored at 0. NaN if not fitted.
        """
        if not self.is_fitted_ or self.model_name_ is None:
            return np.full(len(X), np.nan)

        sf = self.predict_survival_function(X)
        if sf is None:
            return np.full(len(X), np.nan)

        return self._survival_to_rul(sf, t_stop)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fallback predict — not usable without t_stop.

        CoxFrailty requires t_stop to compute RUL = t* - t_stop_current.
        Use predict_with_time(X, t_stop) instead. The BaseRULModel default
        for predict_with_time() will call this method with t_stop ignored —
        override ensures correct behavior when called via the uniform interface.

        Returns:
            NaN array of shape (n_windows,) with a RuntimeWarning.
        """
        warnings.warn(
            "CoxFrailty.predict() cannot compute RUL without t_stop. "
            "Use predict_with_time(X, t_stop) instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.full(len(X), np.nan)

    def print_summary(self) -> None:
        """Prints basic model info — R summary not directly accessible."""
        if self.is_fitted_ and self.model_name_ is not None:
            try:
                import rpy2.robjects as ro
                ro.r(f'print(summary({self.model_name_}))')
            except Exception as e:
                print(f"print_summary failed: {e}")
        else:
            print("Model is not fitted.")