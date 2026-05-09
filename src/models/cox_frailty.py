#./src/models/cox_frailty.py

"""Cox proportional hazards model with shared frailty for RUL estimation.

This module implements a shared frailty Cox model as a BaseRULModel subclass,
using survival::coxph with frailty() via the r_repository layer. The shared
frailty term captures unobserved motor-level heterogeneity, making this model
methodologically correct for longitudinal sensor data where multiple rows
per motor violate the independence assumption of standard Cox models.

Uses the Andersen-Gill counting process format with Surv(t_start, t_stop,
evento) where t_start and t_stop are derived from time_in_cycles. The survival
function S(t|X) gives P(T > t) where t is absolute cycle count.

R implementation note:
    frailtypack::frailtyPenal was the original implementation but proved
    numerically unstable on C-MAPSS cross-validation folds (istop=2 failures
    with insufficient events per fold). survival::coxph with frailty() is
    statistically equivalent and converges reliably across all fold sizes.

Production requirement:
    time_in_cycles must be present as a feature column in X at both fit()
    and predict() time. It is used to extract t_current for the conversion
    RUL = t_failure - t_current, and is excluded from the R covariate formula.

Scaling requirement:
    The pipeline's MADScaler must be used instead of RobustScaler. MAD-based
    scaling matches R's internal scaling convention and guarantees convergence,
    while IQR-based scaling causes convergence failures.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array

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
    observations of motor i.

    Predictions are generated via S(t|X) = S0(t)^exp(β'X) using the cumulative
    baseline hazard from basehaz(fit, centered=FALSE), marginalizing over the
    frailty distribution for production inference without group membership.

    The time axis is absolute cycle count. time_in_cycles must be present as a
    feature column in X at both fit() and predict() time so that
    RUL = t_failure - t_current can be computed.

    Args:
        distribution: Distribution of the frailty term. Either 'gamma' or
            'gaussian'. Defaults to 'gamma'.
        maxit: Maximum number of outer iterations for coxph. Defaults to 300.
        confidence_threshold: Cumulative failure probability F(t) = 1 - S(t)
            at which the predicted failure cycle is declared. Defaults to 0.5.
        clipping_threshold: Maximum RUL value applied to predictions.
            Defaults to 125.
    """

    def __init__(
        self,
        distribution: str = 'gamma',
        maxit: int = 300,
        confidence_threshold: float = 0.5,
        clipping_threshold: int = 125,
    ) -> None:
        self.distribution = distribution
        self.maxit = maxit
        self.confidence_threshold = confidence_threshold
        self.clipping_threshold = clipping_threshold
        self.is_fitted_: bool = False
        self._model_name: str | None = None
        self._feature_names: list[str] | None = None
        self._column_names: list[str] | None = None

    def prepare_training_data(
        self,
        list_ids: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Prepares training data in the Andersen-Gill counting process format.

        Loads per-motor CSV files and assembles a feature matrix, a structured
        survival target array, a scalar RUL array, and a groups array for
        GroupKFold cross-validation.

        The survival target uses the counting process format with t_start and
        t_stop derived from time_in_cycles:
            t_start = time_in_cycles - 1  (start of interval)
            t_stop  = time_in_cycles      (end of interval)

        This guarantees t_start < t_stop for all rows and uses absolute cycle
        count as the time axis, compatible with recurrentAG=TRUE in frailtyPenal.

        For train motors (evento=1): evento=True at the last cycle only.
        For test motors (evento=0): evento=False for all cycles.

        Also stores the column names from X in self._column_names so that
        fit() can reconstruct the named DataFrame when X arrives as ndarray
        from the sklearn pipeline.

        Args:
            list_ids: Array of motor unit identifiers whose CSV files will be
                loaded from data/clean/.

        Returns:
            Tuple of (X, y_fit, y_metrics, groups) where:
                X: Feature matrix of shape (n_samples, n_features), including
                    time_in_cycles as the first column.
                y_fit: Structured numpy array of shape (n_samples,) with dtype
                    [('evento', bool), ('t_start', float), ('t_stop', float)].
                y_metrics: Scalar RUL array of shape (n_samples,) for metrics.
                groups: Integer array of shape (n_samples,) mapping each row
                    to its motor unit identifier.
        """
        X_list: list[pd.DataFrame] = []
        y_survival_list: list[tuple[bool, float, float]] = []
        y_count_list: list[float] = []
        groups_list: list[int] = []

        for idx in list_ids:
            df_motor = pd.read_csv(f'data/clean/data_motor_{idx}.csv')

            # es_evento=1 if this motor has an observed failure (any cycle has evento=1).
            # Using max() instead of iloc[0] because evento=1 only at the last cycle
            # after the dataset correction — iloc[0] would always return 0.
            es_evento: bool = bool(df_motor['evento'].max())
            X_motor = df_motor.drop(columns=['RUL', 'evento'])
            X_list.append(X_motor)

            n_rows = len(df_motor)
            times = df_motor['time_in_cycles'].tolist()
            ruls = df_motor['RUL'].tolist()

            for i, (t, rul_val) in enumerate(zip(times, ruls)):
                t_start = float(t) - 1.0
                t_stop = float(t)
                # evento=1 only at the last cycle of train motors
                is_last = (i == n_rows - 1)
                evento_row = bool(es_evento and is_last)
                y_survival_list.append((evento_row, t_start, t_stop))
                y_count_list.append(float(rul_val))

            groups_list.extend([idx] * n_rows)

        X = pd.concat(X_list, ignore_index=True)
        self._column_names = list(X.columns)

        y_surv = np.array(
            y_survival_list,
            dtype=[('evento', bool), ('t_start', float), ('t_stop', float)]
        )
        y_count = np.array(y_count_list)
        groups = np.array(groups_list)

        return X, y_surv, y_count, groups

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: object
    ) -> 'CoxFrailty':
        """Fits the shared frailty Cox model via R's frailtypack.

        Extracts the event indicator, t_start and t_stop from the structured
        y array, then calls fit_cox_frailty from the r_repository layer.
        The fitted model is stored in R's global environment and referenced
        by name. On failure, emits a RuntimeWarning and marks the model as
        not fitted.

        When X arrives as ndarray from the sklearn pipeline, column names are
        restored using self._column_names, which is populated by
        prepare_training_data(). time_in_cycles is excluded from the R
        covariate formula since it is implicitly encoded in t_start and t_stop.

        Args:
            X: Feature matrix of shape (n_samples, n_features). When passed
                through the sklearn pipeline, arrives as ndarray; column names
                are restored from self._column_names.
            y: Structured survival array of shape (n_samples,) with dtype
                [('evento', bool), ('t_start', float), ('t_stop', float)].
            **kwargs: Accepts 'groups' as an integer array of shape (n_samples,)
                mapping each row to its motor unit identifier. When provided,
                used as the frailty clustering variable. When absent, each row
                is treated as an independent motor.

        Returns:
            Self, following the scikit-learn estimator convention.
        """
        groups_raw = kwargs.get('groups', None)
        groups: np.ndarray | None = (
            groups_raw if isinstance(groups_raw, np.ndarray) else None
        )
        try:
            # Restore column names when X arrives as ndarray from pipeline
            col_names: list[str] = (
                self._column_names
                if self._column_names is not None
                else [f'col_{i}' for i in range(
                    X.shape[1] if not isinstance(X, pd.DataFrame)
                    else len(X.columns)
                )]
            )
            X_df: pd.DataFrame = (
                X if isinstance(X, pd.DataFrame)
                else pd.DataFrame(X, columns=col_names)
            )
            self._feature_names = list(X_df.columns)

            evento = y['evento'].astype(int)
            t_start = y['t_start'].astype(float)
            t_stop = y['t_stop'].astype(float)
            motor_id = groups if groups is not None else np.arange(len(X_df))

            # Exclude time_in_cycles from R covariates — it is encoded in
            # t_start and t_stop and would introduce multicollinearity
            X_sensors = X_df.drop(columns=['time_in_cycles'], errors='ignore')

            # Remove previous model from R globalenv if it exists
            if self._model_name is not None:
                remove_model(self._model_name)
                self._model_name = None

            self._model_name = fit_cox_frailty(
                X=X_sensors,
                t_start=t_start,
                t_stop=t_stop,
                evento=evento,
                motor_id=motor_id,
                distribution=self.distribution,
                maxit=self.maxit,
            )
            self.is_fitted_ = True

        except Exception as e:
            self.is_fitted_ = False
            self._model_name = None
            warnings.warn(
                f"Fit failed for distribution='{self.distribution}', "
                f"maxit={self.maxit}: "
                f"{type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2
            )

        return self

    def _survival_to_rul(
        self,
        survival_functions: list[tuple[np.ndarray, np.ndarray]],
        t_current: np.ndarray,
    ) -> np.ndarray:
        """Converts individual survival functions to RUL estimates.

        For each survival function S(t|x), computes F(t|x) = 1 - S(t|x)
        and finds the first time t where F(t|x) >= confidence_threshold.
        RUL is then computed as t_failure - t_current for each sample.

        Since the time axis is absolute cycle count, t_current is required
        to convert predicted failure time to remaining useful life.

        If the threshold is never reached, the maximum observed time is
        used as a fallback.

        Args:
            survival_functions: List of (times, probs) tuples as returned
                by predict_survival_functions.
            t_current: Array of shape (n_samples,) with the current cycle
                of each motor, extracted from the time_in_cycles feature.

        Returns:
            RUL array of shape (n_samples,), clipped to clipping_threshold.
        """
        rul_list: list[float] = []

        for i, (times, probs) in enumerate(survival_functions):
            failure_probs = 1.0 - probs
            exceeds = np.where(failure_probs >= self.confidence_threshold)[0]

            if len(exceeds) > 0:
                t_failure = float(times[exceeds[0]])
            else:
                t_failure = float(times[-1])

            rul = max(t_failure - float(t_current[i]), 0.0)
            rul_list.append(rul)

        raw_rul = np.array(rul_list)
        return np.minimum(raw_rul, self.clipping_threshold)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates clipped RUL predictions from the survival function.

        Extracts time_in_cycles from X to compute RUL = t_failure - t_current.
        The time_in_cycles column must be present in X as it is included in
        the feature matrix during prepare_training_data.

        Returns NaN predictions if the model was not successfully fitted,
        allowing the GGS loop to handle failed configurations gracefully.

        Args:
            X: Feature matrix of shape (n_samples, n_features). Must include
                time_in_cycles as a column. Must be pre-scaled by the pipeline's
                MADScaler.

        Returns:
            Predicted RUL array of shape (n_samples,), clipped to
            clipping_threshold. Contains NaN values if the model is not fitted.
        """
        if not self.is_fitted_ or self._model_name is None:
            return np.full(X.shape[0], np.nan)

        X_checked = check_array(X)
        col_names: list[str] = (
            self._feature_names
            if self._feature_names is not None
            else (
                self._column_names
                if self._column_names is not None
                else [f'col_{i}' for i in range(X_checked.shape[1])]
            )
        )
        X_df = pd.DataFrame(X_checked, columns=col_names)

        # Extract t_current from time_in_cycles before passing sensors to R
        t_current = X_df['time_in_cycles'].to_numpy()
        X_sensors = X_df.drop(columns=['time_in_cycles'])

        survival_functions = predict_survival_functions(
            model_name=self._model_name,
            X_new=X_sensors
        )

        return self._survival_to_rul(survival_functions, t_current)