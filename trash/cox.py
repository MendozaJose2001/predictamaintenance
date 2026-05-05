import warnings

from typing import Sequence

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.functions import StepFunction
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array

from src.models.base_model import BaseRULModel


class CoxPiecewise(BaseRULModel, BaseEstimator, RegressorMixin):
    """Cox Proportional Hazards model with elastic net penalty for RUL estimation.

    Wraps scikit-survival's CoxnetSurvivalAnalysis as a scikit-learn compatible
    estimator. The model learns a survival function S(t|x) for each input vector,
    from which RUL is estimated as the cycle at which the cumulative failure
    probability F(t|x) = 1 - S(t|x) exceeds a confidence threshold, minus one.

    The clipping threshold is applied to predictions to ensure consistency with
    the piecewise training space, following the same convention as the Negative
    Binomial model.

    Args:
        alphas: Regularization strength. When None, sksurv generates an
            automatic regularization path. When a float is provided, the model
            trains at exactly that penalization value. Defaults to None.
        l1_ratio: Elastic net mixing parameter. 1.0 corresponds to pure L1
            (Lasso) and 0.0 to pure L2 (Ridge). Defaults to 0.5.
        confidence_threshold: Cumulative failure probability F(t) = 1 - S(t)
            at which the predicted failure cycle is declared. For example, 0.9
            means the model predicts the cycle at which the motor has a 90%
            probability of having failed. Defaults to 0.5 (median survival).
        clipping_threshold: Maximum RUL value applied to predictions, consistent
            with the piecewise training convention. Defaults to 125.
    """

    def __init__(
        self,
        alphas: float | None = None,
        l1_ratio: float = 0.5,
        confidence_threshold: float = 0.5,
        clipping_threshold: int = 125,
    ) -> None:
        self.alphas = alphas
        self.l1_ratio = l1_ratio
        self.confidence_threshold = confidence_threshold
        self.clipping_threshold = clipping_threshold
        self.is_fitted_: bool = False
        self.model_: CoxnetSurvivalAnalysis | None = None

    def prepare_training_data(
        self,
        list_ids: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Prepares training data in the format required by the Cox model.

        Loads per-motor CSV files and assembles a feature matrix, a structured
        survival target array, a scalar RUL array, and a groups array for
        GroupKFold cross-validation. Each row corresponds to a single cycle.

        The survival target encodes:
            - evento=True, tiempo=total_lifetime for train motors (observed failure).
            - evento=False, tiempo=last_observed_cycle for test motors (censored).

        All rows of the same motor share the same (evento, tiempo) pair, since
        the survival time is a property of the motor unit, not of individual cycles.

        Args:
            list_ids: Array of motor unit identifiers whose CSV files will be
                loaded from data/clean/.

        Returns:
            Tuple of (X, y_fit, y_metrics, groups) where:
                X: Feature matrix of shape (n_samples, n_features), excluding
                    time_in_cycles, RUL, and evento columns.
                y_fit: Structured numpy array of shape (n_samples,) with dtype
                    [('evento', bool), ('tiempo', float)], passed to fit().
                y_metrics: Scalar RUL array of shape (n_samples,), passed to
                    scoring functions for consistent metric computation.
                groups: Integer array of shape (n_samples,) mapping each row
                    to its motor unit identifier, used for GroupKFold.
        """
        X_list: list[pd.DataFrame] = []
        y_survival_list: list[tuple[bool, float]] = []
        y_count_list: list[float] = []
        groups_list: list[int] = []

        for idx in list_ids:
            df_motor = pd.read_csv(f'data/clean/data_motor_{idx}.csv')

            es_evento: bool = bool(df_motor['evento'].iloc[0])
            max_tiempo_observado: float = float(df_motor['time_in_cycles'].max())
            tiempo_muerte_real: float = max_tiempo_observado + float(df_motor['RUL'].iloc[-1])

            X_motor = df_motor.drop(columns=['time_in_cycles', 'RUL', 'evento'])
            X_list.append(X_motor)

            n_rows = len(df_motor)
            if es_evento:
                y_survival_list.extend([(True, tiempo_muerte_real)] * n_rows)
            else:
                y_survival_list.extend([(False, max_tiempo_observado)] * n_rows)

            y_count_list.extend(df_motor['RUL'].tolist())
            groups_list.extend([idx] * n_rows)

        X = pd.concat(X_list, ignore_index=True)
        y_surv = np.array(
            y_survival_list,
            dtype=[('evento', bool), ('tiempo', float)]
        )
        y_count = np.array(y_count_list)
        groups = np.array(groups_list)

        return X, y_surv, y_count, groups

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CoxPiecewise':
        """Fits the Cox elastic net model on the structured survival target.

        Instantiates and fits a CoxnetSurvivalAnalysis with fit_baseline_model=True,
        which is required to call predict_survival_function at inference time.
        On fitting failure, emits a RuntimeWarning and marks the model as not
        fitted, allowing GridSearchCV to handle the failure via error_score.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Structured survival array of shape (n_samples,) with dtype
                [('evento', bool), ('tiempo', float)].

        Returns:
            Self, following the scikit-learn estimator convention.
        """
        try:
            # sksurv requires alphas as array-like, not scalar
            alphas = [self.alphas] if isinstance(self.alphas, float) else self.alphas
            self.model_ = CoxnetSurvivalAnalysis(
                alphas=alphas,
                l1_ratio=self.l1_ratio,
                fit_baseline_model=True,
                max_iter=100000,
                tol=1e-7,
                verbose=False
            )
            self.model_.fit(X, y)
            self.is_fitted_ = True

        except Exception as e:
            self.is_fitted_ = False
            self.model_ = None
            warnings.warn(
                f"Fit failed for alphas={self.alphas}, "
                f"l1_ratio={self.l1_ratio}: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2
            )

        return self

    def _survival_to_rul(self, survival_functions: np.ndarray) -> np.ndarray:
        """Converts individual survival functions to RUL estimates.

        For each survival function S(t|x), computes the cumulative failure
        probability F(t|x) = 1 - S(t|x) and finds the first cycle t where
        F(t|x) >= confidence_threshold. That cycle is the predicted failure
        point, and RUL is defined as that cycle minus one.

        If F(t|x) never reaches the threshold within the observed time range,
        the maximum observed time is used as a fallback, yielding RUL = max_t - 1.

        Args:
            survival_functions: Array of StepFunction objects of shape (n_samples,),
                as returned by CoxnetSurvivalAnalysis.predict_survival_function.

        Returns:
            RUL array of shape (n_samples,), clipped to clipping_threshold.
        """
        rul_list: list[float] = []

        for sf in survival_functions:
            times: np.ndarray = sf.x
            survival_probs: np.ndarray = sf(times)
            failure_probs: np.ndarray = 1.0 - survival_probs

            # Find first time where F(t) >= confidence_threshold
            exceeds = np.where(failure_probs >= self.confidence_threshold)[0]

            if len(exceeds) > 0:
                t_failure = float(times[exceeds[0]].item())
            else:
                t_failure = float(times[-1].item())

            rul = max(t_failure - 1.0, 0.0)
            rul_list.append(rul)

        raw_rul = np.array(rul_list)
        return np.minimum(raw_rul, self.clipping_threshold)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates clipped RUL predictions from the survival function.

        Returns NaN predictions if the model was not successfully fitted,
        allowing GridSearchCV to handle failed configurations via error_score.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted RUL array of shape (n_samples,), clipped to
            clipping_threshold. Contains NaN values if the model is not fitted.
        """
        if not self.is_fitted_ or self.model_ is None:
            return np.full(X.shape[0], np.nan)

        X = check_array(X)

        # Retrieve the alpha used during training to ensure predict_survival_function
        # evaluates the correct solution on the regularization path. When alphas is
        # None, sksurv selects the path automatically and stores it in alphas_;
        # otherwise we use the scalar value provided at initialization.
        if self.alphas is None:
            alpha_value: float = float(self.model_.alphas_[0])
        else:
            alpha_value = float(self.alphas)

        survival_functions = self.model_.predict_survival_function(X, alpha=alpha_value)

        return self._survival_to_rul(survival_functions)