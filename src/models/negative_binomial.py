# ./src/models/negative_binomial.py

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from statsmodels.base.elastic_net import RegularizedResultsWrapper

from src.models.base_model import BaseRULModel


class NegativeBinomialPiecewise(BaseRULModel, BaseEstimator, RegressorMixin):
    """Negative Binomial GLM for piecewise RUL estimation.

    Wraps a statsmodels Negative Binomial GLM as a scikit-learn compatible
    estimator. The target is clipped to a piecewise threshold during training,
    reflecting the operational assumption that very early degradation stages
    carry limited prognostic information. Predictions are also clipped to the
    same threshold to ensure consistency with the training space.

    Supports elastic net regularization via statsmodels fit_regularized and
    three link functions: log, identity, and sqrt.

    Args:
        alpha: Dispersion parameter of the Negative Binomial distribution.
            Controls the degree of overdispersion relative to a Poisson model.
            Defaults to 1.0.
        clipping_threshold: Maximum RUL value used for piecewise clipping of
            both the training target and the predictions. Defaults to 125.
        alpha_reg: Regularization strength. When 0.0, no regularization is
            applied and the standard MLE fit is used. Defaults to 0.0.
        l1_ratio: Mixing parameter for elastic net regularization. A value of
            1.0 corresponds to pure L1 (Lasso) and 0.0 to pure L2 (Ridge).
            Only used when alpha_reg > 0. Defaults to 0.5.
        link_type: Link function for the GLM. One of 'log', 'identity', or
            'sqrt'. The log link guarantees strictly positive predictions and
            is recommended for RUL estimation. Defaults to 'log'.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        clipping_threshold: int = 125,
        alpha_reg: float = 0.0,
        l1_ratio: float = 0.5,
        link_type: str = 'log'
    ) -> None:
        self.alpha = alpha
        self.clipping_threshold = clipping_threshold
        self.alpha_reg = alpha_reg
        self.l1_ratio = l1_ratio
        self.link_type = link_type
        self.is_fitted_: bool = False
        self.model_stats_: GLMResultsWrapper | RegularizedResultsWrapper | None = None

    def _get_link(self) -> sm.families.links.Link:
        """Returns the statsmodels link function corresponding to link_type.

        Falls back to the log link if an unrecognized link_type is provided.

        Returns:
            A statsmodels link function instance.
        """
        links: dict[str, sm.families.links.Link] = {
            'log': sm.families.links.Log(),
            'identity': sm.families.links.Identity(),
            'sqrt': sm.families.links.Sqrt()
        }
        return links.get(self.link_type, sm.families.links.Log())

    def prepare_training_data(
        self,
        list_ids: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Prepares training data in the format required by the Negative Binomial model.

        Loads per-motor CSV files and assembles a feature matrix, a scalar RUL
        target array, and a groups array for GroupKFold cross-validation. Each
        row in the output corresponds to a single observation cycle of a motor.

        For this model family, y_fit and y_metrics are identical — both are
        scalar RUL arrays — since the Negative Binomial model trains directly
        on RUL targets and the evaluation metrics operate on the same space.

        Args:
            list_ids: Array of motor unit identifiers whose CSV files will be
                loaded from data/clean/.

        Returns:
            Tuple of (X, y_fit, y_metrics, groups) where:
                X: Feature matrix of shape (n_samples, n_features), excluding
                    time_in_cycles, RUL, and evento columns.
                y_fit: Scalar RUL array of shape (n_samples,).
                y_metrics: Identical to y_fit for this model family.
                groups: Integer array of shape (n_samples,) mapping each row
                    to its motor unit identifier, used for GroupKFold.
        """
        X_list: list[pd.DataFrame] = []
        y_count_list: list[float] = []
        groups_list: list[int] = []

        for idx in list_ids:
            df_motor = pd.read_csv(f'data/clean/data_motor_{idx}.csv')

            X_motor = df_motor.drop(columns=['time_in_cycles', 'RUL', 'evento'])
            X_list.append(X_motor)

            y_count_list.extend(df_motor['RUL'].tolist())
            groups_list.extend([idx] * len(df_motor))

        X = pd.concat(X_list, ignore_index=True)
        y_count = np.array(y_count_list)
        groups = np.array(groups_list)

        return X, y_count, y_count, groups

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> 'NegativeBinomialPiecewise':
        """Fits the Negative Binomial GLM on piecewise-clipped RUL targets.

        Applies piecewise clipping to the target before fitting, capping all
        RUL values at clipping_threshold. When alpha_reg > 0, fits using
        elastic net regularization via statsmodels fit_regularized. On fitting
        failure, emits a RuntimeWarning with the error details and marks the
        estimator as not fitted, allowing GridSearchCV to handle the failure
        gracefully via error_score.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: RUL target array of shape (n_samples,). Values are not
                pre-clipped; clipping is applied internally.

        Returns:
            Self, following the scikit-learn estimator convention.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        y_piecewise = np.minimum(y, self.clipping_threshold)
        X_with_const = sm.add_constant(X, has_constant='add')

        try:
            family_nb = sm.families.NegativeBinomial(
                alpha=self.alpha,
                link=self._get_link()
            )
            model = sm.GLM(y_piecewise, X_with_const, family=family_nb)

            if self.alpha_reg > 0:
                self.model_stats_ = model.fit_regularized(
                    method='elastic_net',
                    alpha=self.alpha_reg,
                    L1_wt=self.l1_ratio,
                    maxiter=500
                )
            else:
                self.model_stats_ = model.fit()

            self.is_fitted_ = True

        except Exception as e:
            self.is_fitted_ = False
            self.model_stats_ = None
            warnings.warn(
                f"Fit failed for link='{self.link_type}', "
                f"alpha_reg={self.alpha_reg}: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates clipped RUL predictions for the given input features.

        Returns NaN predictions if the model was not successfully fitted,
        allowing GridSearchCV to handle failed configurations via error_score.
        Predictions are clipped to clipping_threshold to ensure consistency
        with the piecewise training space.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted RUL array of shape (n_samples,), clipped to
            clipping_threshold. Contains NaN values if the model is not fitted.
        """
        if not self.is_fitted_ or self.model_stats_ is None:
            return np.full(X.shape[0], np.nan)

        X = check_array(X)
        X_with_const = sm.add_constant(X, has_constant='add')
        raw_predictions = self.model_stats_.predict(X_with_const)

        return np.minimum(raw_predictions, self.clipping_threshold)