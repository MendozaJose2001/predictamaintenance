#./src/models/base_model.py

# ./src/models/base_model.py

from abc import ABC, abstractmethod

import numpy as np


class BaseRULModel(ABC):
    """Abstract base class defining the contract for all RUL estimation models.

    Every model in the PredictaMaintenance pipeline must inherit from this class
    and implement all abstract methods. This ensures that the GGSTrainingManager
    and TestManager can interact with any model in a uniform way, regardless of
    the underlying statistical family (regression, survival, ensemble, etc.).

    Subclasses are responsible for defining how their training data is prepared,
    including the format and structure of the target variable, which may differ
    between model families (e.g. scalar RUL for regression models vs. structured
    survival arrays for Cox and Random Survival Forest models).

    predict_with_time contract:
        All models expose predict_with_time(X, t_stop) as a uniform prediction
        interface that accepts the current cycle t_stop alongside the feature
        matrix X. This enables the GGSTrainingManager and RULProductionPredictor
        to call a single method regardless of model family, without branching on
        model type.

        Regression models (NB, SVR, DT, RF, XGB) inherit the default
        implementation, which delegates to predict(X) and ignores t_stop —
        regression models predict RUL directly from X without needing the
        current cycle. This preserves full backward compatibility with existing
        serialized .pkl models.

        Survival models (CoxPHModel, WeibullAFTModel, SurvivalTreeModel)
        override predict_with_time() to compute RUL = t* - t_stop, where t*
        is the predicted failure cycle from the death curve F(t|X).
    """

    @abstractmethod
    def prepare_training_data(
        self,
        list_ids: np.ndarray
    ) -> tuple:
        """Prepares the training data required by this model family.

        Every model must return a 4-element tuple that separates the target
        used for fitting from the target used for metric evaluation. This
        decoupling allows survival models (which require a structured array
        for fit) and regression models (which use scalar RUL) to share the
        same training loop without imposing a single-y constraint.

        Args:
            list_ids: Array of motor unit identifiers whose CSV files will be
                loaded from data/clean/.

        Returns:
            Tuple of (X, y_fit, y_metrics, groups) where:
                X: Feature matrix of shape (n_samples, n_features).
                y_fit: Target array passed to pipeline.fit(). Format is
                    model-specific — scalar RUL for regression models,
                    structured survival array for Cox-family models.
                y_metrics: Scalar RUL array of shape (n_samples,) passed to
                    scoring functions. Always y_count regardless of model
                    family, enabling consistent metric computation across
                    all models.
                groups: Integer array of shape (n_samples,) mapping each row
                    to its motor unit identifier, used for GroupKFold.
        """
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> 'BaseRULModel':
        """Fits the model on the provided training data.

        Must follow the scikit-learn estimator convention by returning self,
        enabling use within sklearn Pipeline objects.

        Survival models read t_stop and evento from **kwargs to build their
        survival target. Regression models accept but silently ignore kwargs.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array in the format required by this model family.
            **kwargs: Optional keyword arguments. Survival models read
                t_stop (np.ndarray) and evento (np.ndarray). Regression
                models ignore all kwargs.

        Returns:
            Self, following the scikit-learn estimator convention.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates RUL predictions for the given input features.

        Predictions must be expressed as estimated remaining useful life in
        cycles, clipped to the model's internal clipping_threshold. This
        ensures consistency with the piecewise training space and with the
        metric computation in Metrics._comun_values.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted RUL array of shape (n_samples,), clipped to
            clipping_threshold.
        """
        ...

    def predict_with_time(
        self,
        X: np.ndarray,
        t_stop: np.ndarray,
    ) -> np.ndarray:
        """Generates RUL predictions using the feature matrix and current cycle.

        Default implementation delegates to predict(X) and ignores t_stop —
        correct for regression models that predict RUL directly from features
        without needing the current cycle.

        Survival models (CoxPHModel, WeibullAFTModel, SurvivalTreeModel)
        override this method to compute RUL = t* - t_stop, where t* is
        the predicted failure cycle obtained from the death curve F(t|X).

        This method is the single prediction entry point used by both
        GGSTrainingManager._evaluate_fold() and
        RULProductionPredictor.predict(). Calling predict_with_time()
        uniformly across all model families eliminates type-based branching
        and preserves full backward compatibility with serialized .pkl models
        — existing regression model .pkl files inherit this default without
        requiring deserialization or re-training.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            t_stop: Current absolute cycle of each sample, shape (n_samples,).
                Used by survival models to compute RUL = t* - t_stop.
                Ignored by regression models.

        Returns:
            Predicted RUL array of shape (n_samples,), clipped to
            clipping_threshold.
        """
        return self.predict(X)