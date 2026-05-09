#./src/models/base_model.py

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

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array in the format required by this model family.

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