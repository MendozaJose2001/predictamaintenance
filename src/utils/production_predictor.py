"""Frozen Estimator for production RUL inference.

This module implements RULProductionPredictor — an immutable object that
encapsulates a trained RULPipeline and a trained BaseRULModel, exposing
only the inference interface. The object does not support retraining by
design: it is a black-box that receives a sensor DataFrame and returns
RUL predictions.

Design decisions:
    Frozen Estimator pattern:
        The object receives already-trained pipeline and model in __init__
        and stores them as immutable state. fit() is not implemented — any
        retraining attempt raises NotImplementedError. This reflects the
        operational reality: in production the model is a fixed artifact,
        not a learning object.

    BaseEstimator + RegressorMixin inheritance:
        Following the project convention, the object inherits from both
        sklearn classes. BaseEstimator provides get_params() / set_params()
        from __init__ arguments, enabling introspection and sklearn tooling
        integration. RegressorMixin provides score() at no additional cost.

    Automatic hyperparameter extraction from pipeline:
        Configuration hyperparameters (feature_set, window_size, n_components,
        clipping_threshold) are extracted directly from the pipeline received
        in __init__. This eliminates redundancy — the caller does not repeat
        information already inside the pipeline — and guarantees internal
        consistency.

    Dual pipeline/model references:
        pipeline and model are stored both as self.pipeline / self.model
        (for get_params() via BaseEstimator) and as self.pipeline_ / self.model_
        (sklearn post-construction convention for check_is_fitted()). This is
        correct and generates no conflicts since the object is immutable —
        set_params() should never be called in production.

        FUTURE OPTIMIZATION: evaluate whether to expose only pipeline_ and
        model_ (without single underscore), sacrificing get_params() for
        pipeline/model and eliminating the redundancy. Requires verifying
        that no downstream component depends on get_params() for these
        attributes.

    from_config — factory method for full construction:
        The classmethod from_config receives the model class and a flat
        hyperparameter dict (pipeline + model mixed) along with training
        DataFrames. Internally it separates params via _PIPELINE_PARAMS,
        builds and trains the pipeline and model, and returns a
        RULProductionPredictor ready for production. It is the recommended
        entry point for building predictors — avoids the caller having to
        manage pipeline and model construction and training separately.

    predict() receives raw DataFrame:
        Unlike BaseRULModel.predict() which receives PCA-reduced arrays,
        this predict() receives the raw sensor DataFrame — exactly what
        arrives in production. The internal pipeline.transform() handles
        the full Nodo2→Nodo3→Nodo4 transformation.

    Uniform prediction interface via predict_with_time():
        All models are called via model.predict_with_time(X, t_stop) —
        the uniform interface defined in BaseRULModel. Regression models
        (NB, SVR, DT, RF, XGB) inherit the BaseRULModel default which
        delegates to predict(X) and ignores t_stop. Survival models
        (CoxPHModel, WeibullAFTModel, SurvivalTreeModel) override
        predict_with_time() to compute RUL = t* - t_stop using the death
        curve F(t|X). t_stop is extracted directly from pipeline.transform()
        — no additional computation required.

        This design preserves the interface of regression models already
        serialized in .pkl files — they inherit predict_with_time() from
        BaseRULModel without requiring re-training or re-serialization.

    Output modes in predict():
        predict() supports two modes via return_mode:
        - 'last' (default): returns only the prediction for the last window
          — the RUL estimate at the current cycle. Shape (1,).
        - 'all': returns predictions for all generated windows. Useful for
          visualizing the motor degradation trajectory. Shape (n_windows,).
        Both modes return np.ndarray for type consistency.

    evento=0 in production:
        In production all data is naturally censored (no motor has failed
        yet). The pipeline handles this silently — if evento is absent from
        X_df it is simply not used. No special action required.

    Insufficient history:
        If X_df contains fewer cycles than window_size, build_windows()
        inside the pipeline raises ValueError. This exception propagates
        without catching — it is the caller's responsibility to validate
        the history before predicting.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from src.models.base_model import BaseRULModel
from src.pipeline.rul_pipeline import RULPipeline


# Pipeline-level keys — same definition as ggs_training_manager
_PIPELINE_PARAMS: frozenset[str] = frozenset({
    'feature_set',
    'window_size',
    'n_components',
    'clipping_threshold',
})


def _split_params(params: dict) -> tuple[dict, dict]:
    """Splits a flat param dict into pipeline and model param dicts.

    Reuses the same logic as ggs_training_manager._split_params — keeps
    the separation contract consistent across the project.

    Args:
        params: Flat dictionary with all hyperparameters mixed.

    Returns:
        Tuple of (pipeline_params, model_params).
    """
    pipeline_params = {k: v for k, v in params.items() if k in _PIPELINE_PARAMS}
    model_params    = {k: v for k, v in params.items() if k not in _PIPELINE_PARAMS}
    return pipeline_params, model_params


class RULProductionPredictor(BaseEstimator, RegressorMixin):
    """Frozen Estimator for production RUL inference.

    Encapsulates a trained RULPipeline and a trained BaseRULModel in an
    immutable sklearn-compatible object. Receives raw sensor DataFrames
    and returns RUL predictions in cycles.

    The object is an inference black-box — retraining is not supported.
    To build a predictor from hyperparameters use the from_config()
    classmethod, which internally manages pipeline and model construction
    and training.

    Args:
        pipeline: Already trained RULPipeline (is_fitted_=True). Configuration
            hyperparameters are extracted automatically from this object.
        model: Already trained BaseRULModel (is_fitted_=True). Any compatible
            subclass: DecisionTreeModel, RandomForestModel, SVRModel,
            NegativeBinomialPiecewise, CoxPHModel, WeibullAFTModel.
        model_name: Human-readable model name for logging and metadata.
            If None, inferred from model.__class__.__name__. Defaults to None.

    Attributes:
        pipeline_: Trained RULPipeline. Available after construction.
        model_: Trained BaseRULModel. Available after construction.
        model_name_: Resolved model name. Available after construction.
        feature_set: Extracted from pipeline.feature_set.
        window_size: Extracted from pipeline.window_size.
        n_components: Extracted from pipeline.n_components.
        clipping_threshold: Extracted from pipeline.clipping_threshold.

    Raises:
        ValueError: If pipeline.is_fitted_ is False.
        ValueError: If model.is_fitted_ is False.

    Example:
        >>> # Recommended construction — via from_config
        >>> predictor = RULProductionPredictor.from_config(
        ...     model_class=NegativeBinomialPiecewise,
        ...     params={
        ...         'feature_set': 'B', 'window_size': 30,
        ...         'n_components': 20, 'clipping_threshold': 115,
        ...         'link_type': 'log', 'alpha': 0.1,
        ...     },
        ...     X_df=X_df,
        ...     y_df=y_df,
        ... )
        >>> # Standard prediction — last window
        >>> rul_now = predictor.predict(df_motor)
        >>> # Full degradation trajectory
        >>> rul_traj = predictor.predict(df_motor, return_mode='all')
    """

    def __init__(
        self,
        pipeline: RULPipeline,
        model: BaseRULModel,
        model_name: str | None = None,
    ) -> None:
        # Validate that pipeline and model are trained before storing
        if not getattr(pipeline, 'is_fitted_', False):
            raise ValueError(
                "The provided pipeline is not trained (is_fitted_=False). "
                "Call pipeline.fit_transform() before building the predictor."
            )
        if not getattr(model, 'is_fitted_', False):
            raise ValueError(
                "The provided model is not trained (is_fitted_=False). "
                "Call model.fit() before building the predictor."
            )

        # __init__ arguments — for get_params() via BaseEstimator
        self.pipeline   = pipeline
        self.model      = model
        self.model_name = model_name

        # Hyperparameters extracted from pipeline — single source of truth
        self.feature_set:        str = pipeline.feature_set
        self.window_size:        int = pipeline.window_size
        self.n_components:       int = pipeline.n_components
        self.clipping_threshold: int = pipeline.clipping_threshold

        # Post-construction attributes — sklearn trailing underscore convention
        self.pipeline_: RULPipeline  = pipeline
        self.model_: BaseRULModel    = model
        self.model_name_: str        = (
            model_name
            if model_name is not None
            else model.__class__.__name__
        )

    @staticmethod
    def _load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the full training partition from DatasetManager.

        Reads the 140 training motors from data/clean/ and assembles
        X_df and y_df ready for pipeline.fit_transform().

        Returns:
            Tuple of (X_df, y_df) where:
                X_df: Feature DataFrame without RUL column.
                y_df: Target DataFrame with unit_number, time_in_cycles, RUL.
        """
        from src.dataset_manager import DatasetManager

        m_train, _ = DatasetManager.split_dataset()

        dfs = []
        for idx in m_train:
            df = pd.read_csv(f'data/clean/data_motor_{idx}.csv')
            df.insert(0, 'unit_number', idx)
            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)
        X_df = df_all.drop(columns=['RUL'])
        y_df = df_all[['unit_number', 'time_in_cycles', 'RUL']].copy()

        return X_df, y_df

    @classmethod
    def from_config(
        cls,
        model_class: type,
        params: dict,
        X_df: pd.DataFrame | None = None,
        y_df: pd.DataFrame | None = None,
        model_name: str | None = None,
    ) -> 'RULProductionPredictor':
        """Builds a RULProductionPredictor from hyperparameters and data.

        Recommended factory method for building production predictors.
        Receives a flat dict with all hyperparameters (pipeline + model
        mixed), separates them internally, builds and trains the pipeline
        and model, and returns a RULProductionPredictor ready for inference.

        Pipeline keys (feature_set, window_size, n_components,
        clipping_threshold) are separated automatically from model keys —
        the caller does not need to make this distinction.

        If X_df and y_df are not provided, training data is loaded
        automatically from DatasetManager (140 C-MAPSS FD001 train motors).

        For survival models (CoxPHModel, WeibullAFTModel, SurvivalTreeModel),
        fit() receives t_stop and evento extracted from the pipeline — required
        to build the survival target. For regression models, these kwargs are
        silently ignored per the BaseRULModel contract.

        Args:
            model_class: Model class to instantiate. Must be a BaseRULModel
                subclass compatible with the sliding window pipeline.
            params: Flat dict with all hyperparameters mixed. Pipeline keys
                (feature_set, window_size, n_components, clipping_threshold)
                are separated automatically. The rest is passed to the model
                constructor.
            X_df: Training DataFrame without RUL column. If None, loaded
                automatically via DatasetManager. Defaults to None.
            y_df: DataFrame with columns unit_number, time_in_cycles, RUL.
                If None, loaded automatically via DatasetManager.
                Defaults to None.
            model_name: Human-readable name for logging and metadata. If None,
                inferred from model_class.__name__. Defaults to None.

        Returns:
            RULProductionPredictor ready for inference.
        """
        if X_df is None or y_df is None:
            X_df, y_df = cls._load_training_data()

        pipeline_params, model_params = _split_params(params)

        # Build and train pipeline on full training data
        pipeline = RULPipeline(**pipeline_params)
        X, y_rul, t_stop, evento, groups = pipeline.fit_transform(X_df, y_df)

        # Build and train model — t_stop and evento passed via kwargs for
        # survival models; silently ignored by regression models
        model = model_class(
            **model_params,
            clipping_threshold=pipeline_params['clipping_threshold'],
        )
        model.fit(X, y_rul, t_stop=t_stop, evento=evento)

        return cls(
            pipeline=pipeline,
            model=model,
            model_name=model_name,
        )

    def save(
        self,
        metrics: dict | None = None,
        base_dir=None,
    ):
        """Serializes the predictor and writes its JSON metadata.

        Convenience wrapper over predictor_io.save_predictor() — allows
        saving the predictor directly from the object without explicitly
        importing the I/O module.

        Args:
            metrics: Dict with GGS cross-validation metrics. Recommended
                keys: mean_S_score, mean_MAE, mean_RMSE, mean_C_index.
                If None, saved as empty dict. Defaults to None.
            base_dir: Root directory for production outputs. If None,
                uses the predictor_io default (outputs/production/).
                Defaults to None.

        Returns:
            PredictorSession with paths and metadata of the saved predictor.
        """
        from pathlib import Path
        from src.utils.predictor_io import save_predictor, _DEFAULT_BASE_DIR

        resolved_metrics  = metrics  if metrics  is not None else {}
        resolved_base_dir = Path(base_dir) if base_dir is not None else _DEFAULT_BASE_DIR

        return save_predictor(
            predictor=self,
            metrics=resolved_metrics,
            base_dir=resolved_base_dir,
        )

    def fit(
        self,
        X: object = None,
        y: object = None,
        **kwargs: object,
    ) -> 'RULProductionPredictor':
        """Not implemented — the production predictor is immutable.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "RULProductionPredictor is immutable — retraining is not supported. "
            "To train a new model use GGSTrainingManager and build a new "
            "RULProductionPredictor with from_config()."
        )

    def predict(
        self,
        X_df: pd.DataFrame,
        return_mode: str = 'last',
    ) -> np.ndarray:
        """Generates RUL predictions from a raw sensor DataFrame.

        Applies the full pipeline (Nodo 2 → Nodo 3 → Nodo 4) on the input
        DataFrame using the trained transformers, then calls
        model.predict_with_time(X, t_stop) to obtain RUL predictions.

        t_stop is extracted directly from pipeline.transform() — it represents
        the last observed cycle of each sliding window, computed from
        time_in_cycles in the input DataFrame without any additional
        computation. Regression models ignore t_stop via the BaseRULModel
        default; survival models use it to compute RUL = t* - t_stop.

        Args:
            X_df: Sensor history DataFrame for the motor to evaluate. Must
                contain at least window_size cycles. Required columns:
                time_in_cycles and the 16 sensor columns used in training.
                unit_number is optional — if absent, id=0 is assigned.
            return_mode: Output mode. One of:
                - 'last' (default): returns only the last window prediction
                  — the RUL estimate at the current cycle. Shape (1,).
                - 'all': returns predictions for all generated windows.
                  Useful for visualizing the degradation trajectory.
                  Shape (n_windows,).

        Returns:
            Predicted RUL array clipped to clipping_threshold.
            Shape (1,) if return_mode='last', (n_windows,) if return_mode='all'.

        Raises:
            ValueError: If X_df has fewer cycles than window_size.
            ValueError: If return_mode is not 'last' or 'all'.
            NotFittedError: If the object was not built correctly
                (pipeline_ or model_ absent).
        """
        check_is_fitted(self, ['pipeline_', 'model_'])

        if return_mode not in ('last', 'all'):
            raise ValueError(
                f"return_mode must be 'last' or 'all', got '{return_mode}'."
            )

        # transform() without y_df — production mode without ground truth.
        # t_stop is extracted by the pipeline from time_in_cycles.
        X, _y_rul, t_stop, _evento, _groups = self.pipeline_.transform(
            X_df,
            y_df=None,
        )

        # predict_with_time() is the uniform interface for all model families.
        # Regression models delegate to predict(X) via BaseRULModel default.
        # Survival models compute RUL = t* - t_stop via their override.
        y_pred = self.model_.predict_with_time(X, t_stop)

        if return_mode == 'last':
            return y_pred[-1:]

        return y_pred