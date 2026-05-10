"""Frozen Estimator sklearn-compatible para inferencia RUL en producción.

Este módulo implementa RULProductionPredictor — un objeto inmutable que
encapsula un RULPipeline entrenado y un BaseRULModel entrenado, exponiendo
exclusivamente la interfaz de inferencia. El objeto no soporta reentrenamiento
por diseño: es una caja negra que recibe un DataFrame de sensores y devuelve
predicciones de RUL.

Design decisions:
    Patrón Frozen Estimator:
        El objeto recibe pipeline y modelo ya entrenados en __init__ y los
        almacena como estado inmutable. No implementa fit() — cualquier
        intento de reentrenamiento lanza NotImplementedError con un mensaje
        claro. Este diseño refleja la realidad operacional: en producción
        el modelo es un artefacto fijo, no un objeto en aprendizaje.

    Herencia BaseEstimator + RegressorMixin:
        Siguiendo el convenio del proyecto, el objeto hereda de ambas clases
        sklearn. BaseEstimator provee get_params() / set_params() gratis a
        partir de los argumentos de __init__, habilitando introspección e
        integración con herramientas sklearn. RegressorMixin provee score()
        como bonus sin costo adicional.

    Extracción automática de hiperparámetros desde pipeline:
        Los hiperparámetros de configuración (feature_set, window_size,
        n_components, clipping_threshold) se extraen directamente del
        pipeline recibido en __init__. Esto elimina redundancia — el caller
        no repite información que ya está dentro del pipeline — y garantiza
        consistencia interna.

    Doble referencia a pipeline y model:
        pipeline y model se almacenan tanto como self.pipeline / self.model
        (para get_params() vía BaseEstimator) como self.pipeline_ / self.model_
        (convenio sklearn post-construcción para check_is_fitted()). Esto es
        correcto y no genera conflictos dado que el objeto es inmutable —
        set_params() nunca debería llamarse en producción.

        OPTIMIZACIÓN FUTURA: evaluar si conviene exponer solo pipeline_ y
        model_ (sin guión simple) sacrificando get_params() para pipeline/model,
        eliminando la redundancia. Requiere verificar si algún componente
        downstream depende de get_params() para estos atributos.

    from_config — factory method para construcción completa:
        El classmethod from_config recibe la clase del modelo y un dict
        plano de hiperparámetros (pipeline + modelo mezclados) junto con
        los DataFrames de entrenamiento. Internamente separa los params
        usando _PIPELINE_PARAMS, construye y entrena el pipeline y el
        modelo, y retorna un RULProductionPredictor listo para producción.
        Es el punto de entrada recomendado para construir predictores — evita
        que el caller tenga que gestionar la construcción y entrenamiento
        de pipeline y modelo por separado.

    predict() recibe DataFrame crudo:
        A diferencia de BaseRULModel.predict() que recibe arrays PCA-reducidos,
        este predict() recibe el DataFrame de sensores en crudo — exactamente
        lo que llega en producción. El pipeline.transform() interno maneja
        toda la transformación Nodo2→Nodo3→Nodo4.

    Modos de salida en predict():
        predict() soporta dos modos via return_mode:
        - 'last': devuelve solo la predicción de la última ventana (default).
          Caso estándar en producción — interesa el RUL en el ciclo actual.
        - 'all': devuelve predicciones para todas las ventanas generadas.
          Útil para visualizar la trayectoria de degradación del motor.
        Ambos modos devuelven np.ndarray para consistencia de tipos.

    evento=0 en producción:
        En producción todos los datos son naturalmente censurados (ningún
        motor ha fallado aún). El pipeline maneja esto silenciosamente —
        si evento no está en X_df simplemente no se usa. No se requiere
        ninguna acción especial.

    Historial insuficiente:
        Si X_df contiene menos de window_size ciclos, build_windows() dentro
        del pipeline lanza ValueError. Esta excepción se propaga sin atrapar —
        es responsabilidad del caller validar el historial antes de predecir.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from src.models.base_model import BaseRULModel
from src.pipeline.rul_pipeline import RULPipeline


# Claves que pertenecen al pipeline — misma definición que ggs_training_manager
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
    """Frozen Estimator para inferencia RUL en producción.

    Encapsula un RULPipeline entrenado y un BaseRULModel entrenado en un
    objeto inmutable sklearn-compatible. Recibe DataFrames de sensores crudos
    y devuelve predicciones de RUL en ciclos.

    El objeto es una caja negra de inferencia — no soporta reentrenamiento.
    Para construir un predictor desde hiperparámetros usar el classmethod
    from_config(), que gestiona internamente la construcción y entrenamiento
    del pipeline y el modelo.

    Args:
        pipeline: RULPipeline ya entrenado (is_fitted_=True). Los
            hiperparámetros de configuración se extraen automáticamente
            de este objeto.
        model: BaseRULModel ya entrenado (is_fitted_=True). Cualquier
            subclase compatible: DecisionTreeModel, RandomForestModel,
            SVRModel, NegativeBinomialPiecewise.
        model_name: Nombre legible del modelo para logging y metadata.
            Si None, se infiere de model.__class__.__name__. Defaults to None.

    Attributes:
        pipeline_: RULPipeline entrenado. Disponible tras construcción.
        model_: BaseRULModel entrenado. Disponible tras construcción.
        model_name_: Nombre del modelo resuelto. Disponible tras construcción.
        feature_set: Extraído de pipeline.feature_set.
        window_size: Extraído de pipeline.window_size.
        n_components: Extraído de pipeline.n_components.
        clipping_threshold: Extraído de pipeline.clipping_threshold.

    Raises:
        ValueError: Si pipeline.is_fitted_ es False.
        ValueError: Si model.is_fitted_ es False.

    Example:
        >>> # Construcción recomendada — via from_config
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
        >>> # Predicción estándar — última ventana
        >>> rul_now = predictor.predict(df_motor)
        >>> # Trayectoria completa de degradación
        >>> rul_traj = predictor.predict(df_motor, return_mode='all')
    """

    def __init__(
        self,
        pipeline: RULPipeline,
        model: BaseRULModel,
        model_name: str | None = None,
    ) -> None:
        # Validar que pipeline y modelo estén entrenados antes de almacenarlos
        if not getattr(pipeline, 'is_fitted_', False):
            raise ValueError(
                "El pipeline proporcionado no está entrenado (is_fitted_=False). "
                "Llama a pipeline.fit_transform() antes de construir el predictor."
            )
        if not getattr(model, 'is_fitted_', False):
            raise ValueError(
                "El modelo proporcionado no está entrenado (is_fitted_=False). "
                "Llama a model.fit() antes de construir el predictor."
            )

        # Argumentos de __init__ — para get_params() sklearn
        self.pipeline   = pipeline
        self.model      = model
        self.model_name = model_name

        # Hiperparámetros extraídos del pipeline — única fuente de verdad
        self.feature_set:        str = pipeline.feature_set
        self.window_size:        int = pipeline.window_size
        self.n_components:       int = pipeline.n_components
        self.clipping_threshold: int = pipeline.clipping_threshold

        # Atributos post-construcción — convenio sklearn con guión bajo
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
        """Construye un RULProductionPredictor desde hiperparámetros y datos.

        Factory method recomendado para construir predictores de producción.
        Recibe un dict plano con todos los hiperparámetros (pipeline + modelo
        mezclados), los separa internamente, construye y entrena el pipeline
        y el modelo, y retorna un RULProductionPredictor listo para inferencia.

        Las claves de pipeline (feature_set, window_size, n_components,
        clipping_threshold) se separan automáticamente de las claves del
        modelo — el caller no necesita hacer esta distinción.

        Si X_df e y_df no se proporcionan, los datos de entrenamiento se
        cargan automáticamente desde DatasetManager (140 motores de train
        del dataset C-MAPSS FD001).

        Args:
            model_class: Clase del modelo a instanciar. Debe ser una subclase
                de BaseRULModel compatible con el sliding window pipeline:
                NegativeBinomialPiecewise, SVRModel, DecisionTreeModel,
                RandomForestModel.
            params: Dict plano con todos los hiperparámetros mezclados.
                Las claves de pipeline (feature_set, window_size, n_components,
                clipping_threshold) se separan automáticamente. El resto
                se pasa al constructor del modelo.
            X_df: DataFrame de entrenamiento sin columna RUL. Si None,
                se carga automáticamente via DatasetManager. Defaults to None.
            y_df: DataFrame con columnas unit_number, time_in_cycles, RUL.
                Si None, se carga automáticamente via DatasetManager.
                Defaults to None.
            model_name: Nombre legible para logging y metadata. Si None,
                se infiere de model_class.__name__. Defaults to None.

        Returns:
            RULProductionPredictor listo para inferencia.

        Example:
            >>> # Uso mínimo — carga datos automáticamente
            >>> predictor = RULProductionPredictor.from_config(
            ...     model_class=NegativeBinomialPiecewise,
            ...     params={
            ...         'feature_set': 'B', 'window_size': 30,
            ...         'n_components': 20, 'clipping_threshold': 115,
            ...         'link_type': 'log', 'alpha': 0.1,
            ...         'alpha_reg': 0.0, 'l1_ratio': 0.0,
            ...     },
            ... )
            >>> # Uso con datos externos
            >>> predictor = RULProductionPredictor.from_config(
            ...     model_class=NegativeBinomialPiecewise,
            ...     params={...},
            ...     X_df=X_df,
            ...     y_df=y_df,
            ... )
        """
        # Cargar datos si no se proporcionan
        if X_df is None or y_df is None:
            X_df, y_df = cls._load_training_data()

        pipeline_params, model_params = _split_params(params)

        # Construir y entrenar pipeline sobre datos completos de train
        pipeline = RULPipeline(**pipeline_params)
        X, y_rul, t_stop, evento, groups = pipeline.fit_transform(X_df, y_df)

        # Construir y entrenar modelo sobre output del pipeline
        model = model_class(
            **model_params,
            clipping_threshold=pipeline_params['clipping_threshold'],
        )
        model.fit(X, y_rul)

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
        """Serializa el predictor y escribe su metadata JSON.

        Wrapper conveniente sobre predictor_io.save_predictor() — permite
        guardar el predictor directamente desde el objeto sin importar
        explícitamente el módulo de I/O.

        Args:
            metrics: Dict con métricas de validación cruzada del GGS.
                Claves recomendadas: mean_S_score, mean_MAE, mean_RMSE,
                mean_C_index. Si None, se guarda como dict vacío.
                Defaults to None.
            base_dir: Directorio raíz para outputs de producción.
                Si None, usa el default de predictor_io
                (outputs/production/). Defaults to None.

        Returns:
            PredictorSession con paths y metadata del predictor guardado.

        Example:
            >>> session = predictor.save()
            >>> session = predictor.save(metrics={"mean_S_score": 12.3})
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
        """No implementado — el predictor de producción es inmutable.

        Raises:
            NotImplementedError: Siempre.
        """
        raise NotImplementedError(
            "RULProductionPredictor es inmutable — no soporta reentrenamiento. "
            "Para entrenar un nuevo modelo usa GGSTrainingManager y construye "
            "un nuevo RULProductionPredictor con from_config()."
        )

    def predict(
        self,
        X_df: pd.DataFrame,
        return_mode: str = 'last',
    ) -> np.ndarray:
        """Genera predicciones de RUL a partir de un DataFrame de sensores.

        Aplica el pipeline completo (Nodo 2 → Nodo 3 → Nodo 4) sobre el
        DataFrame de entrada usando los transformadores entrenados, luego
        pasa el resultado al modelo para obtener las predicciones de RUL.

        En producción X_df contiene el historial de ciclos de un único motor
        hasta el momento actual. No debe contener la columna RUL (desconocida
        en producción). La columna evento tampoco es necesaria — si ausente,
        el pipeline la trata como censurada (evento=0) por defecto.

        Args:
            X_df: DataFrame con el historial de sensores del motor. Debe
                contener al menos window_size ciclos. Columnas requeridas:
                time_in_cycles y las 16 columnas de sensores usadas en
                entrenamiento. unit_number es opcional — si ausente se
                asigna id=0 automáticamente.
            return_mode: Modo de salida. Uno de:
                - 'last' (default): devuelve solo la predicción de la última
                  ventana — el RUL estimado en el ciclo más reciente.
                  Shape (1,).
                - 'all': devuelve predicciones para todas las ventanas
                  generadas. Útil para visualizar la trayectoria de
                  degradación. Shape (n_windows,).

        Returns:
            Array de RUL predicho clipeado a clipping_threshold.
            Shape (1,) si return_mode='last', (n_windows,) si return_mode='all'.

        Raises:
            ValueError: Si X_df tiene menos ciclos que window_size.
            ValueError: Si return_mode no es 'last' ni 'all'.
            NotFittedError: Si el objeto no fue construido correctamente
                (pipeline_ o model_ ausentes).
        """
        check_is_fitted(self, ['pipeline_', 'model_'])

        if return_mode not in ('last', 'all'):
            raise ValueError(
                f"return_mode debe ser 'last' o 'all', got '{return_mode}'."
            )

        # transform() sin y_df — modo producción sin ground truth
        X, _y_rul, _t_stop, _evento, _groups = self.pipeline_.transform(
            X_df,
            y_df=None,
        )

        y_pred = self.model_.predict(X)

        if return_mode == 'last':
            return y_pred[-1:]

        return y_pred