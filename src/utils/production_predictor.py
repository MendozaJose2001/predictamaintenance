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

    Hiperparámetros en __init__:
        Los hiperparámetros de configuración (feature_set, window_size,
        n_components, clipping_threshold) se exponen explícitamente en
        __init__ además de pipeline y model. Esto garantiza que get_params()
        devuelva la configuración completa del sistema sin necesidad de
        inspeccionar el pipeline internamente — útil para metadata y logging.

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


class RULProductionPredictor(BaseEstimator, RegressorMixin):
    """Frozen Estimator para inferencia RUL en producción.

    Encapsula un RULPipeline entrenado y un BaseRULModel entrenado en un
    objeto inmutable sklearn-compatible. Recibe DataFrames de sensores crudos
    y devuelve predicciones de RUL en ciclos.

    El objeto es una caja negra de inferencia — no soporta reentrenamiento.
    Para entrenar un nuevo modelo usar GGSTrainingManager y construir un
    nuevo RULProductionPredictor con los resultados.

    Args:
        pipeline: RULPipeline ya entrenado (is_fitted_=True). Debe contener
            el DimReducer ajustado sobre los datos de entrenamiento.
        model: BaseRULModel ya entrenado (is_fitted_=True). Cualquier
            subclase compatible: DecisionTreeModel, RandomForestModel,
            SVRModel, NegativeBinomialPiecewise.
        feature_set: Clave del conjunto de features usado en Nodo 3.
            Uno de: 'A', 'B', 'C', 'D'. Almacenado para metadata.
        window_size: Tamaño de ventana deslizante usado en Nodo 2.
            Almacenado para metadata e introspección.
        n_components: Número de componentes PCA usados en Nodo 4.
            Almacenado para metadata e introspección.
        clipping_threshold: Umbral de clipping RUL aplicado en predicción.
            Almacenado para metadata e introspección.
        model_name: Nombre legible del modelo para logging y metadata.
            Si None, se infiere de model.__class__.__name__. Defaults to None.

    Attributes:
        pipeline_: RULPipeline entrenado. Disponible tras construcción.
        model_: BaseRULModel entrenado. Disponible tras construcción.
        model_name_: Nombre del modelo resuelto. Disponible tras construcción.

    Raises:
        ValueError: Si pipeline.is_fitted_ es False.
        ValueError: Si model.is_fitted_ es False.

    Example:
        >>> predictor = RULProductionPredictor(
        ...     pipeline=trained_pipeline,
        ...     model=trained_model,
        ...     feature_set='D',
        ...     window_size=30,
        ...     n_components=15,
        ...     clipping_threshold=125,
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
        feature_set: str,
        window_size: int,
        n_components: int,
        clipping_threshold: int,
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

        # Hiperparámetros expuestos en __init__ para get_params() sklearn.
        # Nota: pipeline y model aparecen también como pipeline_ y model_
        # (ver Design decisions — Doble referencia en docstring de módulo).
        self.pipeline = pipeline
        self.model = model
        self.feature_set = feature_set
        self.window_size = window_size
        self.n_components = n_components
        self.clipping_threshold = clipping_threshold
        self.model_name = model_name

        # Atributos post-construcción — convenio sklearn con guión bajo
        self.pipeline_: RULPipeline = pipeline
        self.model_: BaseRULModel = model
        self.model_name_: str = (
            model_name
            if model_name is not None
            else model.__class__.__name__
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
            "un nuevo RULProductionPredictor con los resultados."
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