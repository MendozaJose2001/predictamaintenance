"""I/O de persistencia para RULProductionPredictor.

Este módulo gestiona la serialización, almacenamiento y carga de objetos
RULProductionPredictor entrenados. Sigue la misma filosofía que ggs_io.py —
separación completa entre lógica de persistencia y lógica del objeto.

Directory structure managed by this module:

    outputs/production/
        metadata/
            {ModelName}_{feature_set}_{window_size}w_{n_components}pc_{timestamp}.json
        models/
            {ModelName}_{feature_set}_{window_size}w_{n_components}pc_{timestamp}.pkl

Naming convention:
    El nombre del archivo se genera automáticamente a partir de los
    hiperparámetros del pipeline y el timestamp de creación:

        {ModelName}_{feature_set}_{window_size}w_{n_components}pc_{timestamp}

    Ejemplo:
        RandomForestModel_D_30w_15pc_20250509_1423

    Este nombre aparece tanto en el .pkl como en el .json de metadata,
    garantizando trazabilidad entre artefactos.

Metadata JSON structure:
    {
        "model_name":   "RandomForestModel",
        "timestamp":    "20250509_1423",
        "model_file":   "RandomForestModel_D_30w_15pc_20250509_1423.pkl",
        "pipeline_params": {
            "feature_set":        "D",
            "window_size":        30,
            "n_components":       15,
            "clipping_threshold": 125
        },
        "model_params": {
            "max_depth":        5,
            "min_samples_leaf": 10,
            "max_features":     "sqrt"
        },
        "metrics": {
            "mean_S_score": 12.3,
            "mean_MAE":     8.5,
            "mean_RMSE":    11.2,
            "mean_C_index": 0.78
        }
    }

    La información queda completamente jerarquizada en tres bloques
    temáticos: identificación, configuración (pipeline + modelo),
    y rendimiento (métricas de validación cruzada).

Design decisions:
    Separación metadata / modelo:
        El .json y el .pkl se guardan en subdirectorios distintos.
        Esto permite inspeccionar la configuración y métricas de todos
        los modelos disponibles sin cargar ningún .pkl en memoria —
        útil para selección rápida del mejor modelo en producción.

    model_params via get_params():
        Los hiperparámetros del modelo se extraen directamente del objeto
        via model.get_params(). Esto funciona porque todos los modelos
        del proyecto heredan de BaseEstimator — no se requiere ningún
        contrato adicional.

    Serialización con joblib:
        joblib es el estándar de facto para serializar objetos sklearn.
        Maneja eficientemente arrays numpy grandes (como los de RandomForest)
        mediante compresión opcional, a diferencia de pickle estándar.

    load_predictor acepta path o nombre:
        Se puede cargar un predictor por path absoluto/relativo completo,
        o por nombre de archivo relativo a base_dir/models/. Esto permite
        tanto uso programático preciso como carga conveniente por nombre.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib

from src.utils.production_predictor import RULProductionPredictor


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_BASE_DIR = Path('outputs/production')

_METADATA_DIR = 'metadata'
_MODELS_DIR   = 'models'


def _ensure_dirs(base_dir: Path) -> None:
    """Creates metadata/ and models/ subdirectories if they don't exist."""
    for subdir in [_METADATA_DIR, _MODELS_DIR]:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PredictorSession dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictorSession:
    """Encapsula los paths y metadata de un predictor guardado.

    Creada por save_predictor() — no instanciar directamente.

    Attributes:
        model_name:      Nombre de la clase del modelo. Ej: 'RandomForestModel'.
        timestamp:       Timestamp de creación 'YYYYMMDD_HHMM'.
        pipeline_params: Dict con hiperparámetros del pipeline
                         (feature_set, window_size, n_components,
                         clipping_threshold).
        model_params:    Dict con hiperparámetros propios del modelo
                         extraídos via model.get_params().
        metrics:         Dict con métricas de validación cruzada
                         (mean_S_score, mean_MAE, mean_RMSE, mean_C_index).
        path_model:      Path al archivo .pkl serializado.
        path_metadata:   Path al archivo .json de metadata.
    """
    model_name:      str
    timestamp:       str
    pipeline_params: dict
    model_params:    dict
    metrics:         dict
    path_model:      Path
    path_metadata:   Path

    @property
    def stem(self) -> str:
        """Base filename stem común a .pkl y .json.

        Format: '{ModelName}_{feature_set}_{window_size}w_{n_components}pc_{timestamp}'
        Example: 'RandomForestModel_D_30w_15pc_20250509_1423'
        """
        return self.path_model.stem


# ---------------------------------------------------------------------------
# _build_stem — nombre autogenerado
# ---------------------------------------------------------------------------

def _build_stem(
    model_name: str,
    pipeline_params: dict,
    timestamp: str,
) -> str:
    """Builds the filename stem from model name, pipeline params and timestamp.

    Args:
        model_name:      Class name of the model.
        pipeline_params: Dict with feature_set, window_size, n_components.
        timestamp:       Timestamp string 'YYYYMMDD_HHMM'.

    Returns:
        Stem string of the form:
        '{ModelName}_{feature_set}_{window_size}w_{n_components}pc_{timestamp}'
    """
    feature_set  = pipeline_params['feature_set']
    window_size  = pipeline_params['window_size']
    n_components = pipeline_params['n_components']
    return (
        f'{model_name}_{feature_set}_{window_size}w_{n_components}pc_{timestamp}'
    )


# ---------------------------------------------------------------------------
# _extract_model_params — filtra clipping_threshold de model_params
# ---------------------------------------------------------------------------

_PIPELINE_PARAM_KEYS: frozenset[str] = frozenset({
    'feature_set',
    'window_size',
    'n_components',
    'clipping_threshold',
})


def _extract_model_params(predictor: RULProductionPredictor) -> dict:
    """Extracts model-specific hyperparameters from the predictor.

    Uses model.get_params() if available (all concrete BaseRULModel
    subclasses in this project also inherit from BaseEstimator, so
    get_params() is always present in practice). Falls back to an empty
    dict if the method is absent — defensive against future model
    implementations that may not inherit from BaseEstimator.

    Removes keys that belong to the pipeline layer (clipping_threshold)
    to avoid duplication with pipeline_params.

    Args:
        predictor: Fitted RULProductionPredictor.

    Returns:
        Dict of model hyperparameters with pipeline-level keys removed.
        Empty dict if the model does not expose get_params().
    """
    raw: dict = getattr(predictor.model_, 'get_params', lambda: {})()
    return {k: v for k, v in raw.items() if k not in _PIPELINE_PARAM_KEYS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_predictor(
    predictor: RULProductionPredictor,
    metrics: dict,
    base_dir: Path = _DEFAULT_BASE_DIR,
) -> PredictorSession:
    """Serializes a RULProductionPredictor and writes its metadata JSON.

    Generates an automatic filename from the predictor's configuration,
    serializes the object with joblib, and writes a structured metadata
    JSON file with pipeline params, model params, and cross-validation
    metrics.

    Args:
        predictor: Fitted RULProductionPredictor to persist.
        metrics:   Dict with cross-validation metrics from GGS. Expected
                   keys: mean_S_score, mean_MAE, mean_RMSE, mean_C_index.
                   Additional keys are preserved as-is.
        base_dir:  Root directory for production outputs.
                   Defaults to outputs/production/.

    Returns:
        PredictorSession with paths and metadata for the saved predictor.
    """
    _ensure_dirs(base_dir)

    # Build structured metadata blocks
    pipeline_params = {
        'feature_set':        predictor.feature_set,
        'window_size':        predictor.window_size,
        'n_components':       predictor.n_components,
        'clipping_threshold': predictor.clipping_threshold,
    }
    model_params = _extract_model_params(predictor)
    model_name   = predictor.model_name_
    timestamp    = datetime.now().strftime('%Y%m%d_%H%M')
    stem         = _build_stem(model_name, pipeline_params, timestamp)

    path_model    = base_dir / _MODELS_DIR   / f'{stem}.pkl'
    path_metadata = base_dir / _METADATA_DIR / f'{stem}.json'

    # Serialize predictor
    joblib.dump(predictor, path_model)

    # Write metadata JSON
    metadata = {
        'model_name':      model_name,
        'timestamp':       timestamp,
        'model_file':      f'{stem}.pkl',
        'pipeline_params': pipeline_params,
        'model_params':    model_params,
        'metrics':         metrics,
    }
    with open(path_metadata, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    session = PredictorSession(
        model_name=model_name,
        timestamp=timestamp,
        pipeline_params=pipeline_params,
        model_params=model_params,
        metrics=metrics,
        path_model=path_model,
        path_metadata=path_metadata,
    )

    print(f"  ✅ Predictor saved: {path_model.name}")
    print(f"  📄 Metadata saved: {path_metadata.name}")

    return session


def load_predictor(
    path: str | Path,
    base_dir: Path = _DEFAULT_BASE_DIR,
) -> RULProductionPredictor:
    """Loads a RULProductionPredictor from disk.

    Accepts either a full path to the .pkl file or a filename relative
    to base_dir/models/. This allows both precise programmatic loading
    and convenient loading by name.

    Args:
        path:     Full path to the .pkl file, or filename (with or without
                  .pkl extension) relative to base_dir/models/.
        base_dir: Root directory for production outputs.
                  Defaults to outputs/production/.

    Returns:
        Loaded RULProductionPredictor ready for inference.

    Raises:
        FileNotFoundError: If the .pkl file does not exist at the
            resolved path.
        ValueError: If the loaded object is not a RULProductionPredictor.
    """
    resolved = Path(path)

    # If not absolute and doesn't exist as-is, resolve relative to models/
    if not resolved.is_absolute() and not resolved.exists():
        # Add .pkl extension if missing
        if resolved.suffix != '.pkl':
            resolved = resolved.with_suffix('.pkl')
        resolved = base_dir / _MODELS_DIR / resolved

    if not resolved.exists():
        raise FileNotFoundError(
            f"Predictor file not found: {resolved}. "
            f"Check the filename or base_dir='{base_dir}'."
        )

    obj = joblib.load(resolved)

    if not isinstance(obj, RULProductionPredictor):
        raise ValueError(
            f"Loaded object is not a RULProductionPredictor. "
            f"Got: {type(obj).__name__}."
        )

    print(f"  ✅ Predictor loaded: {resolved.name}")
    return obj