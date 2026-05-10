# PredictaMaintenance

Sistema de estimación de Vida Útil Remanente (RUL) para motores turbofan, desarrollado sobre el dataset NASA C-MAPSS FD001. Implementa modelos de machine learning clásico con un pipeline de ventanas deslizantes, extracción de características estadísticas y reducción de dimensionalidad mediante PCA.

---

## ¿Qué hace este sistema?

A partir de series temporales de sensores de motores en degradación, el sistema predice cuántos ciclos le quedan a cada motor antes del fallo. El pipeline sigue cuatro etapas:

```
Sensores crudos
    ↓ Nodo 2 — Ventanas deslizantes
    ↓ Nodo 3 — Extracción de características estadísticas
    ↓ Nodo 4 — RobustScaler + PCA global
    ↓ Modelo  — NegativeBinomial, SVR, DecisionTree, RandomForest o XGBoost
        → MAE, RMSE, S-Score, C-Index
```

La selección de hiperparámetros (conjunto de features, tamaño de ventana, componentes PCA, parámetros del modelo) se realiza mediante **Group Grid Search** con validación cruzada GroupKFold por motor — garantizando que ningún motor aparece simultáneamente en entrenamiento y validación.

---

## Estructura del proyecto

```
├── data/
│   └── clean/              ← CSVs preprocesados, uno por motor
├── src/
│   ├── pipeline/
│   │   ├── windowing.py            ← Nodo 2: ventanas deslizantes
│   │   ├── feature_extraction.py   ← Nodo 3: características estadísticas
│   │   ├── dim_reduction.py        ← Nodo 4: RobustScaler + PCA
│   │   └── rul_pipeline.py         ← Orquestador Nodos 2-4
│   ├── models/
│   │   ├── negative_binomial.py
│   │   ├── svr_model.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── xgb_model.py
│   ├── utils/
│   │   ├── ggs_io.py               ← I/O del GGS (checkpoints, resultados)
│   │   ├── validate_regressor.py   ← Validación rápida de cualquier modelo
│   │   ├── production_predictor.py ← Frozen Estimator para producción
│   │   └── predictor_io.py         ← I/O de modelos de producción
│   ├── dataset_manager.py
│   ├── ggs_training_manager.py
│   └── metrics_manager.py
├── outputs/
│   ├── ggs/
│   │   ├── metadata/       ← param_grid y configuración de cada run
│   │   ├── checkpoints/    ← estado parcial (se elimina al completar)
│   │   └── results/        ← resultados finales en CSV
│   └── production/
│       ├── metadata/       ← JSON con configuración y métricas del predictor
│       └── models/         ← predictores serializados (.pkl)
├── tests/
├── main.py                 ← CLI de entrenamiento
└── requirements.txt
```

---

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd Premant

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Entrenar un modelo

### Uso básico

```bash
# Entrenar Negative Binomial
python main.py --model nb

# Entrenar SVR
python main.py --model svr

# Entrenar Decision Tree
python main.py --model dt

# Entrenar Random Forest
python main.py --model rf

# Entrenar XGBoost
python main.py --model xgb
```

### Opciones disponibles

```bash
python main.py --model rf --folds 5 --top 15 --jobs 2
```

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--model` | requerido | `nb`, `svr`, `dt`, `rf` o `xgb` |
| `--folds` | 5 | Número de folds GroupKFold |
| `--top` | 10 | Configuraciones a mostrar al terminar |
| `--jobs` | 1 | Núcleos paralelos (ver sección de paralelismo) |
| `--debug` | False | Grid reducido para verificación rápida |

### Verificación rápida (2 configs, 2 folds, ~2 min)

```bash
python main.py --model dt --debug
python main.py --model rf --debug
python main.py --model xgb --debug
```

---

## Paralelismo

El GGS evalúa configuraciones en paralelo usando `joblib`. El número de núcleos se controla con `--jobs`:

```bash
# Secuencial — un config a la vez (default)
python main.py --model rf

# 2 núcleos en paralelo
python main.py --model rf --jobs 2

# 3 núcleos (recomendado para máquinas con 4+ cores)
python main.py --model rf --jobs 3

# Todos los núcleos disponibles
python main.py --model rf --jobs -1
```

**Comportamiento por modo:**

| | `--jobs 1` | `--jobs N>1` |
|--|-----------|-------------|
| Progreso | por configuración | por configuración |
| Checkpointing | ✅ después de cada config | ✅ con proceso principal |
| Resume | ✅ | ✅ |

**Recomendación:** usar `--jobs 2` o `--jobs 3` — el overhead de comunicación hace que valores más altos no siempre escalen linealmente para este pipeline.

**Nota para Random Forest y XGBoost:** `n_jobs=1` está fijo internamente en ambos modelos para evitar conflictos con el paralelismo del GGS.

---

## Validación rápida de modelos

Antes de lanzar un GGS completo, se puede validar cualquier modelo con un motor o con GroupKFold:

```bash
# Entrenamiento completo (140 motores)
python -m src.utils.validate_regressor --model dt
python -m src.utils.validate_regressor --model rf

# GroupKFold (simula el GGS real)
python -m src.utils.validate_regressor --model dt --kfold
python -m src.utils.validate_regressor --model rf --kfold

# Con parámetros personalizados
python -m src.utils.validate_regressor --model rf --kfold --window-size 30 --feature-set B
```

---

## Resultados

Al terminar el GGS, los resultados se guardan automáticamente:

```
outputs/ggs/results/{ModelName}_{hash}_{timestamp}.csv
```

Cada fila corresponde a una configuración de hiperparámetros con sus métricas promedio en validación cruzada:

| Columna | Descripción |
|---------|-------------|
| `feature_set` | Conjunto de características (A/B/C/D) |
| `window_size` | Tamaño de ventana deslizante |
| `n_components` | Componentes PCA |
| `clipping_threshold` | Umbral de clipping del RUL |
| `mean_S_score` | S-Score NASA (menor es mejor) |
| `mean_C_index` | C-Index (mayor es mejor) |
| `mean_MAE` | Error absoluto medio en ciclos |
| `mean_RMSE` | Error cuadrático medio en ciclos |

Para analizar los resultados de un GGS:

```bash
python analyze_ggs_results.py
```

### Conjuntos de características disponibles

| Set | Características | Features |
|-----|----------------|---------|
| A | mean, std, rms, slope, rvalue | 80 |
| B | median, rms, q25, q75, slope, rvalue, autocorr×2 | 128 |
| C | B + runs_ratio, hurst_rs | 160 |
| D | B + fft_coef_1/2/3 | 176 |

---

## Reanudar un GGS interrumpido

El sistema guarda un checkpoint después de cada configuración evaluada. Si el proceso se interrumpe, basta con volver a ejecutar el mismo comando — el GGS detecta automáticamente el checkpoint y continúa desde donde quedó:

```bash
# Primera ejecución — se interrumpe a mitad
python main.py --model rf

# Segunda ejecución — reanuda automáticamente
python main.py --model rf
```

El checkpoint se identifica por el modelo y el `param_grid`. Si se cambia el `param_grid`, se crea una sesión nueva sin afectar la anterior.

---

## Correr los tests

```bash
pytest tests/ -v
```

Los tests usan fixtures session-scoped — Nodo 3 (extracción de features) corre una sola vez para toda la suite.

---

## Hiperparámetros del GGS

### Negative Binomial

| Hiperparámetro | Valores |
|---------------|---------|
| feature_set | A, B, C, D |
| window_size | 15, 20, 25, 30 |
| n_components | 10, 15, 20 |
| clipping_threshold | 115, 120, 125, 130 |
| link_type | log |
| alpha (dispersión) | 0.1, 0.5, 1.0, 1.5 |
| alpha_reg (penalización) | 0.0, 0.1, 0.5 |
| l1_ratio | 0.0, 0.5, 1.0 |

### SVR

| Hiperparámetro | Valores |
|---------------|---------|
| feature_set | A, B, C, D |
| window_size | 15, 20, 25, 30 |
| n_components | 10, 15, 20 |
| clipping_threshold | 115, 120, 125, 130 |
| kernel | rbf, linear, poly |
| C | 0.1, 1.0, 10.0 |
| epsilon | 0.01, 0.1 |
| gamma | 'scale', 0.01 |
| degree | 2, 3 |

### Decision Tree

| Hiperparámetro | Valores |
|---------------|---------|
| feature_set | A, B, C, D |
| window_size | 15, 20, 25, 30 |
| n_components | 10, 15, 20 |
| clipping_threshold | 115, 120, 125, 130 |
| max_depth | 5, 10, None |
| min_samples_leaf | 1, 5, 10 |
| min_samples_split | 2, 10 |
| max_features | 'sqrt', 1.0 |

### Random Forest

| Hiperparámetro | Valores |
|---------------|---------|
| feature_set | A, B, C, D |
| window_size | 15, 20, 25, 30 |
| n_components | 10, 15, 20 |
| clipping_threshold | 115, 120, 125, 130 |
| n_estimators | 50, 100, 200 |
| max_depth | 5, 10, None |
| min_samples_leaf | 1, 5, 10 |
| max_features | 'sqrt', 1.0 |

### XGBoost

> ⚠️ **Advertencia:** El espacio de búsqueda de XGBoost (~10,000 configuraciones) es significativamente mayor que el de los modelos anteriores. Se recomienda correrlo solo después de haber analizado los resultados de DT y RF — si hay consistencia en los hiperparámetros del pipeline óptimos, se pueden fijar para reducir el grid antes de lanzar este GGS.

| Hiperparámetro | Valores |
|---------------|---------|
| feature_set | A, B, C, D |
| window_size | 20, 25, 30 |
| n_components | 10, 15, 20 |
| clipping_threshold | 115, 120, 125 |
| n_estimators | 200, 300 |
| learning_rate | 0.05, 0.1 |
| max_depth | 3, 5 |
| subsample | 0.8, 1.0 |
| colsample_bytree | 0.8, 1.0 |
| reg_lambda | 0.1, 1.0, 10.0 |
| min_child_weight | 1, 5 |

Parámetros internos fijos (no forman parte del grid): `objective='reg:squarederror'`, `tree_method='hist'`, `n_jobs=1`, `random_state=42`.

---

## Modelos de producción

Una vez completado el GGS y seleccionada la configuración ganadora, el sistema permite empaquetar el modelo entrenado en un objeto de producción listo para inferencia.

### ¿Qué es un `RULProductionPredictor`?

Es un **Frozen Estimator** sklearn-compatible que encapsula el pipeline completo (Nodos 2-4) y el modelo entrenado en un único objeto inmutable. Recibe datos crudos de sensores y devuelve predicciones de RUL — sin necesidad de gestionar el pipeline externamente.

El objeto es una caja negra de inferencia: no soporta reentrenamiento. Para obtener un nuevo predictor hay que crear uno desde cero con `from_config()`.

### Entrada esperada

Un `pd.DataFrame` con el historial de ciclos de **un único motor**, con las mismas columnas de sensores usadas en entrenamiento:

| Columna | Requerida | Notas |
|---------|-----------|-------|
| `time_in_cycles` | ✅ Sí | Índice de ciclo, estrictamente creciente |
| `op_setting_1` | ✅ Sí | Setting operacional 1 |
| `op_setting_2` | ✅ Sí | Setting operacional 2 |
| `T24` | ✅ Sí | Temperatura total salida LPC (°R) |
| `T30` | ✅ Sí | Temperatura total salida HPC (°R) |
| `T50` | ✅ Sí | Temperatura total salida LPT (°R) |
| `P30` | ✅ Sí | Presión total salida HPC (psia) |
| `Nf` | ✅ Sí | Velocidad física del fan (rpm) |
| `Nc` | ✅ Sí | Velocidad física del núcleo (rpm) |
| `Ps30` | ✅ Sí | Presión estática salida HPC (psia) |
| `phi` | ✅ Sí | Ratio flujo combustible / Ps30 (pps/psi) |
| `NRf` | ✅ Sí | Velocidad corregida del fan (rpm) |
| `NRc` | ✅ Sí | Velocidad corregida del núcleo (rpm) |
| `BPR` | ✅ Sí | Bypass ratio |
| `htBleed` | ✅ Sí | Entalpía de sangrado |
| `W31` | ✅ Sí | Sangrado de refrigeración HPT (lbm/s) |
| `W32` | ✅ Sí | Sangrado de refrigeración LPT (lbm/s) |
| `unit_number` | No | Si ausente, se asigna `id=0` automáticamente |
| `evento` | No | Si ausente, se trata como censurado (`evento=0`) |
| `RUL` | No | No existe en producción — no se debe incluir |

El DataFrame debe contener **al menos `window_size` ciclos** — en caso contrario se lanza `ValueError`.

El valor de `window_size` depende de la configuración con la que fue entrenado el predictor. Se puede consultar de tres formas:

```python
# 1. Directamente desde el objeto en memoria
print(predictor.window_size)

# 2. Desde el pipeline interno
print(predictor.pipeline_.window_size)

# 3. Desde el JSON de metadata (sin cargar el .pkl)
import json
with open('outputs/production/metadata/NegativeBinomialPiecewise_B_30w_20pc_20260509_1755.json') as f:
    meta = json.load(f)
print(meta['pipeline_params']['window_size'])  # → 30
```

La tercera opción es útil para validar el historial disponible **antes** de cargar el modelo en memoria.

### Salida esperada

Un `np.ndarray` con predicciones de RUL en ciclos, clippeado a `clipping_threshold`:

```python
# Modo 'last' (default) — RUL estimado en el ciclo más reciente
rul_now = predictor.predict(df_motor)
# shape: (1,)  →  rul_now[0] = 42.3 ciclos

# Modo 'all' — trayectoria completa de degradación
rul_traj = predictor.predict(df_motor, return_mode='all')
# shape: (n_windows,)  →  [115.0, 114.2, ..., 42.3]
```

### Crear un predictor de producción

```python
from src.models.negative_binomial import NegativeBinomialPiecewise
from src.utils.production_predictor import RULProductionPredictor

# Parámetros ganadores del GGS (pipeline + modelo mezclados)
best_params = {
    # Pipeline
    'feature_set':        'B',
    'window_size':        30,
    'n_components':       20,
    'clipping_threshold': 115,
    # Modelo
    'link_type': 'log',
    'alpha':     0.1,
    'alpha_reg': 0.0,
    'l1_ratio':  0.0,
}

# from_config entrena internamente sobre los 140 motores de train
predictor = RULProductionPredictor.from_config(
    model_class=NegativeBinomialPiecewise,
    params=best_params,
)
```

También es posible pasar los datos de entrenamiento explícitamente:

```python
predictor = RULProductionPredictor.from_config(
    model_class=NegativeBinomialPiecewise,
    params=best_params,
    X_df=X_df,
    y_df=y_df,
)
```

### Guardar un predictor

```python
# Sin métricas
session = predictor.save()

# Con métricas del GGS (recomendado para trazabilidad)
session = predictor.save(metrics={
    'mean_S_score': 12.3,
    'mean_MAE':     8.5,
    'mean_RMSE':    11.2,
    'mean_C_index': 0.78,
})

print(session.path_model)     # outputs/production/models/NegativeBinomialPiecewise_B_30w_20pc_...pkl
print(session.path_metadata)  # outputs/production/metadata/NegativeBinomialPiecewise_B_30w_20pc_...json
```

El archivo `.json` de metadata tiene la siguiente estructura:

```json
{
    "model_name": "NegativeBinomialPiecewise",
    "timestamp": "20260509_1755",
    "model_file": "NegativeBinomialPiecewise_B_30w_20pc_20260509_1755.pkl",
    "pipeline_params": {
        "feature_set": "B",
        "window_size": 30,
        "n_components": 20,
        "clipping_threshold": 115
    },
    "model_params": {
        "link_type": "log",
        "alpha": 0.1,
        "alpha_reg": 0.0,
        "l1_ratio": 0.0
    },
    "metrics": {
        "mean_S_score": 12.3,
        "mean_MAE": 8.5,
        "mean_RMSE": 11.2,
        "mean_C_index": 0.78
    }
}
```

### Cargar un predictor

```python
from src.utils.predictor_io import load_predictor

# Por nombre de archivo (con o sin extensión .pkl)
predictor = load_predictor('NegativeBinomialPiecewise_B_30w_20pc_20260509_1755')

# Por path absoluto
predictor = load_predictor('outputs/production/models/NegativeBinomialPiecewise_B_30w_20pc_20260509_1755.pkl')
```

### Usar el predictor en producción

```python
import pandas as pd

# Cargar historial del motor a evaluar
df_motor = pd.read_csv('data/clean/data_motor_96.csv')
df_motor.insert(0, 'unit_number', 96)
X_motor = df_motor.drop(columns=['RUL'])

# Predicción puntual — ciclo actual
rul_ahora = predictor.predict(X_motor)
print(f"RUL estimado: {rul_ahora[0]:.1f} ciclos")

# Trayectoria completa
rul_trayectoria = predictor.predict(X_motor, return_mode='all')
```

### Exportar a otro proyecto

El predictor se serializa con `joblib` — es completamente portable. Para usarlo en otro proyecto basta con:

1. Copiar el archivo `.pkl` desde `outputs/production/models/`
2. Instalar las mismas dependencias (`requirements.txt`)
3. Copiar los módulos `src/utils/production_predictor.py` y `src/utils/predictor_io.py`

```python
import joblib

# Cargar directamente sin predictor_io
predictor = joblib.load('NegativeBinomialPiecewise_B_30w_20pc_20260509_1755.pkl')
rul = predictor.predict(df_motor)
```

> **Nota:** El `.pkl` contiene el pipeline completo (scaler + PCA) y el modelo entrenado. No se requiere acceso al dataset original ni reentrenamiento.

---

## Dataset

NASA C-MAPSS FD001 — 200 motores turbofan simulados, una condición operativa, un modo de fallo.

| Propiedad | Valor |
|-----------|-------|
| Motores de entrenamiento | 140 (70%) |
| Motores de test | 60 (30%) |
| Sensores | 16 (tras eliminar varianza cero) |
| Ciclos por motor | 128–362 |

La división train/test es fija y reproducible via `DatasetManager.split_dataset()`.

---

## Referencia

Alomari, Y., Andó, M., & Baptista, M. L. (2023). Advancing aircraft engine RUL predictions: an interpretable integrated approach of feature engineering and aggregated feature importance. *Scientific Reports*, 13, 13466.