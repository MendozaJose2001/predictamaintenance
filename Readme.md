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
    ↓ Modelo  — NegativeBinomial, SVR, DecisionTree o RandomForest
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
│   │   └── random_forest.py
│   ├── utils/
│   │   ├── ggs_io.py               ← I/O del GGS (checkpoints, resultados)
│   │   └── validate_regressor.py   ← Validación rápida de cualquier modelo
│   ├── dataset_manager.py
│   ├── ggs_training_manager.py
│   └── metrics_manager.py
├── outputs/
│   └── ggs/
│       ├── metadata/       ← param_grid y configuración de cada run
│       ├── checkpoints/    ← estado parcial (se elimina al completar)
│       └── results/        ← resultados finales en CSV
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
```

### Opciones disponibles

```bash
python main.py --model rf --folds 5 --top 15 --jobs 2
```

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--model` | requerido | `nb`, `svr`, `dt` o `rf` |
| `--folds` | 5 | Número de folds GroupKFold |
| `--top` | 10 | Configuraciones a mostrar al terminar |
| `--jobs` | 1 | Núcleos paralelos (ver sección de paralelismo) |
| `--debug` | False | Grid reducido para verificación rápida |

### Verificación rápida (2 configs, 2 folds, ~2 min)

```bash
python main.py --model dt --debug
python main.py --model rf --debug
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

**Nota para Random Forest:** `n_jobs=1` está fijo internamente en el modelo para evitar conflictos con el paralelismo del GGS.

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
| `Success` | 1 si la config convergió, 0 si falló |

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