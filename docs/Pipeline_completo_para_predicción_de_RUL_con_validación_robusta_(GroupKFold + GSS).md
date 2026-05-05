# Pipeline completo para predicción de RUL con validación robusta (GroupKFold + GSS)

Este documento describe el pipeline teórico diseñado para predecir el Remaining Useful Life (RUL) de motores a partir del dataset C-MAPSS. El enfoque integra feature engineering basado en ventanas deslizantes, extracción selectiva de características con `tsfresh`, reducción de dimensionalidad con PCA y modelos de supervivencia (Random Survival Forest, XGBoost Survival, etc.), con una etapa final de conversión de la función de supervivencia a RUL mediante un percentil de confianza optimizable. Todas las transformaciones se realizan dentro de cada pliegue de la validación cruzada por grupos (GroupKFold) para evitar la fuga de información y garantizar la generalización a nuevos motores.

---

## Nodos del pipeline

### Nodo 0 – Preparación global (fuera del GSS)
- **Entrada**: Datos brutos del dataset C-MAPSS (train y test originales).
- **Acciones**:
  - Fusión de los conjuntos de entrenamiento y prueba (para tener todos los motores juntos).
  - Eliminación de columnas con **IQR global igual a cero** (sensores constantes en todo el dataset). Esto elimina variables sin capacidad predictiva.
  - Definición de la columna `RUL_real` (dada originalmente para los puntos de fallo) y de `evento` (1 en el último ciclo de motores de training, 0 en el resto, incluidos todos los ciclos de los motores de test).
- **Observación**: Este paso se realiza **una sola vez** antes de la validación cruzada. No introduce fuga porque las columnas constantes lo son en todos los motores.

---

### Nodo 1 – Escalado robusto (dentro de cada fold)
- **Objetivo**: Normalizar las características (sensores y settings) para que tengan escala comparable, mitigando el efecto de outliers.
- **Método**: `RobustScaler` (usa mediana y rango intercuartil en lugar de media y desviación típica).
- **Dentro del fold**:
  - Se ajusta el escalador con **todos los ciclos de los motores de entrenamiento**.
  - Se transforman los motores de entrenamiento (mismos ciclos) y también los de validación (usando los parámetros del entrenamiento).
- **Hiperparámetros**: Ninguno.

---

### Nodo 2 – Ventanas deslizantes y construcción de la variable objetivo (supervivencia)
- **Parámetro clave**: `window_size` (a optimizar en GSS).
- **Procedimiento**:
  - Para cada motor, ordenar los ciclos ascendentemente.
  - Para cada ciclo `t` desde `window_size` hasta el último ciclo del motor:
    - Construir la ventana que contiene los ciclos `t - window_size + 1` ... `t` (datos ya escalados).
    - Descartar los primeros `window_size - 1` ciclos de cada motor (no generan ventana completa). Esto es aceptable porque, con el modelo de RUL *piecewise linear* que incluye un clipping (valor umbral), esos ciclos caen en la zona plana y no aportan información relevante.
  - La salida para cada ventana es:
    - **Covariables**: matriz con los valores de sensores y settings (escalados) dentro de la ventana.
    - **Target para supervivencia**: tripleta `(start, stop, event)` donde:
      - `start` = ciclo inicial de la ventana.
      - `stop` = ciclo final de la ventana ( `t` ).
      - `event` = 1 si `stop` es el ciclo de fallo del motor (solo en la última ventana de los motores de training originales), 0 en cualquier otro caso.
- **RUL piecewise y clipping** (`clipping_value` es otro hiperparámetro a optimizar):
  - En el entrenamiento, se define `RUL_clipped = min(RUL_real, clipping_value)`. Esta variable no se usa como target en los modelos de supervivencia, pero sirve para construir la función de pérdida personalizada o para podar las ventanas (se puede emplear para filtrar muestras con RUL > clipping_value, aunque normalmente no es necesario).
  - En el nodo 6 (conversión a RUL) el clipping influye indirectamente porque la función de supervivencia se estima a partir de datos donde el RUL real está recortado.
- **Dentro del fold**: Todo el proceso se aplica por separado a los motores de entrenamiento y a los de validación (sobre los datos ya escalados por el nodo 1).

---

### Nodo 3 – Extracción de características con `tsfresh`
- **Objetivo**: Transformar cada ventana (serie temporal multivariante) en un vector de características numéricas que resuman la información relevante para la degradación.
- **Enfoque**: Se utiliza la librería `tsfresh` con una **configuración personalizada** (no la extracción masiva por defecto) para reducir la dimensionalidad y evitar el sobreajuste. Se seleccionan las siguientes **categorías** (justificadas por la literatura y la revisión sistemática de Vollert & Theissler 2021):
  - **Estadísticos y Descriptores**: mediana, MAD (desviación absoluta mediana), mínimo, máximo, IQR, energía robusta, etc.
  - **Tendencia y Correlación**: pendiente robusta (Theil‑Sen), autocorrelación (Spearman), tendencia lineal, etc.
- **Implementación**:
  - Se construye un diccionario `kind_to_fc_parameters` que contiene solo los `feature calculators` deseados (por ejemplo, `"mean"`, `"median"`, `"standard_deviation"`, `"agg_linear_trend"`, `"autocorrelation"`, `"partial_autocorrelation"`, etc.) con sus parámetros si es necesario.
  - Para cada sensor individualmente, se extraen las características de la serie temporal de la ventana (longitud = `window_size`). El resultado es un vector de características por sensor.
  - **Concatenación**: Los vectores de todos los sensores se concatenan para formar **una única fila** por ventana.
- **Dimensionalidad resultante** (habitual):
  - Ejemplo: 14 sensores × 10–20 características por sensor = 140–280 características por ventana. Este número es todavía alto, por lo que se requiere el siguiente nodo de reducción dimensional.
- **Dentro del fold**: La extracción se aplica por separado a las ventanas de entrenamiento y validación. Es determinista (no requiere ajuste de parámetros), salvo la configuración de los calculators, que se trata como hiperparámetro (por ejemplo, elegir entre usar mediana o media, incluir o no FFT, etc.).
- **Hiperparámetros asociados**: La propia selección de calculators y sus parámetros (por ejemplo, `lag` en autocorrelación, `bin` en entropía) pueden ser parte del GSS, aunque por simplicidad inicial se fija una configuración razonable.

---

### Nodo 4 – PCA global (reducción de dimensionalidad)
- **Objetivo**: Reducir el número de características (salida del nodo 3) a un conjunto más pequeño de componentes no correlacionados que retengan la mayor parte de la varianza. Esto combate la maldición de la dimensionalidad y facilita el entrenamiento de modelos con validación por grupos.
- **Método**: PCA (Principal Component Analysis) estándar, aplicado sobre la matriz de características concatenadas (todas las ventanas, todos los sensores juntos). **No** se aplica PCA por separado a cada sensor.
- **Dentro del fold**:
  - Se ajusta el PCA exclusivamente con las características de las ventanas de entrenamiento.
  - Se transforman las ventanas de entrenamiento y las de validación usando los componentes obtenidos.
- **Hiperparámetro**: `n_components` (número de componentes principales a retener). Se optimiza en el GSS (junto con `window_size`, `clipping_value`, etc.). Rango típico: 5 a 30.

---

### Nodo 5 – Modelo de supervivencia
- **Objetivo**: A partir de las características reducidas (salida del nodo 4), estimar la función de supervivencia `S(t)` (probabilidad de que el motor sobreviva más allá de un tiempo `t` desde el instante actual).
- **Modelos soportados** (sin fragilidad, para evitar problemas de convergencia):
  - Random Survival Forest (RSF) de `scikit-survival`.
  - XGBoost con objetivo `survival:cox` (o `aft`).
  - Survival Tree.
  - Red neuronal con pérdida de Cox (opcional).
- **Formato de entrada** (X, y):
  - **X**: matriz de características transformada por PCA (`n_samples × n_components`).
  - **y**: estructura que contiene `(start, stop, event)`. Los modelos de supervivencia requieren que el tiempo se exprese en ciclos (no en RUL). Los intervalos `(start, stop)` definen la ventana de observación; `event` indica si al final del intervalo ocurrió el fallo.
- **Dentro del fold**:
  - El modelo se entrena con los datos de entrenamiento (después de PCA).
  - Se evalúa sobre los datos de validación (transformados con los mismos componentes y la misma configuración de extracción).
- **Hiperparámetros**: Dependen del modelo (ej. `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`, etc.). Se optimizan conjuntamente en el GSS.

**Nota importante**: No se incluye ningún efecto aleatorio por cluster (fragilidad) porque se ha demostrado que con un solo evento por motor la estimación de la fragilidad no converge. El uso de GroupKFold en la validación ya protege contra la correlación intra-motor.

---

### Nodo 6 – Conversión de la función de supervivencia a RUL
- **Objetivo**: Obtener una predicción puntual del RUL a partir de `S(t)`.
- **Procedimiento**:
  1. A partir de `S(t)` se calcula la función de fallo acumulado `F(t) = 1 - S(t)`.
  2. Se selecciona un **percentil (o nivel de confianza)** `p` (por ejemplo, 0.5 para la mediana).
  3. Se determina el tiempo `t_p` tal que `F(t_p) = p`. Es decir, el instante (en ciclos desde el inicio de la ventana) en el que la probabilidad acumulada de fallo alcanza `p`.
  4. El **ciclo de fallo estimado** es `stop + t_p` (donde `stop` es el ciclo actual, el final de la ventana).
  5. El RUL predicho es `(stop + t_p) - stop = t_p`, es decir, directamente `t_p`.
- **Interpretación**: Diferentes valores de `p` reflejan diferentes políticas de mantenimiento. Un `p` bajo (ej. 0.2) da un RUL más corto (conservador), mientras que un `p` alto (ej. 0.8) da un RUL más largo (arriesgado).
- **Hiperparámetro adicional**: `percentil` (o `nivel_confianza`) que se optimiza en el GSS junto con los demás. Su optimización permite ajustar la predicción a la métrica de evaluación (por ejemplo, la función Score penaliza más las predicciones tardías, por lo que el GSS tenderá a elegir un percentil más bajo).
- **Dentro del fold**: La conversión se aplica a las predicciones de los datos de validación (y posteriormente a los de test). Dado que `p` es un hiperparámetro global, se prueba el mismo valor en todos los folds durante la búsqueda.

---

## Validación cruzada: GroupKFold y GSS

- **Estrategia**: GroupKFold con `groups = motor_id` (unit_number). Cada motor completo está en un solo pliegue, ya sea en entrenamiento o en validación. Así se evita que ventanas del mismo motor aparezcan en ambos conjuntos, lo que provocaría una fuga de información.
- **Búsqueda de hiperparámetros (GSS)**:
  - Se define una rejilla (`grid`) que incluye:
    - `window_size` (ej. 20, 30, 40)
    - `clipping_value` (ej. 125, 150, 200)
    - `n_components` (ej. 5, 10, 15, 20)
    - Parámetros específicos de `tsfresh` (p.ej. uso de `mean` o `median`, `lag` para autocorrelación)
    - Hiperparámetros del modelo de supervivencia (max_depth, learning_rate, n_estimators, etc.)
    - `percentil` (ej. 0.3, 0.5, 0.7)
  - Para cada combinación se ejecuta el GroupKFold (por ejemplo, 5 splits) y se calcula la métrica de rendimiento (RMSE, Score o R2) sobre las predicciones de RUL finales (salida del nodo 6).
  - Se selecciona la combinación que optimiza la métrica elegida.
- **Prevención de fugas**:
  - El escalado (nodo 1) y el PCA (nodo 4) se ajustan **dentro de cada fold** usando solo los motores de entrenamiento.
  - La extracción de características (nodo 3) es determinista y no requiere parámetros, pero si utilizara estadísticas globales (ej. normalización adicional) también debería hacerse dentro del fold.

---

## Flujo completo (resumen para implementación)
Datos brutos (train+test originales)
│
▼
[Nodo 0] Eliminar columnas con IQR global = 0
│
▼
Para cada combinación de hiperparámetros (GSS):
│
Para cada fold (GroupKFold, grupos = motor_id):
│ │
│ ▼
│ [Nodo 1] RobustScaler.fit() con motores entrenamiento → transform(train+val)
│ │
│ ▼
│ [Nodo 2] Construir ventanas (window_size, clipping_value) → obtener (X_ventana, start, stop, event)
│ │
│ ▼
│ [Nodo 3] Aplicar tsfresh con configuración personalizada → X_caract (muestras × n_features)
│ │
│ ▼
│ [Nodo 4] PCA.fit(X_caract_train) → transform(train+val) → X_pca
│ │
│ ▼
│ [Nodo 5] Entrenar modelo de supervivencia (RSF, XGBoost, etc.) con X_pca_train y (start, stop, event)
│ │
│ ▼
│ [Nodo 6] Obtener S(t) para X_pca_val → convertir a RUL usando percentil → evaluar
│
▼
Promediar métricas, seleccionar mejor combinación de hiperparámetros


---

## Decisiones de diseño fundamentales (justificación)

| Decisión | Justificación |
|----------|----------------|
| **Usar GroupKFold por motor** | Evita que ventanas del mismo motor se separen entre entrenamiento y validación, lo que generaría resultados optimistas y no generalizables. |
| **Escalado y PCA dentro de cada fold** | Previene la fuga de información de los datos de validación hacia los parámetros de transformación. |
| **Eliminación de sensores con IQR global cero** | Son constantes en todo el dataset, no aportan información. Es un paso seguro previo al GSS. |
| **Ventanas deslizantes hacia atrás (lookback)** | Permite predecir el RUL en el ciclo actual usando solo información pasada, como se requiere en una aplicación real. |
| **Descartar los primeros `window_size‑1` ciclos** | Con RUL piecewise + clipping esos ciclos caen en la zona plana y no aportan información relevante. Simplifica la implementación sin pérdida significativa. |
| **Extracción selectiva con `tsfresh` (no exhaustiva)** | Reduce la dimensionalidad inicial (frente a las 800+ características por defecto), facilita el PCA y evita el sobreajuste, manteniendo las categorías más relevantes (estadísticos y tendencia/correlación). |
| **PCA global sobre todas las características concatenadas** | Es la práctica estándar en la literatura (Alomari et al. 2023) y permite interpretar los loadings en términos de combinaciones de sensores. |
| **Modelos de supervivencia sin fragilidad** | Los modelos con fragilidad no convergen con un solo evento por motor (comprobado empíricamente). Los modelos estándar (RSF, XGBoost Survival) no tienen ese problema y funcionan con el formato (start, stop, event). |
| **Conversión de S(t) a RUL mediante percentil optimizable** | La función de supervivencia proporciona una distribución; elegir un percentil permite ajustar la predicción al coste asimétrico de las predicciones tardías (penalizadas más en la métrica Score). Optimizar el percentil en el GSS mejora el rendimiento en esa métrica. |

---

## Posibles extensiones y mejoras futuras

- Añadir características bivariantes (correlaciones entre pares de sensores, diferencias) en el nodo 3.
- Probar modelos de deep learning (LSTM, CNN) en lugar de modelos tabulares, manteniendo la misma estructura de ventanas y PCA.
- Utilizar selección de características más avanzada (AFICv) después del PCA, como en Alomari et al.
- Implementar un sistema de pesos en la función de pérdida para dar más importancia a las predicciones en la región cercana al fallo.

---

*Documento generado como parte del diseño metodológico del proyecto de predicción de RUL sobre el dataset C-MAPSS.*