# Metodología de Selección de Características — PredictaMaintenance
## Evaluación Bootstrap Pareada de Conjuntos de Características

**Versión:** 2.0  
**Estado:** Pendiente de revisión del asesor  
**Equipo:** PredictaMaintenance

---

## 1. Contexto y Motivación

### 1.1 Dataset: NASA C-MAPSS FD001

C-MAPSS FD001 contiene datos de degradación hasta fallo de motores turbofan de aeronaves:

| Propiedad | Valor |
|-----------|-------|
| Total de motores | 200 |
| Conjunto de entrenamiento (70%) | 140 motores |
| Conjunto de prueba (30%) | 60 motores |
| Sensores | 21 originales → 16 tras eliminar varianza cero |
| Condiciones operativas | 1 |
| Modos de fallo | 1 |
| Ciclos por motor | 128–362 |
| Indicador de evento | evento=1 únicamente en el último ciclo de cada motor de entrenamiento |

El dataset fue diseñado para regresión de RUL, no para análisis de supervivencia. Cada motor tiene exactamente un evento terminal al final de su vida operativa.

### 1.2 Arquitectura del Pipeline

El pipeline de PredictaMaintenance transforma series temporales de sensores crudos en matrices de características mediante cuatro etapas secuenciales:

```
Sensores crudos (unit_number, time_in_cycles, sensores, evento)
    ↓
Nodo 2 — build_windows(window_size)
    Ventanas deslizantes de window_size ciclos por motor
    Salida: (n_ventanas, window_size, n_sensores) por motor

    ↓
Nodo 3 — extract_window_features
    Características estadísticas y de tendencia sobre el eje de ventana
    Salida: (n_ventanas, n_características) por motor

    ↓
Nodo 4 — DimReducer(n_components)
    RobustScaler interno + PCA Global
    Ajustado únicamente sobre datos de entrenamiento (sin data leakage)
    Salida: (n_ventanas, n_components) por motor

    ↓
Modelo (NB, SVR, etc.)
    Recibe matriz plana (total_ventanas, n_components)
    El modelo es agnóstico al motor
```

**Decisiones de diseño ya validadas:**
- El Nodo 1 (FeatureScaler sobre sensores crudos) fue eliminado — redundante dado el RobustScaler interno del DimReducer
- El PCA se ajusta globalmente sobre todos los motores de entrenamiento concatenados — no por motor individual
- El clipping_threshold afecta únicamente la construcción del target y_rul, no las características X

### 1.3 Conjunto de Características Actual

Se extraen características sobre cada ventana de window_size ciclos para cada uno de los 16 sensores:

**Estadísticos descriptivos:**
- `median`, `abs_energy`, `q25`, `q75`

**Tendencia y correlación:**
- `slope` (pendiente OLS), `rvalue` (Pearson r con el tiempo)
- `autocorr_lag_1/2/3`, `partial_autocorr_lag_1/2/3`

Total: 8 tipos × 16 sensores = **128 características** por ventana.

---

## 2. Pregunta de Investigación

> **¿Qué combinación de características estadísticas extraídas sobre ventanas deslizantes retiene la mayor cantidad de información sobre la degradación del motor, medida mediante la varianza explicada acumulada de un PCA global ajustado sobre la flota de entrenamiento?**

### 2.1 Por qué la varianza explicada del PCA como criterio

El modelo predictor (NB, SVR) es agnóstico al motor — recibe una matriz plana de componentes PCA y aprende el mapeo hacia el RUL. La calidad de este mapeo depende críticamente de cuánta información de degradación sobrevive la cadena extracción de características → compresión PCA.

La varianza explicada acumulada del PCA (cumvar) mide exactamente esto: **qué proporción de la información original de las características se retiene en los primeros n_components componentes principales**. Un conjunto de características que alcanza mayor cumvar bajo los mismos n_components proporciona al modelo una representación más rica e informativa de la degradación del motor.

Este criterio:
- No requiere entrenar ningún modelo predictor → sin circularidad ✅
- Es independiente del predictor específico utilizado → generalizable ✅
- Mide directamente la retención de información → interpretable ✅
- Está establecido en la literatura de PHM (Alomari et al. 2023) ✅

### 2.2 Por qué NO usar rendimiento predictivo (R², MAE) como criterio

Usar el rendimiento del modelo (regresión Ridge sobre PCs → R²) introduciría el supuesto de que el RUL y los componentes PCA están linealmente relacionados. Dado que los predictores reales (NB, SVR) son no lineales, este proxy podría ordenar sistemáticamente mal los conjuntos de características. Delegamos la responsabilidad de encontrar la relación características-RUL a los modelos — nuestra tarea es proporcionarles características maximalmente informativas.

---

## 3. Conjuntos Candidatos de Características

Se evalúan cinco conjuntos diseñados sistemáticamente para explorar diferentes dominios de información. El Conjunto B es la referencia del pipeline actual; los conjuntos C, D y E añaden un dominio adicional de forma independiente para medir su aporte incremental. El Conjunto A es el baseline mínimo para anclar la comparación.

### 3.1 Definición de los Conjuntos

---

**Conjunto A — Baseline mínimo**

*Composición:* `mean`, `std`, `rms`, `slope`, `rvalue`
*Dimensión:* 5 tipos × 16 sensores = **80 características**

*Dominio de información:* Distribución básica + tendencia lineal

*Justificación:* Representa el conjunto mínimo coherente para capturar el nivel, la dispersión y la tendencia temporal de cada sensor. Sirve como ancla inferior de la comparación — si los conjuntos más complejos no superan significativamente este baseline, justifica una implementación minimal.

*Características excluidas y razón:*
- `median`, `q25`, `q75`: redundantes con `mean` y `std` en este contexto minimal
- `abs_energy`: redundante con `rms` (abs_energy = rms² × window_size con window_size constante)
- Autocorrelaciones: se evalúan en conjunto separado
- FFT y entropía: se evalúan en conjuntos separados

---

**Conjunto B — Referencia del pipeline actual**

*Composición:* `median`, `rms`, `q25`, `q75`, `slope`, `rvalue`, `autocorr_lag_1`, `autocorr_lag_2`
*Dimensión:* 8 tipos × 16 sensores = **128 características**

*Dominio de información:* Distribución robusta + tendencia + memoria de corto plazo

*Justificación:* Refleja el conjunto actual del pipeline de PredictaMaintenance. Usa `median` y cuantiles en lugar de `mean` y `std` por su robustez ante valores atípicos generados por la degradación en ciclos tardíos. Incluye los dos primeros lags de autocorrelación para capturar persistencia de corto plazo.

*Nota sobre abs_energy vs rms:* El análisis empírico sobre motor 1 confirmó que `abs_energy` produce rangos de hasta 2.47×10⁹ mientras que `rms` produce rangos de hasta 9,055 — misma varianza explicada en PCA pero con estabilidad numérica superior. El Conjunto B ya usa `rms`.

*Características excluidas y razón:*
- `mean`, `std`: reemplazados por versiones más robustas (`median`, `q25/q75`)
- `autocorr_lag_3` y `partial_autocorr`: se evalúan en Conjunto C
- FFT, entropía, características de memoria: se evalúan en conjuntos separados

---

**Conjunto C — +Memoria temporal alternativa**

*Composición:* B + `runs_ratio`, `hurst_rs`
*Dimensión:* 10 tipos × 16 sensores = **160 características**

*Dominio de información:* B + persistencia direccional + memoria larga

*Justificación de las características añadidas:*

**runs_ratio** — Estadístico de rachas de signos de la primera diferencia:
```
d_t = sign(x_t - x_{t-1})
runs_ratio = número_de_rachas(d_t) / (window_size - 1)
```
Mide la persistencia direccional de la señal. Alta autocorrelación positiva produce rachas largas (runs_ratio bajo); señales caóticas o ruidosas producen alternancia frecuente (runs_ratio alto). Captura patrones no lineales que la ACF lineal no detecta. Costo: O(n) — ~40 operaciones por sensor por ventana.

**hurst_rs** — Exponente de Hurst por rango reescalado simplificado:
```
R/S = (max(cumsum(x - mean(x))) - min(cumsum(x - mean(x)))) / std(x)
hurst_rs = log(R/S) / log(n)
```
Mide la memoria larga de la señal. H > 0.5 indica persistencia (tendencias que se mantienen); H < 0.5 indica antipersistencia. Complementa autocorr_lag_1/2 que solo captura dependencia de primer y segundo orden. Costo: O(n) — ~60 operaciones por sensor por ventana.

*Por qué no partial_autocorr:* Yule-Walker requiere resolver un sistema lineal de dimensión lag × lag por cada par (ventana, sensor) — O(n²) efectivo. Con 15,814 ventanas × 16 sensores × 3 lags, el costo es prohibitivo para el análisis bootstrap (200 iteraciones × 140 motores). Las alternativas propuestas capturan memoria temporal de forma diferente y con costo O(n).

---

**Conjunto D — +Dominio frecuencial**

*Composición:* B + `fft_coef_1`, `fft_coef_2`, `fft_coef_3`
*Dimensión:* 11 tipos × 16 sensores = **176 características**

*Dominio de información:* B + componentes frecuenciales

*Justificación de las características añadidas:*

**fft_coef_1/2/3** — Valor absoluto de los coeficientes de la FFT para frecuencias 1, 2 y 3 ciclos/ventana:
```
fft_coef_k = |FFT(x)[k]|   para k = 1, 2, 3
```
Captura la amplitud de las frecuencias dominantes dentro de la ventana. Alomari et al. (2023) identifican que el PC1 está fuertemente influenciado por coeficientes FFT de sensores de temperatura y presión. La degradación puede manifestarse como cambios en la distribución espectral de la señal.

*Nota:* fft_coef_0 fue excluido deliberadamente porque es proporcional a la media de la ventana — redundante con `median` ya presente en B.

*Por qué frecuencias 1, 2, 3 y no superiores:* Con window_size=20, las frecuencias superiores a 3-4 ciclos/ventana suelen estar dominadas por ruido de los sensores. Los coeficientes de baja frecuencia capturan variaciones suaves asociadas a la degradación gradual.

---

**Conjunto E — +Complejidad**

*Composición:* B + `permutation_entropy`
*Dimensión:* 9 tipos × 16 sensores = **144 características**

*Dominio de información:* B + desorden/impredictibilidad de la señal

*Justificación de las características añadidas:*

**permutation_entropy** — Entropía de permutación con tau=1, dimensión=3:
```
PE = -sum(p(π) * log2(p(π)))   para todos los patrones ordinales π de dimensión 3
```
Mide la complejidad o impredictibilidad de la serie temporal. Alomari et al. (2023) identifican que el PC2 está dominado casi exclusivamente por la entropía de permutación de sensores de velocidad y presión (Nf, Nc, phi, NRc, BPR). En motores sanos, la señal puede ser más regular; la degradación puede inducir mayor desorden o, paradójicamente, pérdida de complejidad por saturación.

*Costo computacional:* O(n) — construcción de patrones ordinales sobre ventana de 20 ciclos. Moderado pero justificado por la información cualitativa única que aporta.

---

### 3.2 Resumen de Conjuntos

| Conjunto | Características | n_tipos | n_features | Dominio añadido |
|----------|----------------|---------|------------|-----------------|
| A | mean, std, rms, slope, rvalue | 5 | 80 | Baseline mínimo |
| B | median, rms, q25, q75, slope, rvalue, autocorr×2 | 8 | 128 | Referencia actual |
| C | B + runs_ratio, hurst_rs | 10 | 160 | +Memoria alternativa |
| D | B + fft_coef_1/2/3 | 11 | 176 | +Frecuencia |
| E | B + permutation_entropy | 9 | 144 | +Complejidad |

---

## 4. Metodología: Enfoque Híbrido en Dos Fases

La selección del mejor conjunto de características se realiza mediante un procedimiento en dos fases. La Fase 1 es exploratoria y computacionalmente ligera; la Fase 2 (bootstrap + prueba de hipótesis) se aplica únicamente si la Fase 1 no produce un ganador claro. Este diseño es metodológicamente honesto y eficiente: evita la complejidad innecesaria cuando los datos hablan por sí solos, pero garantiza rigor estadístico cuando las diferencias son marginales.

---

### 4.1 Fase 1 — Evaluación Directa con Múltiples Tamaños de Ventana

**Objetivo:** Determinar si existe un conjunto dominante cuyo ranking sea estable e inequívoco a través de diferentes configuraciones de ventana.

**Procedimiento:**

```
Para cada window_size ∈ {15, 20, 25, 30}:

    Para cada conjunto (A, B, C, D, E):

        1. Cargar los 140 motores de entrenamiento completos
        2. Construir ventanas:
           motor_windows = build_windows(df_140_motores, window_size)
        3. Extraer características del conjunto X:
           motor_features = extract_window_features(motor_windows, features=CONJUNTO_X)
        4. Concatenar todas las ventanas globalmente:
           X_global = concatenar([motor_features[m]['X_windows'] para m en motores])
        5. Ajustar RobustScaler + PCA(n_components=15):
           scaler = RobustScaler().fit(X_global)
           X_scaled = scaler.transform(X_global)
           pca = PCA(n_components=15).fit(X_scaled)
        6. Registrar:
           cumvar[CONJUNTO_X][window_size] = pca.explained_variance_ratio_.cumsum()[-1]

Resultado: Matriz 5 × 4 de valores de cumvar
    filas:    conjuntos A, B, C, D, E
    columnas: window_size 15, 20, 25, 30
```

**Criterio de decisión de Fase 1:**

Se procede directamente a la selección (sin Fase 2) si se cumplen **ambas** condiciones:

1. **Dominancia clara:** El conjunto ganador supera al segundo mejor por un margen ≥ 2% de cumvar en al menos 3 de los 4 tamaños de ventana evaluados.

2. **Ranking estable:** El conjunto ganador ocupa la primera posición en los 4 tamaños de ventana evaluados (ranking invariante al window_size).

Si no se cumplen ambas condiciones, se pasa a la Fase 2.

**Reporte de Fase 1:**
- Tabla 5 × 4 de valores de cumvar (conjuntos × window_sizes)
- Gráfico de líneas: cumvar vs window_size por conjunto
- Identificación de conjuntos finalistas (aquellos con diferencias < 2% respecto al mejor)

---

### 4.2 Fase 2 — Bootstrap Pareado (condicional)

**Se aplica únicamente si:** el ranking de Fase 1 es inestable O las diferencias entre los mejores conjuntos son < 2%.

**Objetivo:** Cuantificar la incertidumbre muestral y confirmar que la diferencia observada entre conjuntos finalistas no es producto del azar.

**Justificación del bootstrap pareado sobre K-Fold:**

| Criterio | K-Fold repetido (K=5, R=10) | Bootstrap Pareado (N=200) |
|----------|-----------------------------|---------------------------|
| Observaciones | 50 | 200 |
| Repetición de motores | No | Sí (intra-muestra) |
| Diseño pareado | Parcial | Total — misma muestra para todos los conjuntos |
| Poder estadístico | Moderado | Alto |

El diseño **completamente pareado** del bootstrap es la ventaja decisiva: al evaluar todos los conjuntos sobre la misma muestra en cada iteración, se elimina la variabilidad de composición como factor de confusión. La correlación intra-muestra por repetición de motores no afecta la validez porque la dependencia es entre observaciones de la misma iteración — no entre iteraciones independientes.

**Procedimiento:**

```
Hiperparámetros: window_size del mejor resultado en Fase 1, n_components=15

Para iteración i = 1 hasta 200:

    Paso 1 — Muestra bootstrap:
        Remuestrear 140 motores CON reemplazo (remuestreo a nivel de motor)

    Paso 2 — Para CADA conjunto finalista sobre la MISMA muestra_i:
        a. build_windows → extract_features → concatenar → RobustScaler + PCA
        b. Registrar cumvar[CONJUNTO_X][i]

Resultado: vectores de 200 valores pareados por conjunto finalista
```

**Comparación estadística:**

- **Estadísticos descriptivos:** media ± desviación estándar, IC 95% (percentiles 2.5 y 97.5)
- **Prueba:** Wilcoxon signed-rank test pareado entre cada par de conjuntos finalistas
- **Corrección múltiple:** Bonferroni (α' = 0.05 / número de pares)
- **Criterio de selección:** conjunto con mayor cumvar medio Y diferencia estadísticamente significativa (p < α') respecto al Conjunto B (referencia)

---

### 4.3 Árbol de Decisión

```
Fase 1 — Evaluación directa (4 window_sizes × 5 conjuntos)
    ↓
¿Ranking estable Y diferencia ≥ 2%?
    ├── SÍ → Selección directa del conjunto ganador ✅
    │         Reportar tabla 5×4 de cumvar
    │         Justificar cualitativamente la elección
    │
    └── NO → Fase 2 — Bootstrap pareado (N=200)
                ↓
              Wilcoxon + Bonferroni entre finalistas
                ↓
              Selección del conjunto con mayor cumvar
              estadísticamente significativo ✅
```

---

### 4.4 Nota sobre el Criterio del 2%

El umbral de 2% para "diferencia clara" se establece por las siguientes razones:

- Alomari et al. (2023) reportan que 15 componentes explican ~50% de la varianza con su pipeline completo (TSFresh). Diferencias de 2% representan una variación relativa del 4% sobre ese valor — perceptible y prácticamente significativa.
- La literatura de PHM considera que diferencias de rendimiento menores al 2% entre métodos de preprocesamiento raramente se traducen en mejoras observables en las métricas de predicción final (RMSE, S-Score).
- Si dos conjuntos difieren en menos del 2%, la elección puede basarse en criterios secundarios: costo computacional, interpretabilidad, o simplicidad de implementación.



---

## 5. Nota sobre el Análisis de Tamaño de Ventana

El análisis de sensibilidad al tamaño de ventana {15, 20, 25, 30} está integrado en la **Fase 1** del procedimiento (Sección 4.1) — no es una extensión opcional sino parte central de la metodología.

Esta decisión responde a la recomendación conjunta de la literatura consultada:

> *"Si al variar el tamaño de ventana el ranking de conjuntos se mantiene idéntico, se evidencia una señal de degradación dominante robusta que no depende de la segmentación temporal."*

El análisis multi-ventana cumple dos funciones simultáneamente:
1. **Criterio de selección:** ranking estable → evidencia de dominancia real, no artefacto del window_size
2. **Información para el GGS:** identifica qué window_size maximiza el cumvar para el conjunto ganador, informando el rango de búsqueda del hiperparámetro en el Grid Search



---

## 6. Resultados Esperados

1. **Resultado primario:** Mejor conjunto de características (A, B, C, D o E) con justificación estadística
2. **Resultado secundario:** Sensibilidad del cumvar al tamaño de ventana (si el tiempo lo permite)
3. **Actualización del pipeline:** Reemplazar las características actuales en `feature_extraction.py` con el conjunto seleccionado
4. **Sección del paper:** "Metodología de Selección de Características" con resultados bootstrap y p-values Wilcoxon

---

## 7. Preguntas Abiertas para el Asesor

1. ¿Es el bootstrap pareado a nivel de motor una metodología aceptable para la selección de características en PHM? ¿Hay alguna preocupación sobre la repetición de motores dentro de cada muestra?

2. ¿Es la varianza explicada acumulada del PCA global un proxy suficiente para "información sobre la degradación del motor", o debería complementarse con una métrica secundaria (ej. separabilidad promedio entre motores en el espacio de PCs)?

3. ¿Deberíamos incluir un Conjunto F que combine las mejores adiciones individuales de C, D y E (runs_ratio + hurst_rs + fft + entropía) para evaluar si la combinación de todos los dominios es superior?

4. ¿Es window_size=20 el valor de referencia apropiado dado que Alomari usa max_shift=20 (no exactamente window_size=20)?
