# 📊 ANÁLISIS DETALLADO DE RESULTADOS – KNN TITANIC (Supervivencia)

## 🎯 Objetivo del modelo

- **Tarea**: Clasificación binaria de la etiqueta `Survival` (0 = No Sobrevivió, 1 = Sobrevivió).
- **Meta de entrenamiento**: Aprender un clasificador K-Nearest Neighbors que, dado un conjunto de variables demográficas y de viaje, estime la probabilidad de supervivencia de un pasajero del Titanic.
- **Alcance**: Uso de variables estructuradas (clase de boleto, edad, tarifa, composición familiar, sexo y puerto de embarque) con preprocesamiento consistente y reproducible.

## 🧪 Conjunto de datos, variables y partición

- **Partición**: 712 muestras para entrenamiento (80%) | 179 para prueba (20%) | **Clases**: 2.
- **Relación porcentual y entera**:
  - Entrenamiento: 712/891 ≈ 79.91% (redondeado 80%).
  - Prueba: 179/891 ≈ 20.09% (redondeado 20%).
- **Estrategia de partición**: estratificada por la etiqueta `Survived` para conservar la proporción de clases en ambos conjuntos.
- **Características finales (8)**:
  - `Pclass` (entero): Clase del boleto (1=1ª, 2=2ª, 3=3ª). Variable ordinal asociada a condiciones de camarote y acceso.
  - `Age` (numérico): Edad en años (imputada con mediana cuando falta). Factor demográfico clave.
  - `SibSp` (entero): Número de hermanos/esposos a bordo. Indica estructura inmediata de acompañantes.
  - `Parch` (entero): Número de padres/hijos a bordo. Complementa la estructura familiar.
  - `Fare` (numérico): Tarifa pagada por el pasajero. Proxy de clase socioeconómica.
  - `Sex_male` (binaria): 1 si es hombre, 0 si es mujer. Codificación de `Sex` con `get_dummies(drop_first=True)`.
  - `Embarked_Q` (binaria): 1 si embarcó en Queenstown, 0 en otro puerto (codificación de `Embarked`).
  - `Embarked_S` (binaria): 1 si embarcó en Southampton, 0 en otro puerto. Cuando `Embarked_Q=0` y `Embarked_S=0` implica Cherbourg.
- **Preprocesamiento clave**: imputación de `Age` (mediana) y `Embarked` (moda), codificación con `get_dummies(drop_first=True)`, estandarización con `StandardScaler` antes de KNN.

## 📈 Métricas principales (test)

- **Accuracy**: 82.12%
- **Balanced Accuracy**: 78.70%
- **Precision (ponderada)**: 82.71%
- **Recall (ponderado)**: 82.12%
- **F1-Score (ponderado)**: 81.46%
- **Cohen’s Kappa**: 0.603 (acuerdo sustancial)
- **Matthews Corrcoef**: 0.619 (correlación fuerte-moderada)
- **Overfitting score**: 0.0074 (excelente generalización; gap train-test ≈ 0.74%)
- **Validación cruzada (accuracy)**: media ≈ 0.8203; desviación estándar ≈ 0.0329; min ≈ 0.7762; max ≈ 0.8732 (5-fold sobre entrenamiento).

Fuentes: `results/evaluation_report.json`, `results/metrics_summary.csv`.

## 🧩 Matriz de confusión: valores, normalizaciones e interpretación

Matriz absoluta (test, 179 muestras):

```text
                 Predicción
             No Sobrevivió   Sobrevivió
Real
No Sobrevivió        103            7
Sobrevivió            25           44
```

- Tomando «Sobrevivió» como clase positiva:
  - **TP (True Positive)** = 44: pasajero sobrevivió y el modelo predijo «Sobrevivió».
  - **TN (True Negative)** = 103: pasajero no sobrevivió y el modelo predijo «No Sobrevivió».
  - **FP (False Positive)** = 7: pasajero no sobrevivió pero el modelo predijo «Sobrevivió».
  - **FN (False Negative)** = 25: pasajero sobrevivió pero el modelo predijo «No Sobrevivió».

Normalizada por fila (recall por clase):

```text
No Sobrevivió: [0.9364, 0.0636]
Sobrevivió:   [0.3623, 0.6377]
```

Normalizada por columna (precision por predicción):

```text
Pred «No Sobrevivió»: [0.8047, 0.1953]
Pred «Sobrevivió»:    [0.1373, 0.8627]
```

Distribuciones de conteo:

- Reales: [110, 69]
- Predichas: [128, 51]

Lectura operativa del patrón:

- Alta especificidad/recall para «No Sobrevivió» (0.936) y alta precisión al predecir «Sobrevivió» (0.863).
- La mayor masa de error son **FN=25** dentro de «Sobrevivió», es decir, sobrevivientes clasificados como no sobrevivientes.
- El sesgo de predicción hacia «No Sobrevivió» (128 vs 51) concuerda con la diagonal y con el recall más alto de la clase negativa.

## 🌟 Importancia de características (permutación)

Importancias relativas normalizadas (suma ≈ 1.0) derivadas de la caída promedio de accuracy al permutar cada variable:

```text
Sex_male:     0.4983
SibSp:        0.1271
Pclass:       0.1187
Age:          0.1070
Parch:        0.0769
Fare:         0.0452
Embarked_S:   0.0151
Embarked_Q:   0.0117
```

Interpretación focalizada:

- **Sex_male** domina la contribución marginal del modelo, reflejando la diferencia histórica en tasas de supervivencia por sexo.
- **SibSp** y **Parch** capturan estructura familiar; **Pclass** y **Age** modulan la probabilidad de supervivencia en el contexto del evento.
- **Fare** añade señal socioeconómica; `Embarked_Q` y `Embarked_S` aportan peso menor en este conjunto.

## 🔗 Matriz de correlación (`correlation_matrix.png`)

- Propósito: visualizar relaciones lineales entre variables numéricas utilizadas una vez aplicado el preprocesamiento.
- Uso en KNN: identificar redundancias que afecten distancias euclidianas y confirmar que, tras el escalado, ninguna variable domina por magnitud.
- Patrón esperable en Titanic: relación `Pclass`–`Fare` y asociación entre `SibSp` y `Parch`.

## 🔧 Optimización de K (`k_optimization.png`)

- Procedimiento: evaluación de múltiples valores de K mediante validación cruzada 5-fold sobre las características ya escaladas; se registran media y desviación estándar del accuracy para cada K.
- Resultado: selección del K con mayor media (en este caso, **K=12**), registrado en los artefactos del modelo.
- Forma de la curva: K bajos sobreajustan con variabilidad; rango medio muestra estabilidad; K altos pierden patrones locales.

## 📚 Definición de métricas y su lectura en este contexto

- **Accuracy**: proporción de aciertos totales (82.12%).
- **Balanced Accuracy**: promedio del recall por clase (78.70%); corrige posibles desbalances de clase.
- **Precision (ponderada)**: promedio ponderado por soporte; en «Sobrevivió» es 0.863, lo que indica pocas falsas alarmas al predecir positivos.
- **Recall (ponderado)**: en «Sobrevivió» es 0.638; indica cobertura parcial de positivos verdaderos.
- **F1 (ponderado)**: balance entre precision y recall (81.46%).
- **Cohen’s Kappa (0.603)** y **MCC (0.619)**: acuerdo y correlación ajustados por el azar, consistentes con un clasificador informativo y estable.
- **Overfitting score (0.0074)**: diferencia absoluta train–test, indica generalización consistente.

## 🧠 Interpretación global y conclusiones finales

- Con un esquema de 80/20 estratificado (712/179), el modelo muestra **desempeño estable**: accuracy de prueba 82.12% y validación cruzada con media ≈ 0.820 y dispersión moderada.
- La **matriz de confusión** revela alta capacidad para identificar no sobrevivientes y alta precisión al etiquetar sobrevivientes; el principal error son falsos negativos en «Sobrevivió».
- La **importancia de características** sitúa `Sex_male` como señal predominante, seguida por estructura familiar, clase y edad, lo que concuerda con el contexto del naufragio.
- Dado el **objetivo** de predecir `Survival` binario con KNN sobre variables estandarizadas, los resultados obtenidos son coherentes y alineados con la evidencia del conjunto: el modelo captura los patrones principales relevantes para la supervivencia con generalización adecuada.

## 🗂️ Archivos relevantes

- `results/evaluation_report.json` (métricas, matriz de confusión, CV).
- `results/metrics_summary.csv` (resumen ejecutivo).
- `results/model_visualizations/KNN_Titanic_Classifier_confusion_matrix.png` (confusión).
- `results/model_visualizations/KNN_Titanic_Classifier_feature_importance.png` (importancia).
- `results/correlation_matrix.png` (correlaciones).
- `results/k_optimization.png` (optimización de K).

**Fecha de análisis**: 2025-09-09 · **Modelo**: `KNN_Titanic_Classifier` (K=12, euclidiana, weights=uniform)
