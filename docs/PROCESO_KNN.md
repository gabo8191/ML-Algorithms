# 🤖 PROCESO K-NN – TITANIC (de extremo a extremo)

## 📋 Objetivo

Construir un clasificador K-Nearest Neighbors para predecir `Survival` en el Titanic y documentar con precisión operativa cada paso ejecutado: carga, validación, selección de variables, imputación, codificación, partición, escalado, optimización de K, entrenamiento, evaluación, generación de métricas y visualizaciones.

## 🔄 Pipeline implementado (paso a paso exacto)

1. Carga de datos

- Implementación: `src/data_processing/data_loader.py` → `DataLoader.load_titanic_data()` (redirige a `load_csv`).
- Parámetros de lectura: `low_memory=False`, `encoding='utf-8'` y dtypes específicos:
  - `PassengerId:int32`, `Survived:int8`, `Pclass:int8`, `Age:float32`, `SibSp:int8`, `Parch:int8`, `Fare:float32`.
- Operaciones de logging: forma del dataset, tamaño de archivo, uso de memoria, listado de columnas y tipos de datos.

2. Validación de datos

- Implementación: `DataLoader.validate_data(df)`.
- Salidas del validador:
  - Estructura: `shape`, columnas, `dtypes`, memoria (MB), filas duplicadas.
  - Calidad: conteo y porcentaje de nulos por columna.
  - Perfil numérico por columna numérica: media, desviación, mínimo, máximo, conteo de ceros y de negativos.
  - Perfil categórico: cardinalidad, moda y frecuencia de la moda.

3. Selección y transformación inicial de variables

- Implementación: `src/data_processing/data_preprocessor.py` → `DataPreprocessor.select_features_for_classification(df)`.
- Operaciones exactas:
  - Eliminación de columnas no utilizadas: `PassengerId`, `Name`, `Ticket`, `Cabin`.
  - Imputación puntual:
    - `Age` ← mediana de `Age`.
    - `Embarked` ← moda de `Embarked`.
  - Codificación con `pd.get_dummies(..., drop_first=True, dtype=int)`:
    - `Sex` → `Sex_male` (1 = hombre, 0 = mujer).
    - `Embarked` → `Embarked_Q`, `Embarked_S` (cuando ambos son 0, corresponde a Cherbourg).
  - Verificación de `Survived` y retorno de un DataFrame con variables finales + objetivo.

4. Tratamiento formal de valores faltantes

- Implementación: `DataPreprocessor.handle_missing_values(df, strategy='median')`.
- Lógica aplicada:
  - Separación de columnas numéricas y categóricas.
  - Numéricas → `SimpleImputer(strategy='median')` con persistencia en `self.imputers['numeric']`.
  - Categóricas → `SimpleImputer(strategy='most_frequent')` con persistencia en `self.imputers['categorical']`.
  - Si `Survived` tiene nulos, se eliminan esas filas y se registra la cantidad removida.

5. (Definido, no activo por defecto) Remoción de outliers

- Implementación disponible: `DataPreprocessor.remove_outliers(df, method='iqr', threshold=1.5)`.
- Método IQR por columna numérica; preserva `Survived` y reporta filas removidas. En el pipeline por defecto, esta operación está comentada y no se ejecuta.

6. Preparación de características (X) y objetivo (y)

- Implementación: `DataPreprocessor.prepare_features_target(df, target_column='Survived')`.
- Detalles de tipado:
  - `X = df.drop('Survived').astype('float64')`.
  - `y = df['Survived'].astype('int64')` (si fuese `object`, se usa `LabelEncoder`).
- Persistencia de metadatos: `feature_names` y `target_name` se almacenan para trazabilidad.

7. División estratificada en entrenamiento y prueba

- Implementación: `DataPreprocessor.split_data(X, y)`.
- Parámetros efectivos: `test_size` y `random_state` desde `Config` (por defecto 0.2 y semilla fija). Se usa `stratify=y`.
- Fallback: si la estratificación falla, se realiza división sin `stratify` (con advertencia). Se registran las formas de `X_train`, `X_test`, `y_train`, `y_test`.

8. Escalado de características

- Implementación: `DataPreprocessor.scale_features(X_train, X_test)`.
- Algoritmo: `StandardScaler` (fit sobre train; transform sobre train y test). Devuelve `np.ndarray`. El scaler se guarda en `self.scaler`.

9. Optimización del hiperparámetro K

- Implementación: `src/models/knn_classifier.py` → `KNNClassifier.find_optimal_k(X_train_scaled, y_train)`.
- Proceso: para cada K en `Config.K_RANGE`, se ejecuta `cross_val_score(knn, X_train, y_train, cv=Config.CV_FOLDS, scoring='accuracy', n_jobs=-1)`.
- Resultados almacenados en `self.cv_results`: lista de K evaluados, medias y desviaciones estándar por K, `best_k` y `best_score`.
- En la evaluación actual, `best_k = 12`.

10. Entrenamiento del modelo

- Implementación: `KNNClassifier.train(X_train_scaled, y_train, optimize_k=True)`.
- Parámetros del estimador: `n_neighbors=12`, `metric='euclidean'`, `weights='uniform'` (a partir de `Config.get_model_config()` con ajuste de `n_neighbors`).
- Métrica de entrenamiento registrada: `train_accuracy` a partir de predicción sobre `X_train_scaled`.

11. Evaluación del modelo

- Ruta directa: `KNNClassifier.evaluate(X_test_scaled, y_test, class_names)` produce `accuracy`, `classification_report` (como dict) y `confusion_matrix`.
- Ruta integral: `src/evaluation/model_evaluator.py` → `ModelEvaluator.evaluate_model(...)` añade:
  - Métricas básicas (train/test) y avanzadas (Cohen’s kappa, MCC) con `overfitting_score = train_accuracy - test_accuracy`.
  - Métricas por clase (precision, recall, f1, support) y análisis de matriz de confusión (absoluta y normalizada por fila/columna/total; distribución de clases reales y predichas; correctos/incorrectos; tasa de error).
  - Validación cruzada (media, desviación estándar, min, max, lista de scores por fold).
  - Importancia de características por permutación normalizada (si se proporcionan `feature_names`).
- Serialización de resultados:
  - `ModelEvaluator.save_evaluation_report(filepath)` → `results/evaluation_report.json`.
  - `ModelEvaluator.export_results_to_csv(filepath)` → `results/metrics_summary.csv`.

12. Visualizaciones y guardado de imágenes

- Implementación: `ModelEvaluator.generate_visualizations(save_dir='results/model_visualizations', show_plots=False)`.
- Genera: `KNN_Titanic_Classifier_confusion_matrix.png` (matriz de confusión con anotaciones e interpretación estándar en el encabezado).
- La optimización de K puede graficarse desde el clasificador (`KNNClassifier.plot_k_optimization`) y, en el visualizador, con `plot_k_optimization(k_values, cv_scores_mean, cv_scores_std, best_k)`.

## ⚙️ Configuración final del KNN

- Tipo de modelo: `KNeighborsClassifier`.
- Hiperparámetros efectivos: `n_neighbors=12`, `metric='euclidean'`, `weights='uniform'`.
- Preprocesamiento requerido: estandarización con `StandardScaler` (fit en train, transform en test de la misma manera).

## 📈 Resultados numéricos (según evaluación)

- Entrenamiento: Accuracy 0.8287.
- Prueba: Accuracy 0.8212; Balanced Accuracy 0.7870; Precision 0.8271; Recall 0.8212; F1 0.8146.
- Métricas avanzadas: Cohen’s Kappa 0.6034; MCC 0.6190.
- Overfitting score: 0.0074 (diferencia absoluta train–test).
- Validación cruzada (accuracy): media ≈ 0.8203; std ≈ 0.0329; min ≈ 0.7762; max ≈ 0.8732.

## 🧩 Matriz de confusión (desglose exacto)

- Matriz absoluta (test, 179 muestras):

```text
                 Predicción
             No Sobrevivió   Sobrevivió
Real
No Sobrevivió        103            7
Sobrevivió            25           44
```

- TN = 103
- FP = 7
- FN = 25
- TP = 44

- Normalizada por fila (recall por clase):

```text
No Sobrevivió: [0.9364, 0.0636]
Sobrevivió:   [0.3623, 0.6377]
```

- Normalizada por columna (precision por predicción):

```text
Pred «No Sobrevivió»: [0.8047, 0.1953]
Pred «Sobrevivió»:    [0.1373, 0.8627]
```

- Distribuciones: reales [110, 69]; predichas [128, 51].

- Lectura operativa: alta especificidad para «No Sobrevivió» (recall 0.936) y alta precisión al predecir «Sobrevivió» (0.863). La mayor masa de error son FN (25), es decir, sobrevivientes clasificados como no sobrevivientes.

## 🌟 Importancia de características (permutación)

- Importancias normalizadas (suma ≈ 1.0):

```text
Sex_male:   0.4983
SibSp:      0.1271
Pclass:     0.1187
Age:        0.1070
Parch:      0.0769
Fare:       0.0452
Embarked_S: 0.0151
Embarked_Q: 0.0117
```

- Lectura: `Sex_male` concentra la mayor contribución marginal; le siguen vínculos familiares (`SibSp`, `Parch`), clase del boleto (`Pclass`) y edad (`Age`).

## 🔗 Matriz de correlación

- Propósito: cuantificar relaciones lineales entre variables numéricas utilizadas tras el preprocesamiento.
- Uso: identificar redundancias que afecten el cálculo de distancias en KNN y verificar coherencia de escalas tras estandarización.
- Patrón esperable en Titanic: relación entre `Pclass` y `Fare`, y asociación entre `SibSp` y `Parch`.

## 🔧 Optimización de K

- ¿Qué es K y por qué es importante?:

  - K es el número de vecinos más cercanos considerados para clasificar una instancia. Determina el tamaño del vecindario en el espacio de características.
  - Impacto directo en el sesgo–varianza: K bajos (1–5) capturan ruido (sobreajuste); K altos hacen predicciones muy suavizadas (subajuste).
  - En KNN, K es el hiperparámetro crítico porque no existe un ajuste paramétrico interno: toda la flexibilidad del modelo está en la elección del vecindario y la métrica.

- ¿Cómo se calcula y selecciona K en este proyecto?:

  - Para cada valor de K en `Config.K_RANGE`, se entrena un KNN temporal con `n_neighbors=K` y se calcula su desempeño con `cross_val_score` (CV de `Config.CV_FOLDS`).
  - Se registran para cada K: el accuracy medio y su desviación estándar. Estos valores quedan guardados en `cv_results` (llaves: `k_values`, `cv_scores_mean`, `cv_scores_std`, `best_k`, `best_score`).
  - Se elige `best_k` como el K con mayor accuracy medio. En esta corrida, `best_k = 12`.

- Procedimiento: evaluación por validación cruzada para múltiples K, recopilando media y desviación estándar; selección del mejor K (= 12) y registro en `cv_results` para reproducibilidad y graficación.

## 🗂️ Artefactos generados

- `results/evaluation_report.json`
- `results/metrics_summary.csv`
- `results/model_visualizations/KNN_Titanic_Classifier_confusion_matrix.png`
- `results/model_visualizations/KNN_Titanic_Classifier_feature_importance.png`
- `results/correlation_matrix.png`
- `results/k_optimization.png`
