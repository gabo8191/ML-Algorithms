### Vista General del Pipeline de Machine Learning (`ml_analysis.py`)

## Descripción General

Este documento proporciona una guía completa y detallada del pipeline de Machine Learning implementado en `ml_analysis.py` para predecir el éxito de cafeterías. El sistema utiliza 5 algoritmos diferentes para clasificar cafeterías como "exitosas" o "no exitosas" basándose en 6 características operativas clave.

## Arquitectura del Sistema

### Estructura Modular

El pipeline está organizado en módulos especializados:

- **`data_processing/`**: Carga y preprocesamiento de datos
- **`models/`**: Implementación de algoritmos ML
- **`evaluation/`**: Evaluación y comparación de modelos
- **`visualization/`**: Generación de gráficos y reportes
- **`utils/`**: Configuración y logging

### Flujo de Datos

```
CSV → DataLoader → DataPreprocessor → Modelos ML → Evaluación → Visualizaciones → Reportes
```

## Configuración y Argumentos

### Parámetros de Entrada

- **`--data-path`**: Ruta al archivo CSV (default: `data/coffee_shop_revenue.csv`)
- **`--sample-size`**: Tamaño de muestra opcional para pruebas rápidas
- **`--output-dir`**: Directorio de salida (default: `results`)

- **`--algorithms`**: Lista específica de algoritmos a ejecutar
- **`--quick-mode`**: Desactiva optimización de hiperparámetros para ejecución rápida

### Configuración por Defecto

- **División de datos**: 80% entrenamiento, 20% prueba
- **Validación cruzada**: 5 folds estratificados
- **Semilla aleatoria**: 42 (reproducibilidad)
- **Métrica objetivo**: Accuracy
- **Algoritmos**: 5 (LogisticRegression, SVM, DecisionTree, RandomForest, NeuralNetwork)

## Proceso Detallado Paso a Paso

### 1. Recopilación y Validación de Datos

#### 1.1 Carga del Dataset

- **Archivo**: `coffee_shop_revenue.csv` con 2,000 registros de cafeterías
- **Columnas originales**: 7 variables (6 características + 1 objetivo)
- **Validaciones**:

  - Verificación de estructura del archivo
  - Detección de valores faltantes
  - Validación de tipos de datos
  - Cálculo de estadísticas descriptivas

#### 1.2 Información del Dataset

- **Tamaño**: 2,000 filas × 7 columnas

- **Memoria**: ~112 KB
- **Tipos**: 6 numéricas, 1 objetivo
- **Valores faltantes**: Verificación y reporte

### 2. Definición del Objetivo de Clasificación

#### 2.1 Transformación de Variable Continua a Binaria

El dataset original contiene `Daily_Revenue` (variable continua). Para crear un problema de clasificación binaria:

**Proceso**:

1. Establecer un umbral fijo de $2,000 en `Daily_Revenue`
2. Crear variable `Successful`:

   - `1` si `Daily_Revenue ≥ $2,000` → "Exitosa"
   - `0` si `Daily_Revenue < $2,000` → "No Exitosa"

**Ejemplo práctico**:

- Umbral fijo: $2,000
  - Cafeterías con ingresos ≥ $2,000 → "Exitosa"
  - Cafeterías con ingresos < $2,000 → "No Exitosa"

#### 2.2 Justificación de la Definición

- **Objetiva**: Basada en un umbral fijo claro y definido
- **Reproducible**: Mismo criterio aplicable a nuevos datos
- **Simple**: Fácil de entender y aplicar en el negocio
- **Clara**: Criterio de negocio directo y comprensible

### 3. Protocolo de Evaluación

#### 3.1 División de Datos

- **Método**: `train_test_split` con estratificación
- **Proporción**: 80% entrenamiento (1,600 muestras), 20% prueba (400 muestras)
  - **Train**: 952 No Exitosas + 648 Exitosas
  - **Test**: 238 No Exitosas + 162 Exitosas
- **Estratificación**: Mantiene proporción de clases en ambos conjuntos
- **Semilla**: 42 para reproducibilidad

#### 3.1.1 Justificación del balanceo de clases

- Tras crear `Successful` con el umbral fijo de $2,000 en `Daily_Revenue`, el dataset se divide 80/20 con estratificación para conservar proporciones por clase en train y test.
- La verificación del balanceo se realiza calculando conteos de clases con `np.bincount` y la proporción de la clase positiva:
  - Test set: 238 "No Exitosas" (59.5%) y 162 "Exitosas" (40.5%)
  - Train y test mantienen la misma proporción gracias a la estratificación
- Conclusión: el dataset está moderadamente balanceado (59.5% vs 40.5%), lo que es favorable para el entrenamiento de modelos de ML.

#### 3.2 Validación Cruzada

- **Método**: K-Fold estratificado (5 folds)

- **Propósito**: Evaluación robusta del rendimiento

- **Métricas**: Accuracy, F1-Score, Precision, Recall
- **Reporte**: Media ± desviación estándar

#### 3.3 Optimización de Hiperparámetros

- **Método**: Grid Search (desactivado en quick-mode)
- **Algoritmos**: Todos excepto en modo rápido
- **Métrica**: Accuracy (configurable)

- **Validación**: Cross-validation interna

### 4. Preprocesamiento de Datos

#### 4.1 Selección de Características

**Variables seleccionadas** (6 características numéricas):

1. `Number_of_Customers_Per_Day` - Número de clientes diarios
2. `Average_Order_Value` - Valor promedio por orden
3. `Operating_Hours_Per_Day` - Horas de operación diarias
4. `Number_of_Employees` - Número de empleados
5. `Marketing_Spend_Per_Day` - Gasto en marketing diario
6. `Location_Foot_Traffic` - Tráfico peatonal de la ubicación

#### 4.2 Procesamiento de Datos

- **Imputación**: Valores faltantes con mediana (numéricas)
- **Escalado**: StandardScaler para normalizar características
- **Codificación**: No necesario (todas las variables son numéricas)
- **Outliers**: Detección y manejo con método IQR

#### 4.3 Generación de Visualizaciones

- **Matriz de correlación**: Relaciones entre variables originales
- **Distribución de clases**: Balance del dataset
- **Estadísticas descriptivas**: Resumen de cada variable

### 5. Modelado y Entrenamiento

#### 5.1 Algoritmos Implementados

##### Logistic Regression

- **Tipo**: Modelo lineal generalizado
- **Ventajas**: Rápido, interpretable, buen baseline
- **Hiperparámetros**: penalty='l1', solver='liblinear', C=0.1
- **Uso**: Baseline y comparación

##### Support Vector Machine (SVM)

- **Tipo**: Clasificador no lineal con kernel RBF
- **Ventajas**: Excelente separación de clases, robusto
- **Hiperparámetros**: kernel='rbf', C=10.0, gamma=0.01
- **Uso**: Clasificación de alta precisión

##### Decision Tree

- **Tipo**: Árbol de decisión individual
- **Ventajas**: Interpretable, no requiere escalado
- **Hiperparámetros**: criterion='entropy', max_depth=10
- **Uso**: Interpretabilidad y baseline

##### Random Forest

- **Tipo**: Ensemble de árboles de decisión
- **Ventajas**: Robusto, maneja overfitting, feature importance
- **Hiperparámetros**: n_estimators=100, max_depth=10
- **Uso**: Balance entre rendimiento e interpretabilidad

##### Neural Network (MLP)

- **Tipo**: Red neuronal multicapa
- **Ventajas**: Captura no linealidades complejas

- **Hiperparámetros**: hidden_layers=(100,50), activation='tanh'
- **Uso**: Modelado complejo

#### 5.2 Proceso de Entrenamiento

1. **Inicialización**: Crear instancia del modelo con hiperparámetros

2. **Entrenamiento**: `fit(X_train, y_train)` en datos escalados
3. **Predicción**: `predict(X_test)` para clases
4. **Probabilidades**: `predict_proba(X_test)` si está disponible
5. **Validación**: Cross-validation para evaluación robusta

### 6. Evaluación de Modelos

#### 6.1 Métricas Calculadas

- **Accuracy**: Precisión general del modelo
- **Precision**: Proporción de predicciones positivas correctas
- **Recall**: Proporción de casos positivos detectados
- **F1-Score**: Media armónica de precision y recall
- **AUC-ROC**: Área bajo la curva ROC
- **Average Precision**: Rendimiento en clase minoritaria

- **Cohen's Kappa**: Acuerdo más allá del azar
- **Matthews Correlation Coefficient**: Correlación entre predicciones y realidad

#### 6.2 Análisis de Errores

- **Matriz de confusión**: Verdaderos/falsos positivos/negativos
- **Análisis de errores**: Casos mal clasificados
- **Overfitting score**: Diferencia entre train y test accuracy

- **Estabilidad**: Consistencia en validación cruzada

#### 6.3 Visualizaciones Generadas

- **Confusion Matrix**: Errores de clasificación
- **Feature Importance**: Variables más relevantes

- **ROC Curve**: Rendimiento en diferentes umbrales
- **Precision-Recall Curve**: Rendimiento en clase minoritaria
- **Learning Curves**: Evolución del rendimiento

### 7. Comparación y Ranking

#### 7.1 Sistema de Ranking

- **Métricas consideradas**: Accuracy, AUC-ROC, Average Precision
- **Método**: Ranking por cada métrica, luego promedio
- **Manejo de NaN**: Valores faltantes van al final del ranking
- **Resultado**: Overall rank de 1 a 5

#### 7.2 Visualizaciones Comparativas

- **Gráfico de barras**: Métricas por algoritmo
- **Heatmap de rankings**: Posición en cada métrica
- **Gráfico radar**: Rendimiento multidimensional
- **Scatter plot**: Tiempo vs accuracy

### 8. Generación de Reportes

#### 8.1 Archivos de Salida

- **JSON**: `algorithm_comparison_report.json` - Reporte completo
- **CSV**: `algorithm_comparison_metrics.csv` - Métricas tabuladas
- **Modelos**: `*.pkl` - Modelos entrenados guardados
- **Visualizaciones**: PNG en carpetas por algoritmo

#### 8.2 Estructura de Resultados

```
results/
├── algorithm_comparison_report.json
├── algorithm_comparison_metrics.csv
├── correlation_matrix.png
├── comparisons/
│   ├── metrics_comparison.png
│   ├── rankings_heatmap.png
│   ├── radar_comparison.png

│   └── time_vs_accuracy.png
├── logisticregression/
│   ├── *.pkl, *.csv, *.json
│   └── visualizations/
├── svm/
├── decisiontree/

├── randomforest/
└── neuralnetwork/
```

## Ejecución del Pipeline

### Comandos Básicos

```bash

# Ejecución completa
python ml_analysis.py

# Modo rápido (sin optimización de hiperparámetros)
python ml_analysis.py --quick-mode


# Algoritmos específicos
python ml_analysis.py --algorithms LogisticRegression SVM

# Dataset personalizado
python ml_analysis.py --data-path mi_dataset.csv


# Directorio de salida personalizado
python ml_analysis.py --output-dir mis_resultados
```

### Requisitos del Sistema

- **Python**: 3.8+
- **Memoria**: Mínimo 4GB RAM
- **Tiempo**: 5-15 minutos (dependiendo del modo)
- **Espacio**: ~50MB para resultados

### Interpretación de Resultados

1. **Revisar ranking**: Algoritmo con mejor overall rank
2. **Analizar métricas**: Accuracy, AUC, AP para cada modelo
3. **Examinar visualizaciones**: Gráficos de comparación
4. **Revisar feature importance**: Variables más relevantes
5. **Considerar overfitting**: Diferencia train vs test

## Consideraciones Técnicas

### Limitaciones

- **Dataset**: Moderadamente balanceado (59.5% No Exitosa, 40.5% Exitosa)
- **Variables limitadas**: Solo 6 características numéricas
- **Tamaño de muestra**: 2,000 registros (moderado)
- **Overfitting**: Posible en modelos complejos

### Mejoras Futuras

- **Balanceo de clases**: SMOTE, undersampling
- **Más características**: Variables categóricas, derivadas
- **Ensemble methods**: Voting, stacking
- **Optimización avanzada**: Bayesian optimization

### Aplicaciones Prácticas

- **Selección de ubicaciones**: Evaluar viabilidad de nuevas cafeterías
- **Análisis de factores**: Identificar variables críticas para el éxito
- **Predicción de rendimiento**: Estimar probabilidad de éxito
- **Optimización operativa**: Enfocar recursos en factores clave
