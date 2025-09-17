# 🤖 ML-Algorithms – Análisis Completo de Machine Learning

## 📋 Descripción breve

Sistema completo de análisis de Machine Learning para predecir el éxito de cafeterías usando múltiples algoritmos. Implementa los 6 pasos principales del desarrollo de modelos ML y compara el rendimiento de diferentes algoritmos automáticamente.

## 🎯 Objetivo

Predecir si una cafetería es "exitosa" (Daily_Revenue ≥ $2,000) o "no exitosa" (Daily_Revenue < $2,000) basándose en sus métricas operacionales (clientes, ingresos, empleados, marketing, etc.) comparando 5 algoritmos diferentes de ML:

- **Regresión Logística**
- **Máquinas de Vector de Soporte (SVM)**
- **Árboles de Decisión**
- **Random Forest**
- **Redes Neuronales Artificiales (MLP)**

## 🚀 Levantamiento rápido

### 1. Prerrequisitos

- Python 3.8+
- pip actualizado

### 2. Instalación

```bash
cd ML-Algorithms
python -m venv ml_env

# Activar entorno virtual
# Linux/Mac
source ml_env/bin/activate
# Windows (PowerShell)
ml_env\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ejecución

**Análisis completo (todos los algoritmos):**

```bash
python ml_analysis.py
```

**Análisis rápido (sin optimización de hiperparámetros):**

```bash
python ml_analysis.py --quick-mode
```

**Algoritmos específicos:**

```bash
python ml_analysis.py --algorithms LogisticRegression SVM DecisionTree RandomForest NeuralNetwork
```

**Opciones adicionales:**

```bash
python ml_analysis.py --help
```

Tiempo estimado: 5–20 minutos (según configuración y algoritmos seleccionados).

## 📂 Resultados al ejecutar

### Estructura de resultados en `results/`

```
results/
├── algorithm_comparison_report.json     # Comparación completa de todos los algoritmos
├── correlation_matrix.png               # Análisis de correlaciones del dataset
├── comparisons/                         # Visualizaciones comparativas
│   ├── metrics_comparison.png          # Comparación de métricas por algoritmo
│   ├── rankings_heatmap.png            # Heatmap de rankings
│   ├── radar_comparison.png            # Gráfico radar multidimensional
│   └── time_vs_accuracy.png            # Tiempo vs precisión
├── logisticregression/                  # Resultados de Regresión Logística
│   ├── logisticregression_model.pkl
│   ├── coefficients.png                # Visualización de coeficientes
│   └── ...
├── svm/                                 # Resultados de SVM
│   ├── svm_model.pkl
│   ├── decision_boundary.png           # Frontera de decisión (2D)
│   └── ...
├── decisiontree/                        # Resultados de Árbol de Decisión
│   ├── decisiontree_model.pkl
│   ├── tree_visualization.png          # Visualización del árbol
│   ├── feature_importance.png
│   └── ...
├── randomforest/                        # Resultados de Random Forest
│   ├── randomforest_model.pkl
│   ├── feature_importance.png
│   ├── trees_depth_distribution.png
│   └── ...
└── neuralnetwork/                       # Resultados de Red Neuronal
    ├── neuralnetwork_model.pkl
    ├── training_curves.png             # Curvas de pérdida y validación
    ├── network_architecture.png        # Visualización de la arquitectura
    └── ...

## 📚 Documentación

- Análisis e interpretación de resultados: `docs/ANALISIS_RESULTADOS.md`

## 🧭 Estructura mínima del proyecto

```

ML-Algorithms/
├── src/
├── data/
├── results/
└── docs/

```

## 📝 Notas

- Ejecuta `python ml_analysis.py --help` para ver opciones de ejecución (algoritmos, modo rápido, rutas).
```
