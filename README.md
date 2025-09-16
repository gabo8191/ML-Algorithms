# 🤖 K-NN Titanic – Guía de Levantamiento del Proyecto

## 📋 Descripción breve

Clasificador K-Nearest Neighbors (KNN) para predecir la supervivencia de pasajeros del Titanic. Al ejecutarlo, se entrena el modelo, se evalúa y se generan métricas y visualizaciones en la carpeta `results/`.

## 🎯 Objetivo

Entrenar y evaluar un modelo KNN para la etiqueta binaria `Survival` (0 = No Sobrevivió, 1 = Sobrevivió) usando variables demográficas y de viaje, con preprocesamiento reproducible y reportes automatizados.

## 🚀 Levantamiento rápido

1. Prerrequisitos

- Python 3.8+
- pip actualizado

2. instalar

```bash
cd knn-clasification
python -m venv knn_env

# Activar entorno virtual
# Linux/Mac
source knn_env/bin/activate
# Windows (PowerShell)
knn_env\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Ejecutar

```bash
python knn_analysis.py
```

Tiempo estimado: 3–8 minutos (según equipo/configuración).

## 📂 Resultados al ejecutar

Los artefactos principales se guardan en `results/`:

- `evaluation_report.json`: métricas, matriz de confusión y validación cruzada.
- `metrics_summary.csv`: resumen ejecutivo de métricas clave.
- `knn_model.pkl`: modelo entrenado listo para cargar y predecir.
- `correlation_matrix.png`: matriz de correlación de variables numéricas.
- `k_optimization.png`: curva de optimización del hiperparámetro K.
- `model_visualizations/`
  - `KNN_Titanic_Classifier_confusion_matrix.png`: matriz de confusión del modelo.
  - `KNN_Titanic_Classifier_feature_importance.png`: importancia de características.

## 📚 Documentación avanzada

- Proceso detallado del pipeline: `docs/PROCESO_KNN.md`
- Análisis e interpretación de resultados: `docs/ANALISIS_RESULTADOS.md`

## 🧭 Estructura mínima del proyecto

```
knn-clasification/
├── src/
├── data/
├── results/
└── docs/
```

## 📝 Notas

- Para usar `knn_model.pkl` en inferencia, aplica exactamente el mismo preprocesamiento (escalado y columnas) que en el entrenamiento.
