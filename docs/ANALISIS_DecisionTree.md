### Decision Tree — Coffee Shop Success

#### Resumen de rendimiento (test)

- **Accuracy**: 0.8550
- **Precision (weighted)**: 0.8624
- **Recall (weighted)**: 0.8550
- **F1-Score (weighted)**: 0.8561
- **AUC-ROC**: 0.8860
- **Average Precision (AP)**: 0.8022
- **Overfitting score**: 0.0419 (indica sobreajuste moderado)

#### Validación cruzada (5 folds)

#### Partición de datos

- División: 80% entrenamiento, 20% prueba (estratificada)
- **Train: 1,600 muestras**
  - 952 muestras de cafeterías "No Exitosas" (ingresos < $2,000)
  - 648 muestras de cafeterías "Exitosas" (ingresos ≥ $2,000)
- **Test: 400 muestras**
  - 238 muestras de cafeterías "No Exitosas" (ingresos < $2,000)
  - 162 muestras de cafeterías "Exitosas" (ingresos ≥ $2,000)
- Características: 6
- Balance: dataset moderadamente balanceado (59.5% No Exitosa, 40.5% Exitosa)

#### Definición de "éxito" (objetivo)

- `Successful` = 1 si `Daily_Revenue` ≥ $2,000; 0 en caso contrario.
- Esto clasifica como "Exitosa" a la cafetería con ingresos diarios de $2,000 o más.

- **Accuracy (mean ± std)**: 0.8806 ± 0.0322
- **F1 (weighted, mean ± std)**: 0.8845 ± 0.0300

#### Matriz de confusión (test)

- VN: 286, FP: 14, FN: 18, VP: 82 — error rate 8.0%

Interpretación: rendimiento inferior al resto y probabilidad peor calibrada (AUC/AP bajos). Sensible al umbral y a la profundidad; evidencia de sobreajuste al comparar train 0.986 vs test 0.920.

#### Hiperparámetros seleccionados

```text
criterion = 'entropy'
max_depth = 10
min_samples_split = 2
min_samples_leaf = 2
max_features = 'sqrt'
class_weight = None
random_state = 42
```

#### ¿Qué significa cada hiperparámetro?

- `criterion`: función de impureza para dividir nodos (`entropy`/`gini`).
- `max_depth`: profundidad máxima del árbol (controla complejidad).
- `min_samples_split`: mínimo de muestras para dividir un nodo.
- `min_samples_leaf`: mínimo de muestras en una hoja (suaviza reglas y reduce sobreajuste).
- `max_features`: proporción/número de variables consideradas por división.
- `class_weight`: pondera clases.
- `random_state`: semilla.

#### Importancia de características

- Number_of_Customers_Per_Day: 0.459
- Average_Order_Value: 0.382
- Marketing_Spend_Per_Day: 0.096
- Resto pequeños

Recomendaciones: aumentar regularización vía `ccp_alpha`, reducir profundidad o incrementar `min_samples_leaf/split`. Como alternativa, usar ensembles (RandomForest/GradientBoosting).

#### Visualizaciones

- Confusion Matrix: `results/decisiontree/visualizations/DecisionTree_Coffee_Shop_Classifier_confusion_matrix.png`
- Feature Importance: `results/decisiontree/visualizations/DecisionTree_Coffee_Shop_Classifier_feature_importance.png`
- ROC: `results/decisiontree/visualizations/DecisionTree_Coffee_Shop_Classifier_roc_curve.png`
- Precision-Recall: `results/decisiontree/visualizations/DecisionTree_Coffee_Shop_Classifier_precision_recall_curve.png`
