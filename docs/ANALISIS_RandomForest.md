### Random Forest — Coffee Shop Success

#### Resumen de rendimiento (test)

- **Accuracy**: 0.9500
- **Precision (weighted)**: 0.9504
- **Recall (weighted)**: 0.9500
- **F1-Score (weighted)**: 0.9501
- **AUC-ROC**: 0.9909
- **Average Precision (AP)**: 0.9874
- **Overfitting score**: 0.0000

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

- **Accuracy (mean ± std)**: 0.9300 ± 0.0172
- **F1 (weighted, mean ± std)**: 0.9300 ± 0.0172

#### Matriz de confusión (test)

- VN: 294, FP: 6, FN: 12, VP: 88 — error rate 4.5%

Interpretación: desempeño top, robusto y estable; probabilidad bien calibrada (AUC/AP). Ligeramente por debajo de SVM/NN en Accuracy, pero excelente opción por interpretabilidad parcial (importancias) y resiliencia.

#### Hiperparámetros seleccionados

```text
n_estimators = 100
criterion = 'entropy'
max_depth = 10
min_samples_split = 5
min_samples_leaf = 1
max_features = 'sqrt'
bootstrap = True
class_weight = None
random_state = 42
```

#### ¿Qué significa cada hiperparámetro?

- `n_estimators`: número de árboles en el bosque.
- `criterion`: impureza usada por cada árbol.
- `max_depth`, `min_samples_split`, `min_samples_leaf`: límites de complejidad por árbol.
- `max_features`: features consideradas por división (controla correlación entre árboles).
- `bootstrap`: muestreo con reemplazo de observaciones para cada árbol.
- `class_weight`: ponderación por clase.
- `random_state`: semilla.

#### Importancia de características

- Number_of_Customers_Per_Day: 0.461
- Average_Order_Value: 0.422
- Marketing_Spend_Per_Day: 0.089
- Resto pequeños

Conclusión: excelente equilibrio entre rendimiento y robustez; recomendado cuando se quiere menor varianza que un árbol individual y explicabilidad básica.

#### Visualizaciones

- Confusion Matrix: `results/randomforest/visualizations/RandomForest_Coffee_Shop_Classifier_confusion_matrix.png`
- Feature Importance: `results/randomforest/visualizations/RandomForest_Coffee_Shop_Classifier_feature_importance.png`
- ROC: `results/randomforest/visualizations/RandomForest_Coffee_Shop_Classifier_roc_curve.png`
- Precision-Recall: `results/randomforest/visualizations/RandomForest_Coffee_Shop_Classifier_precision_recall_curve.png`
