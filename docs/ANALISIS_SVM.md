### Support Vector Machine (RBF) — Coffee Shop Success

#### Resumen de rendimiento (test)

- **Accuracy**: 0.9475
- **Precision (weighted)**: 0.9478
- **Recall (weighted)**: 0.9475
- **F1-Score (weighted)**: 0.9476
- **AUC-ROC**: 0.9921
- **Average Precision (AP)**: 0.9886
- **Overfitting score**: 0.0213

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

- **Accuracy (mean ± std)**: 0.9263 ± 0.0121
- **F1 (weighted, mean ± std)**: 0.9263 ± 0.0121

#### Matriz de confusión (test)

- VN: 293, FP: 7, FN: 10, VP: 90 — error rate 4.25%

Interpretación: mejor desempeño global y excelente calibración (AUC/AP). El costo de errores se concentra en la clase positiva (10 FN).

#### Hiperparámetros seleccionados

```text
kernel = 'rbf'
C = 10.0
gamma = 0.01
probability = True
class_weight = None
random_state = 42
```

#### ¿Qué significa cada hiperparámetro?

- `kernel`: mapea a un espacio de mayor dimensión; `rbf` captura fronteras no lineales.
- `C`: penaliza errores de entrenamiento (más alto = menos regularización, ajuste más fino).
- `gamma`: controla el alcance de la influencia de un punto (alto = más local, riesgo de sobreajuste).
- `probability`: habilita estimación de probabilidades (Platt scaling).
- `class_weight`: pondera clases para desbalance.
- `random_state`: semilla.

#### Importancia de características (aprox. por permutación)

- Number_of_Customers_Per_Day: 0.468
- Average_Order_Value: 0.449
- Marketing_Spend_Per_Day: 0.078
- Resto ≈ 0.002–0.003

Conclusión: SVM es el mejor comprometiendo precisión y generalización. Recomendado como modelo final; ajustar umbral si se prioriza Recall para “Exitosa”.

#### Visualizaciones

- Confusion Matrix: `results/svm/visualizations/SVM_Coffee_Shop_Classifier_confusion_matrix.png`
- Feature Importance: `results/svm/visualizations/SVM_Coffee_Shop_Classifier_feature_importance.png`
- ROC: `results/svm/visualizations/SVM_Coffee_Shop_Classifier_roc_curve.png`
- Precision-Recall: `results/svm/visualizations/SVM_Coffee_Shop_Classifier_precision_recall_curve.png`
