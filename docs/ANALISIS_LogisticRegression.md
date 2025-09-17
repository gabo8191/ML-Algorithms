### Logistic Regression — Coffee Shop Success

#### Resumen de rendimiento (test)

- **Accuracy**: 0.9300
- **Precision (weighted)**: 0.9324
- **Recall (weighted)**: 0.9300
- **F1-Score (weighted)**: 0.9303
- **AUC-ROC**: 0.9884
- **Average Precision (AP)**: 0.9841
- **Overfitting score (train−test accuracy)**: -0.0038

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

- **Accuracy (mean ± std)**: 0.9200 ± 0.0181
- **F1 (weighted, mean ± std)**: 0.9200 ± 0.0181

#### Matriz de confusión (test)

- Verdaderos Negativos: 218
- Falsos Positivos: 20
- Falsos Negativos: 8
- Verdaderos Positivos: 154
- Error rate: 7.0%

Interpretación: buen desempeño global; ligera asimetría en la clase positiva "Exitosa" (Recall 0.95, Precision 0.89) frente a "No Exitosa" (Recall 0.92).

#### Definición de "éxito" (objetivo)

- `Successful` = 1 si `Daily_Revenue` ≥ $2,000; 0 en caso contrario.
- Esto clasifica como "Exitosa" a la cafetería con ingresos diarios de $2,000 o más.

#### Hiperparámetros seleccionados

```text
penalty = 'l1'
solver = 'liblinear'
C = 0.1
max_iter = 1000
class_weight = None
random_state = 42
```

#### ¿Qué significa cada hiperparámetro?

- `penalty`: tipo de regularización (L1 induce coeficientes cero y selección implícita de variables).
- `solver`: algoritmo de optimización; `liblinear` permite L1 en binario y es robusto.
- `C`: inverso de la regularización (menor C = más regularización).
- `max_iter`: tope de iteraciones del optimizador.
- `class_weight`: pondera clases para tratar desbalance (None = sin ponderación).
- `random_state`: semilla para reproducibilidad.

#### Importancia de características (coeficientes normalizados)

- Number_of_Customers_Per_Day: 0.481
- Average_Order_Value: 0.453
- Marketing_Spend_Per_Day: 0.066
- Resto ≈ 0

Conclusión: modelo lineal fuerte y bien calibrado (AUC/AP altos). Puede servir como baseline rápido y explicable. Para maximizar Recall en “Exitosa”, ajustar umbral o aplicar costo por clase.

#### Visualizaciones

- Confusion Matrix: `results/logisticregression/visualizations/LogisticRegression_Coffee_Shop_Classifier_confusion_matrix.png`
- Feature Importance: `results/logisticregression/visualizations/LogisticRegression_Coffee_Shop_Classifier_feature_importance.png`
- ROC: `results/logisticregression/visualizations/LogisticRegression_Coffee_Shop_Classifier_roc_curve.png`
- Precision-Recall: `results/logisticregression/visualizations/LogisticRegression_Coffee_Shop_Classifier_precision_recall_curve.png`
