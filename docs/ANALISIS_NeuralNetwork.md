### Neural Network (MLP) — Coffee Shop Success

#### Resumen de rendimiento (test)

- **Accuracy**: 0.9525
- **Precision (weighted)**: 0.9534
- **Recall (weighted)**: 0.9525
- **F1-Score (weighted)**: 0.9526
- **AUC-ROC**: 0.9931
- **Average Precision (AP)**: 0.9901
- **Overfitting score**: -0.0050

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

- **Accuracy (mean ± std)**: 0.9313 ± 0.0123
- **F1 (weighted, mean ± std)**: 0.9313 ± 0.0123

#### Matriz de confusión (test)

- VN: 291, FP: 9, FN: 8, VP: 92 — error rate 4.25%

Interpretación: rendimiento top a la par de SVM; estable gracias a `early_stopping`. Ligeramente menor AUC que RF, pero diferencias pequeñas.

#### Hiperparámetros seleccionados

```text
hidden_layer_sizes = (100, 50)
activation = 'tanh'
learning_rate_init = 0.01
solver = 'adam'
max_iter = 1000
early_stopping = True
alpha = 0.0001
random_state = 42
```

#### ¿Qué significa cada hiperparámetro?

- `hidden_layer_sizes`: neuronas por capa oculta.
- `activation`: función de activación de capas ocultas (tanh = no lineal suave).
- `learning_rate_init`: tasa de aprendizaje inicial.
- `solver`: optimizador; `adam` adapta tasas y es robusto.
- `max_iter`: iteraciones máximas.
- `early_stopping`: detiene el entrenamiento si no mejora en validación.
- `alpha`: regularización L2 (weight decay) para prevenir sobreajuste.
- `random_state`: semilla.

#### Importancia de características (permutación)

- Number_of_Customers_Per_Day: 0.467
- Average_Order_Value: 0.449
- Marketing_Spend_Per_Day: 0.078
- Resto ≈ 0.0–0.003

Conclusión: opción potente cuando se dispone de capacidad de cómputo; requiere tuning cuidadoso de tasas de aprendizaje y arquitectura.

#### Visualizaciones

- Confusion Matrix: `results/neuralnetwork/visualizations/NeuralNetwork_Coffee_Shop_Classifier_confusion_matrix.png`
- Feature Importance: `results/neuralnetwork/visualizations/NeuralNetwork_Coffee_Shop_Classifier_feature_importance.png`
- ROC: `results/neuralnetwork/visualizations/NeuralNetwork_Coffee_Shop_Classifier_roc_curve.png`
- Precision-Recall: `results/neuralnetwork/visualizations/NeuralNetwork_Coffee_Shop_Classifier_precision_recall_curve.png`
