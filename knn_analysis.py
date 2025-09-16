#!/usr/bin/env python3
"""Objetivo: Ejecutar el análisis completo de clasificación K-NN para supervivencia del
Titanic de punta a punta. Bosquejo de funciones: `main()` parsea argumentos, configura
`Config` y logging, y orquesta los pasos:
1) Carga y exploración de datos con `DataLoader` (load_titanic_data, sample_data,
   validate_data, get_data_summary).
2) Preprocesamiento con `DataPreprocessor` (preprocess_pipeline, get_feature_names,
   get_target_classes).
3) Análisis de correlaciones y visualización con `DataVisualizer` (analyze_correlations,
   plot_correlation_matrix).
4) Entrenamiento y optimización del modelo con `KNNClassifier` (train, find_optimal_k,
   plot_k_optimization, get_model_summary).
5) Evaluación integral con `ModelEvaluator` (evaluate_model, print_summary,
   save_evaluation_report, export_results_to_csv, generate_visualizations).
6) Importancia de características con `KNNClassifier` (feature_importance_analysis,
   calculate_feature_importance, plot_feature_importance).
7) Persistencia del modelo (`save_model`) y reporte final de artefactos generados."""

import sys
import warnings
from pathlib import Path
import argparse
import matplotlib

matplotlib.use("Agg")

warnings.filterwarnings("ignore")

from src.data_processing import DataLoader, DataPreprocessor
from src.models import KNNClassifier
from src.visualization import DataVisualizer
from src.evaluation import ModelEvaluator
from src.utils import Config, setup_logger


def main():

    parser = argparse.ArgumentParser(
        description="Análisis K-NN de supervivencia del Titanic"
    )
    parser.add_argument(
        "--data-path", default=None, help="Ruta al archivo de datos CSV"
    )
    parser.add_argument(
        "--sample-size", type=int, default=1000, help="Tamaño de muestra para análisis"
    )
    parser.add_argument("--output-dir", default="results", help="Directorio de salida")
    parser.add_argument(
        "--skip-viz", action="store_true", help="Saltar visualizaciones del modelo"
    )
    args = parser.parse_args()

    config = Config()
    if args.data_path:
        config.DATA_PATH = args.data_path
    config.RESULTS_PATH = args.output_dir

    logger = setup_logger("KNN_Analysis", "INFO")

    print("=" * 80)
    print("ANÁLISIS DE CLASIFICACIÓN K-NN - SUPERVIVENCIA DEL TITANIC")
    print("=" * 80)
    print(f"Dataset: {config.DATA_PATH}")
    print(f"Tamaño de muestra: {args.sample_size:,}")
    print(f"Directorio de salida: {config.RESULTS_PATH}")
    print("=" * 80)

    try:
        # ====================================================================
        # 1. CARGA Y EXPLORACIÓN DE DATOS
        # ====================================================================
        print("\n🔄 PASO 1: Carga y exploración de datos")
        print("-" * 50)
        data_loader = DataLoader(config)
        print("Cargando dataset del Titanic...")
        df = data_loader.load_titanic_data()
        data_loader.get_data_summary(df)
        validation_report = data_loader.validate_data(df)
        print(f"\n📊 Validación completada:")
        print(f"  - Filas duplicadas: {validation_report['duplicated_rows']:,}")
        print(f"  - Columnas numéricas: {len(validation_report['numeric_columns'])}")
        print(
            f"  - Columnas categóricas: {len(validation_report['categorical_columns'])}"
        )

        if len(df) > args.sample_size:
            print(f"\n📦 Creando muestra de {args.sample_size:,} registros...")
            df_sample = data_loader.sample_data(n=args.sample_size, random_state=42)
        else:
            print(f"\n📦 Usando todo el dataset ({len(df):,} registros)...")
            df_sample = df.copy()

        print(
            f"✅ Datos cargados: {df_sample.shape[0]:,} filas, {df_sample.shape[1]} columnas"
        )

        # ====================================================================
        # 2. INFORMACIÓN BÁSICA DEL DATASET
        # ====================================================================
        print("\n🔄 PASO 2: Información del dataset")
        print("-" * 50)

        print(
            f"📊 Forma del dataset: {df_sample.shape[0]:,} filas × {df_sample.shape[1]} columnas"
        )
        print(
            f"💾 Memoria utilizada: {df_sample.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )
        print(f"🔍 Valores nulos: {df_sample.isnull().sum().sum():,}")
        print(f"📋 Tipos de datos: {dict(df_sample.dtypes.value_counts())}")

        # ====================================================================
        # 3. ANÁLISIS DE CORRELACIONES (RELACIONES ENTRE VARIABLES)
        # ====================================================================
        print("\n🔄 PASO 3: Análisis de correlaciones entre variables")
        print("-" * 50)

        data_visualizer = DataVisualizer(config)

        print("📊 Análisis de correlaciones se realizará después del preprocesamiento")

        # ====================================================================
        # 4. PREPROCESAMIENTO DE DATOS
        # ====================================================================
        print("\n🔄 PASO 4: Preprocesamiento de datos")
        print("-" * 50)

        preprocessor = DataPreprocessor(config)

        print("Ejecutando pipeline de preprocesamiento...")
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df_sample)

        feature_names = preprocessor.get_feature_names()
        target_classes = preprocessor.get_target_classes()

        if target_classes is None:
            target_classes = ["No Sobrevivió", "Sobrevivió"]

        print(f"✅ Preprocesamiento completado:")
        print(f"  - Características: {len(feature_names)}")
        print(f"  - Clases objetivo: {target_classes}")
        print(f"  - Etiquetas: 0='No Sobrevivió', 1='Sobrevivió'")
        print(f"  - Train set: {X_train.shape}")
        print(f"  - Test set: {X_test.shape}")

        if feature_names:
            print(f"\n📋 Características seleccionadas:")
            for i, feature in enumerate(feature_names, 1):
                print(f"  {i:2d}. {feature}")

        # ====================================================================
        # 4.1. ANÁLISIS DE CORRELACIONES CON DATOS PREPROCESADOS
        # ====================================================================
        print("\n🔄 PASO 4.1: Análisis de correlaciones con datos preprocesados")
        print("-" * 50)

        df_correlation = preprocessor.select_features_for_classification(
            df_sample.copy()
        )

        correlation_analysis = data_visualizer.analyze_correlations(
            df_correlation, threshold=0.3
        )

        print(
            "📊 Variables del dataset del Titanic para correlación (después de preprocesamiento):"
        )
        for i, col in enumerate(correlation_analysis["numeric_columns"], 1):
            unique_vals = df_correlation[col].nunique()
            min_val = df_correlation[col].min()
            max_val = df_correlation[col].max()
            print(
                f"  {i:2d}. {col} (valores únicos: {unique_vals}, rango: {min_val:.2f} - {max_val:.2f})"
            )

        print(f"\n📊 Correlaciones más fuertes encontradas (umbral > 0.3):")
        strong_corrs = correlation_analysis["strong_correlations"]

        if strong_corrs:
            for i, corr_info in enumerate(strong_corrs[:10], 1):  # Top 10 correlaciones
                direction = (
                    "📈 Positiva" if corr_info["correlation"] > 0 else "📉 Negativa"
                )
                print(
                    f"  {i:2d}. {corr_info['var1']} ↔ {corr_info['var2']}: {corr_info['correlation']:.3f} ({direction})"
                )
        else:
            print("  • No se encontraron correlaciones fuertes (>0.3)")

        if not args.skip_viz:
            print("\n📊 Generando matriz de correlación visual...")
            correlation_path = Path(config.RESULTS_PATH) / "correlation_matrix.png"
            data_visualizer.plot_correlation_matrix(
                df_correlation, save_path=str(correlation_path), show=False
            )
            print(f"  ✅ Matriz de correlación guardada en: {correlation_path}")

        if target_classes:
            print(f"\n🎯 Clases objetivo: {target_classes}")

        # ====================================================================
        # 5. ENTRENAMIENTO DEL MODELO K-NN
        # ====================================================================
        print("\n🔄 PASO 5: Entrenamiento del modelo K-NN")
        print("-" * 50)

        knn_model = KNNClassifier(config)

        print("Entrenando modelo con optimización de hiperparámetros...")
        knn_model.train(X_train, y_train, optimize_k=True, use_grid_search=False)

        model_summary = knn_model.get_model_summary()
        print(f"✅ Modelo entrenado:")
        print(f"  - Mejor K: {model_summary['best_k']}")
        print(f"  - Accuracy entrenamiento: {model_summary['train_accuracy']:.4f}")

        if not args.skip_viz:
            k_plot_path = Path(config.RESULTS_PATH) / "k_optimization.png"
            knn_model.plot_k_optimization(save_path=str(k_plot_path), show=False)
            print(f"  - Gráfico K guardado en: {k_plot_path}")

        # ====================================================================
        # 6. EVALUACIÓN DEL MODELO
        # ====================================================================
        print("\n🔄 PASO 6: Evaluación del modelo")
        print("-" * 50)

        evaluator = ModelEvaluator(config)
        evaluation_results = evaluator.evaluate_model(
            model=knn_model.model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name="KNN_Titanic_Classifier",
            class_names=target_classes,
            feature_names=feature_names,
        )

        evaluator.print_summary()

        report_path = Path(config.RESULTS_PATH) / "evaluation_report.json"
        evaluator.save_evaluation_report(str(report_path), include_predictions=False)
        print(f"\n💾 Reporte guardado en: {report_path}")

        csv_path = Path(config.RESULTS_PATH) / "metrics_summary.csv"
        evaluator.export_results_to_csv(str(csv_path))
        print(f"📊 Métricas exportadas a: {csv_path}")

        # ====================================================================
        # 7. VISUALIZACIONES DEL MODELO
        # ====================================================================
        if not args.skip_viz:
            print("\n🔄 PASO 7: Generación de visualizaciones")
            print("-" * 50)

            viz_dir = Path(config.RESULTS_PATH) / "model_visualizations"

            viz_files = evaluator.generate_visualizations(
                save_dir=str(viz_dir), show_plots=False
            )

            print(f"✅ Visualizaciones generadas: {len(viz_files)} archivos")
            for name, path in viz_files.items():
                print(f"  - {name}: {path}")

        # ====================================================================
        # 8. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
        # ====================================================================
        print("\n🔄 PASO 8: Análisis de importancia de características")
        print("-" * 50)

        if feature_names:
            importance_dict = knn_model.feature_importance_analysis(
                X_train, y_train, feature_names
            )

            print("📈 Importancia de características (Top 10):")
            for i, (feature, importance) in enumerate(
                list(importance_dict.items())[:10], 1
            ):
                print(f"  {i:2d}. {feature}: {importance:.4f}")

            print(
                "📊 Visualización de importancia disponible en: model_visualizations/"
            )

        # ====================================================================
        # 9. VISUALIZACIONES DE RESUMEN FINAL
        # ====================================================================
        print("\n🔄 PASO 9: Generando visualizaciones de resumen final")
        print("-" * 50)

        if feature_names:
            importance_dict = knn_model.calculate_feature_importance(
                X_test, y_test, feature_names
            )
        else:
            importance_dict = {}

        cv_results_dict = (
            knn_model.cv_results
            if hasattr(knn_model, "cv_results") and knn_model.cv_results
            else {}
        )

        print(f"📊 Análisis de importancia de características completado")
        print(f"💼 Insights de supervivencia del Titanic generados")

        # ====================================================================
        # 10. GUARDAR MODELO
        # ====================================================================
        print("\n🔄 PASO 10: Guardando modelo")
        print("-" * 50)

        model_path = Path(config.RESULTS_PATH) / "knn_model.pkl"
        knn_model.save_model(str(model_path))
        print(f"💾 Modelo guardado en: {model_path}")

        # ====================================================================
        # RESUMEN FINAL
        # ====================================================================
        print("\n" + "=" * 80)
        print("🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 80)

        final_metrics = evaluation_results["performance_metrics"]["test_metrics"]
        print(f"📊 MÉTRICAS FINALES:")
        print(f"  - Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  - Precision: {final_metrics['precision']:.4f}")
        print(f"  - Recall: {final_metrics['recall']:.4f}")
        print(f"  - F1-Score: {final_metrics['f1_score']:.4f}")
        print(f"  - Mejor K: {model_summary['best_k']}")

        print(f"\n📁 ARCHIVOS GENERADOS:")
        results_dir = Path(config.RESULTS_PATH)
        if results_dir.exists():
            all_files = list(results_dir.rglob("*"))
            file_count = len([f for f in all_files if f.is_file()])
            print(f"  - Total de archivos: {file_count}")
            print(f"  - Directorio: {results_dir.absolute()}")

            print(f"\n📋 ARCHIVOS PRINCIPALES:")
            print(f"  • evaluation_report.json - Reporte técnico completo")
            print(f"  • metrics_summary.csv - Resumen ejecutivo")
            print(
                f"  • knn_model.pkl - Modelo entrenado para supervivencia del Titanic"
            )
            print(f"  • correlation_matrix.png - Matriz de correlación")
            print(f"  • k_optimization.png - Optimización de K")
            print(f"  • final_summary_dashboard.png - 🆕 Dashboard de resumen final")
            print(f"  • business_insights_summary.png - 🆕 Insights de supervivencia")
            print(f"  • model_visualizations/ - Matriz de confusión e importancia")

        print("\n✅ Análisis completo finalizado con éxito!")
        print(
            "🎯 Se han generado visualizaciones para el análisis de supervivencia del Titanic."
        )

    except Exception as e:
        logger.error(f"Error durante el análisis: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
