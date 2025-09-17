"""Clasificador de Red Neuronal Artificial con optimización de hiperparámetros.

Hiperparámetros principales:
- hidden_layer_sizes: Tamaño de las capas ocultas
- activation: Función de activación ('identity', 'logistic', 'tanh', 'relu')
- solver: Algoritmo de optimización ('lbfgs', 'sgd', 'adam')
- alpha: Parámetro de regularización L2
- learning_rate: Tipo de tasa de aprendizaje ('constant', 'invscaling', 'adaptive')
- learning_rate_init: Tasa de aprendizaje inicial
- max_iter: Número máximo de iteraciones
- early_stopping: Si usar parada temprana
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.neural_network import MLPClassifier

from .base_classifier import BaseClassifier


class NeuralNetworkClassifier(BaseClassifier):
    """Clasificador de Red Neuronal Artificial (MLP)"""

    def get_algorithm_name(self) -> str:
        return "Neural Network (MLP)"

    def get_default_params(self) -> Dict[str, Any]:
        """Parámetros por defecto optimizados para la mayoría de casos"""
        return {
            "hidden_layer_sizes": (100, 50),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "max_iter": 1000,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": self.config.get_hyperparameter_config()["random_state"],
        }

    def get_param_grid(self) -> Dict[str, List]:
        """Grilla de hiperparámetros para optimización (reducida para velocidad)"""
        return {
            "hidden_layer_sizes": [
                (50,),
                (100,),
                (100, 50),  # Reducido de 8 a 3 combinaciones
            ],
            "activation": [
                "relu",
                "tanh",
            ],  # Reducido de 3 a 2 valores (eliminado logistic)
            "solver": ["adam", "lbfgs"],  # Reducido de 3 a 2 valores (eliminado sgd)
            "alpha": [0.0001, 0.001, 0.01],  # Reducido de 4 a 3 valores
            "learning_rate_init": [0.001, 0.01],  # Reducido de 3 a 2 valores
            "max_iter": [1000, 2000],  # Reducido de 3 a 2 valores
        }

    def _create_model(self, **params) -> MLPClassifier:
        """Crear instancia de MLPClassifier"""
        default_params = self.get_default_params()
        default_params.update(params)

        # Validar combinaciones de parámetros
        if default_params["solver"] == "lbfgs" and isinstance(
            default_params["hidden_layer_sizes"], tuple
        ):
            if len(default_params["hidden_layer_sizes"]) > 1:
                # lbfgs funciona mejor con una sola capa oculta
                default_params["hidden_layer_sizes"] = (
                    default_params["hidden_layer_sizes"][0],
                )

        return MLPClassifier(**default_params)

    def get_loss_curve(self) -> Optional[np.ndarray]:
        """Obtener curva de pérdida durante el entrenamiento"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        if hasattr(self.model, "loss_curve_"):
            return self.model.loss_curve_
        return None

    def get_validation_scores(self) -> Optional[np.ndarray]:
        """Obtener scores de validación si early_stopping está habilitado"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        if hasattr(self.model, "validation_scores_"):
            return self.model.validation_scores_
        return None

    def get_n_iterations(self) -> int:
        """Obtener número de iteraciones realizadas"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.n_iter_

    def get_network_architecture(self) -> Dict[str, Any]:
        """Obtener información sobre la arquitectura de la red"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        architecture = {
            "input_layer_size": self.model.coefs_[0].shape[0],
            "hidden_layers": [],
            "output_layer_size": self.model.coefs_[-1].shape[1],
            "total_parameters": 0,
        }

        # Analizar capas ocultas
        for i, coef in enumerate(self.model.coefs_[:-1]):
            layer_info = {
                "layer_index": i + 1,
                "neurons": coef.shape[1],
                "weights": coef.shape[0] * coef.shape[1],
                "biases": coef.shape[1],
            }
            layer_info["total_params"] = layer_info["weights"] + layer_info["biases"]
            architecture["hidden_layers"].append(layer_info)
            architecture["total_parameters"] += layer_info["total_params"]

        # Capa de salida
        output_coef = self.model.coefs_[-1]
        output_params = (
            output_coef.shape[0] * output_coef.shape[1] + output_coef.shape[1]
        )
        architecture["total_parameters"] += output_params

        return architecture

    def plot_loss_curve(self, save_path: Optional[str] = None, show: bool = True):
        """Visualizar curva de pérdida durante el entrenamiento"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        loss_curve = self.get_loss_curve()
        validation_scores = self.get_validation_scores()

        if loss_curve is None:
            raise ValueError("No hay curva de pérdida disponible")

        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(12, 5))

        # Subplot 1: Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(loss_curve, label="Training Loss", color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Subplot 2: Validation scores (si están disponibles)
        plt.subplot(1, 2, 2)
        if validation_scores is not None:
            plt.plot(validation_scores, label="Validation Score", color="orange")
            plt.xlabel("Iterations")
            plt.ylabel("Score")
            plt.title("Validation Score Curve")
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            plt.text(
                0.5,
                0.5,
                "No validation scores available\n(early_stopping=False)",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Validation Score")

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            self.logger.info(
                f"Gráfico de curvas de entrenamiento guardado en: {save_path}"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_network_architecture(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """Visualizar arquitectura de la red neuronal"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.patches as patches  # type: ignore

        architecture = self.get_network_architecture()

        fig, ax = plt.subplots(figsize=(12, 8))

        # Configurar límites
        max_neurons = max(
            [architecture["input_layer_size"]]
            + [layer["neurons"] for layer in architecture["hidden_layers"]]
            + [architecture["output_layer_size"]]
        )

        n_layers = len(architecture["hidden_layers"]) + 2  # Input + Hidden + Output

        ax.set_xlim(0, n_layers + 1)
        ax.set_ylim(0, max_neurons + 1)

        # Dibujar capas
        layer_x = 1

        # Capa de entrada
        for i in range(architecture["input_layer_size"]):
            circle = patches.Circle(
                (layer_x, i + 1), 0.1, color="lightblue", ec="black"
            )
            ax.add_patch(circle)
        ax.text(
            layer_x,
            max_neurons + 0.5,
            f'Input\n({architecture["input_layer_size"]})',
            ha="center",
            fontsize=10,
            weight="bold",
        )

        layer_x += 1

        # Capas ocultas
        for layer_idx, layer in enumerate(architecture["hidden_layers"]):
            for i in range(layer["neurons"]):
                circle = patches.Circle(
                    (layer_x, i + 1), 0.1, color="lightgreen", ec="black"
                )
                ax.add_patch(circle)
            ax.text(
                layer_x,
                max_neurons + 0.5,
                f'Hidden {layer_idx + 1}\n({layer["neurons"]})',
                ha="center",
                fontsize=10,
                weight="bold",
            )
            layer_x += 1

        # Capa de salida
        for i in range(architecture["output_layer_size"]):
            circle = patches.Circle(
                (layer_x, i + 1), 0.1, color="lightcoral", ec="black"
            )
            ax.add_patch(circle)
        ax.text(
            layer_x,
            max_neurons + 0.5,
            f'Output\n({architecture["output_layer_size"]})',
            ha="center",
            fontsize=10,
            weight="bold",
        )

        ax.set_aspect("equal")
        ax.set_title(
            f'Neural Network Architecture\nTotal Parameters: {architecture["total_parameters"]:,}'
        )
        ax.axis("off")

        # Añadir información adicional
        info_text = f"Activation: {self.model.activation}\nSolver: {self.model.solver}\nIterations: {self.get_n_iterations()}"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        if save_path is not None:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            self.logger.info(f"Gráfico de arquitectura de red guardado en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def analyze_convergence(self) -> Dict[str, Any]:
        """Analizar convergencia del entrenamiento"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        loss_curve = self.get_loss_curve()
        validation_scores = self.get_validation_scores()

        analysis = {
            "n_iterations": self.get_n_iterations(),
            "final_loss": loss_curve[-1] if loss_curve is not None else None,
            "converged": self.model.n_iter_ < self.model.max_iter,
            "early_stopped": hasattr(self.model, "best_validation_score_"),
        }

        if loss_curve is not None and len(loss_curve) > 10:
            # Analizar tendencia de la pérdida en las últimas iteraciones
            recent_loss = loss_curve[-10:]
            loss_trend = np.polyfit(range(len(recent_loss)), recent_loss, 1)[0]
            analysis["loss_trend"] = "decreasing" if loss_trend < -1e-6 else "stable"

        if validation_scores is not None:
            analysis["best_validation_score"] = np.max(validation_scores)
            analysis["final_validation_score"] = validation_scores[-1]

        self.logger.info(f"Análisis de convergencia: {analysis}")
        return analysis
