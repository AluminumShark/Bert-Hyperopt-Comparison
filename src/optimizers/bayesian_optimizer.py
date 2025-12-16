"""Bayesian optimization for BERT hyperparameter tuning.

This module implements Bayesian optimization using Gaussian processes
to efficiently search the hyperparameter space.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

if TYPE_CHECKING:
    from src.bert_classifier import BertTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """Bayesian optimization for finding the best BERT hyperparameters.

    Uses Gaussian processes to model the objective function and smart
    acquisition functions to decide which hyperparameters to try next.
    """

    def __init__(
        self,
        trainer: BertTrainer,
        train_data: tuple[list, list],
        val_data: tuple[list, list],
        n_calls: int = 30,
        n_initial_points: int = 8,
        acq_func: str = "EI",
        random_state: int = 42,
    ) -> None:
        """Initialize the Bayesian optimizer.

        Args:
            trainer: The BERT trainer instance
            train_data: Training texts and labels as a tuple
            val_data: Validation texts and labels as a tuple
            n_calls: Total number of hyperparameter combinations to try
            n_initial_points: Number of random points before using the model
            acq_func: Acquisition function ('EI', 'LCB', 'PI', etc.)
            random_state: Random seed for reproducibility
        """
        self.trainer = trainer
        self.train_texts, self.train_labels = train_data
        self.val_texts, self.val_labels = val_data
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.random_state = random_state

        # Validate acquisition function
        valid_acq_funcs = ["GP_HEDGE", "EI", "LCB", "MES", "PVRS", "PI", "EIPS", "PIPS"]
        self.acq_func = acq_func.upper() if isinstance(acq_func, str) else "EI"
        if self.acq_func not in valid_acq_funcs:
            self.acq_func = "EI"

        self.search_space = self._define_search_space()
        self.evaluation_history: list[dict] = []
        self.best_params_history: list[dict] = []
        self.best_score_history: list[float] = []

        logger.info("Initialized Bayesian optimizer with search space: %s", self.search_space)
        logger.info(
            "Planning %d initial points + %d total evaluations",
            self.n_initial_points,
            self.n_calls,
        )
        logger.info("Using acquisition function: %s", self.acq_func)

    def _define_search_space(self) -> list:
        """Define the hyperparameter search space for BERT fine-tuning."""
        search_space = [
            Real(1e-5, 5e-5, name="learning_rate", prior="log-uniform"),
            Categorical([8, 16, 32], name="batch_size"),
            Integer(2, 5, name="epochs"),
            Real(0.1, 0.5, name="dropout_rate"),
            Categorical([64, 128, 256], name="max_length"),
            Real(0.1, 0.5, name="warmup_ratio"),
        ]

        logger.info("Search space defined with %d dimensions", len(search_space))
        for dim in search_space:
            dim_type = getattr(dim, "dtype", type(dim).__name__)
            dim_bounds = getattr(dim, "bounds", getattr(dim, "categories", "N/A"))
            logger.info("Parameter: %s - Type: %s - Range: %s", dim.name, dim_type, dim_bounds)

        return search_space

    def _objective_function(self, **params: Any) -> float:
        """Evaluate hyperparameters by training BERT.

        Args:
            **params: Dictionary of hyperparameters to evaluate

        Returns:
            Negative validation accuracy (since skopt minimizes)
        """
        start_time = time.time()

        try:
            logger.info("Evaluating hyperparameter set #%d", len(self.evaluation_history) + 1)
            logger.info("Parameters: %s", params)

            # Clean training data
            train_texts, train_labels = self._clean_data(self.train_texts, self.train_labels)
            val_texts, val_labels = self._clean_data(self.val_texts, self.val_labels)

            # Train and evaluate
            result = self.trainer.train_and_evaluate(
                train_texts, train_labels, val_texts, val_labels, **params
            )

            accuracy = result["accuracy"]
            f1_score = result.get("f1_score", 0.0)
            loss = result.get("loss", float("inf"))
            eval_time = time.time() - start_time

            # Record evaluation
            self.evaluation_history.append(
                {
                    "iteration": len(self.evaluation_history) + 1,
                    "params": params,
                    "accuracy": accuracy,
                    "f1_score": f1_score,
                    "loss": loss,
                    "eval_time": eval_time,
                    "timestamp": time.time(),
                }
            )

            # Track best parameters
            if not self.best_params_history or accuracy > max(self.best_score_history):
                self.best_params_history.append(params.copy())
                logger.info("New best accuracy: %.4f with params: %s", accuracy, params)
            else:
                self.best_params_history.append(self.best_params_history[-1])

            self.best_score_history.append(max(self.best_score_history + [accuracy]))

            logger.info(
                "Results: accuracy=%.4f, f1=%.4f, loss=%.4f, time=%.2fs",
                accuracy,
                f1_score,
                loss,
                eval_time,
            )

            return -accuracy

        except Exception as e:
            logger.exception("Error during hyperparameter evaluation: %s", e)
            return 0.0

    def _clean_data(self, texts: list, labels: list) -> tuple[list[str], list[int]]:
        """Clean and validate text-label pairs."""
        clean_texts = []
        clean_labels = []

        for i in range(min(len(texts), len(labels))):
            text = texts[i]
            label = labels[i]

            if isinstance(text, str) and text.strip():
                clean_texts.append(text.strip())
                try:
                    clean_labels.append(int(label))
                except (ValueError, TypeError):
                    clean_labels.append(0)

        return clean_texts, clean_labels

    def optimize(self) -> dict[str, Any]:
        """Run Bayesian optimization to find the best hyperparameters.

        Returns:
            Dictionary with best parameters, score, and optimization history
        """
        logger.info("Starting Bayesian optimization...")
        logger.info("Will evaluate %d hyperparameter combinations", self.n_calls)

        start_time = time.time()

        try:
            result = gp_minimize(
                func=self._objective_function,
                dimensions=self.search_space,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                acq_func=self.acq_func,
                random_state=self.random_state,
                n_jobs=1,
                verbose=False,
            )

            total_time = time.time() - start_time
            best_accuracy = -result.fun

            # Extract best parameters
            best_params = {dim.name: result.x[i] for i, dim in enumerate(self.search_space)}

            convergence_analysis = self._analyze_convergence()

            optimization_results = {
                "best_params": best_params,
                "best_fitness": best_accuracy,
                "fitness_history": [-score for score in result.func_vals],
                "execution_time": total_time,
                "n_evaluations": len(result.func_vals),
                "convergence_analysis": convergence_analysis,
                "evaluation_history": self.evaluation_history,
            }

            self._print_optimization_summary(optimization_results)

            logger.info("Bayesian optimization completed in %.2f seconds", total_time)
            logger.info("Best accuracy achieved: %.4f", best_accuracy)

            return optimization_results

        except Exception as e:
            logger.exception("Error during Bayesian optimization: %s", e)
            return {
                "best_params": {},
                "best_fitness": 0.0,
                "fitness_history": [],
                "execution_time": time.time() - start_time,
                "error": str(e),
            }

    def _analyze_convergence(self) -> dict[str, Any]:
        """Analyze optimization convergence."""
        if len(self.best_score_history) < 3:
            return {"status": "insufficient_data", "recommendation": "Need more evaluations"}

        recent_improvement = self.best_score_history[-1] - self.best_score_history[-3]

        if recent_improvement > 0.01:
            status = "improving"
        elif recent_improvement > 0.001:
            status = "slow_convergence"
        else:
            status = "converged"

        search_efficiency = self._calculate_search_efficiency()

        recommendations = {
            "improving": "Continue optimization - good progress being made",
            "slow_convergence": "Consider increasing exploration or different acquisition function",
            "converged": "Optimization has likely found a good solution",
            "insufficient_data": "Need more evaluations to assess convergence",
        }

        return {
            "status": status,
            "recent_improvement": recent_improvement,
            "search_efficiency": search_efficiency,
            "recommendation": recommendations.get(status, "Unknown status"),
        }

    def _calculate_search_efficiency(self) -> float:
        """Calculate search efficiency metric."""
        if not self.best_score_history:
            return 0.0

        final_score = self.best_score_history[-1]
        evaluations_to_best = len(self.best_score_history)

        return final_score / evaluations_to_best if evaluations_to_best > 0 else 0.0

    def _print_optimization_summary(self, results: dict[str, Any]) -> None:
        """Print optimization summary."""
        logger.info("=" * 60)
        logger.info("BAYESIAN OPTIMIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info("Best validation accuracy: %.4f", results["best_fitness"])
        logger.info("Total evaluations: %d", results["n_evaluations"])
        logger.info("Execution time: %.2f seconds", results["execution_time"])
        logger.info(
            "Average time per evaluation: %.2fs",
            results["execution_time"] / results["n_evaluations"],
        )

        logger.info("\nBest hyperparameters found:")
        for param, value in results["best_params"].items():
            if isinstance(value, float):
                logger.info("  %s: %.6f", param, value)
            else:
                logger.info("  %s: %s", param, value)

        if "convergence_analysis" in results:
            conv = results["convergence_analysis"]
            logger.info("\nConvergence status: %s", conv["status"])
            logger.info("Recommendation: %s", conv["recommendation"])

        logger.info("=" * 60)

    def plot_optimization_progress(self, save_path: str | None = None) -> None:
        """Plot optimization progress.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.evaluation_history:
            logger.warning("No evaluation history to plot")
            return

        if len(self.evaluation_history) < 2:
            logger.warning("Insufficient data to plot progress")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Bayesian Optimization Process Analysis", fontsize=16, fontweight="bold")

        # Convergence curve
        ax1 = axes[0, 0]
        iterations = range(1, len(self.best_score_history) + 1)
        ax1.plot(iterations, self.best_score_history, "b-o", linewidth=2, markersize=4)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Best Accuracy")
        ax1.set_title("Convergence Curve")
        ax1.grid(True, alpha=0.3)

        # All evaluations
        ax2 = axes[0, 1]
        all_scores = [record["accuracy"] for record in self.evaluation_history]
        ax2.scatter(range(1, len(all_scores) + 1), all_scores, alpha=0.6, s=30)
        ax2.plot(iterations, self.best_score_history, "r-", linewidth=2, label="Best Trajectory")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("All Evaluation Results")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning rate exploration
        ax3 = axes[1, 0]
        lr_values = [record["params"]["learning_rate"] for record in self.evaluation_history]
        scatter = ax3.scatter(
            range(1, len(lr_values) + 1), lr_values, c=all_scores, cmap="viridis", s=50, alpha=0.7
        )
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Learning Rate")
        ax3.set_title("Learning Rate Exploration")
        ax3.set_yscale("log")
        plt.colorbar(scatter, ax=ax3, label="Accuracy")

        # Evaluation time distribution
        ax4 = axes[1, 1]
        eval_times = [record["eval_time"] for record in self.evaluation_history]
        ax4.hist(eval_times, bins=10, alpha=0.7, edgecolor="black")
        ax4.axvline(
            np.mean(eval_times),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(eval_times):.1f}s",
        )
        ax4.set_xlabel("Evaluation Time (seconds)")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Evaluation Time Distribution")
        ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Progress plot saved: %s", save_path)

        plt.show()

    def export_results(self, filepath: str) -> None:
        """Export detailed results to JSON file.

        Args:
            filepath: Path to save the results
        """
        export_data = {
            "search_space": [
                {
                    "name": dim.name,
                    "type": type(dim).__name__,
                    "bounds": getattr(dim, "bounds", None),
                    "categories": getattr(dim, "categories", None),
                }
                for dim in self.search_space
            ],
            "optimization_config": {
                "n_calls": self.n_calls,
                "n_initial_points": self.n_initial_points,
                "acq_func": self.acq_func,
                "random_state": self.random_state,
            },
            "evaluation_history": self.evaluation_history,
            "best_score_history": self.best_score_history,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Results exported: %s", filepath)


class BayesianOptimizationBestPractices:
    """Best practices guide for Bayesian optimization."""

    @staticmethod
    def recommend_n_calls(search_space_size: int, time_budget_hours: float) -> int:
        """Recommend number of evaluations based on constraints.

        Args:
            search_space_size: Number of dimensions in search space
            time_budget_hours: Available time in hours

        Returns:
            Recommended number of evaluations
        """
        base_calls = max(15, search_space_size * 8)
        time_limited_calls = int(time_budget_hours * 60 / 5)

        recommended = min(base_calls, time_limited_calls)

        logger.info("Recommended evaluations: %d", recommended)
        logger.info("  Based on search space: %d", base_calls)
        logger.info("  Based on time budget: %d", time_limited_calls)

        return recommended

    @staticmethod
    def choose_acquisition_function(exploration_preference: str) -> str:
        """Choose acquisition function based on preference.

        Args:
            exploration_preference: 'conservative', 'balanced', or 'aggressive'

        Returns:
            Acquisition function name
        """
        mapping = {
            "conservative": "LCB",
            "balanced": "EI",
            "aggressive": "PI",
        }
        return mapping.get(exploration_preference, "EI")


if __name__ == "__main__":
    logger.info("Bayesian Optimizer module loaded successfully!")
    logger.info("Usage: Import BayesianOptimizer and call optimize()")
