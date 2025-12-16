"""Hyperparameter optimization experiment framework.

This module provides the main experiment runner that orchestrates
the comparison of different optimization methods for BERT fine-tuning.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.bert_classifier import BertTrainer
from src.optimizers.bayesian_optimizer import BayesianOptimizer
from src.optimizers.ga_optimizer import GAHyperparameterOptimizer
from src.optimizers.pso_optimizer import PSOHyperparameterOptimizer


class HyperparameterOptimizationExperiment:
    """Framework for comparing different hyperparameter optimization methods.

    This class handles data loading, runs all optimization algorithms,
    and collects results for comparison.
    """

    def __init__(self, data_size: str = "small") -> None:
        """Initialize the experiment.

        Args:
            data_size: Size of dataset - 'small' (1000), 'medium' (5000), or 'large' (full)
        """
        self.data_size = data_size
        self.results: dict[str, Any] = {}
        self.trainer = BertTrainer()

        # Optimization configuration
        self.experiment_config = {
            "ga": {"population_size": 8, "generations": 5},
            "pso": {"n_particles": 8, "n_iterations": 5},
            "bayesian": {"n_calls": 15},
        }

        # Dataset attributes
        self.train_texts: list[str] = []
        self.train_labels: list[int] = []
        self.val_texts: list[str] = []
        self.val_labels: list[int] = []
        self.test_texts: list[str] = []
        self.test_labels: list[int] = []

        print(f"Experiment initialized with dataset size: {data_size}")
        self.load_data()

    def load_data(self) -> None:
        """Load and prepare the IMDB dataset for sentiment analysis."""
        print("Loading IMDB dataset...")

        try:
            dataset = load_dataset("imdb")

            train_texts = dataset["train"]["text"]
            train_labels = dataset["train"]["label"]
            test_texts = dataset["test"]["text"]
            test_labels = dataset["test"]["label"]

            # Determine sample size
            if self.data_size == "small":
                sample_size = 1000
            elif self.data_size == "medium":
                sample_size = 5000
            else:
                sample_size = len(train_texts)

            # Sample if needed
            if sample_size < len(train_texts):
                indices = np.random.choice(len(train_texts), sample_size, replace=False)
                train_texts = [train_texts[i] for i in indices]
                train_labels = [train_labels[i] for i in indices]

            # Split into train/validation
            (
                self.train_texts,
                self.val_texts,
                self.train_labels,
                self.val_labels,
            ) = train_test_split(
                train_texts,
                train_labels,
                test_size=0.2,
                random_state=42,
                stratify=train_labels,
            )

            # Prepare test set
            if sample_size < len(test_texts):
                test_indices = np.random.choice(
                    len(test_texts), min(sample_size, 1000), replace=False
                )
                self.test_texts = [test_texts[i] for i in test_indices]
                self.test_labels = [test_labels[i] for i in test_indices]
            else:
                self.test_texts = test_texts[:1000]
                self.test_labels = test_labels[:1000]

            print("Dataset loaded successfully!")
            print(f"Training samples: {len(self.train_texts)}")
            print(f"Validation samples: {len(self.val_texts)}")
            print(f"Test samples: {len(self.test_texts)}")

        except Exception as e:
            print(f"Error loading IMDB dataset: {e}")
            print("Creating synthetic data for testing...")
            self._create_dummy_data()

    def _create_dummy_data(self) -> None:
        """Create synthetic sentiment data for testing."""
        print("Creating synthetic sentiment data...")

        positive_examples = [
            "This movie is excellent and amazing!",
            "I love this film, it's fantastic!",
            "Great acting and wonderful story!",
            "Best movie I've ever seen!",
            "Brilliant and entertaining film!",
        ] * 50

        negative_examples = [
            "This movie is terrible and boring.",
            "I hate this film, it's awful!",
            "Poor acting and bad story.",
            "Worst movie I've ever seen!",
            "Horrible and disappointing film.",
        ] * 50

        all_texts = positive_examples + negative_examples
        all_labels = [1] * len(positive_examples) + [0] * len(negative_examples)

        # Shuffle
        combined = list(zip(all_texts, all_labels))
        np.random.shuffle(combined)
        all_texts, all_labels = zip(*combined)

        # Split
        train_split = int(len(all_texts) * 0.6)
        val_split = int(len(all_texts) * 0.8)

        self.train_texts = list(all_texts[:train_split])
        self.train_labels = list(all_labels[:train_split])
        self.val_texts = list(all_texts[train_split:val_split])
        self.val_labels = list(all_labels[train_split:val_split])
        self.test_texts = list(all_texts[val_split:])
        self.test_labels = list(all_labels[val_split:])

        print("Synthetic data created successfully!")
        print(f"Training samples: {len(self.train_texts)}")
        print(f"Validation samples: {len(self.val_texts)}")
        print(f"Test samples: {len(self.test_texts)}")

    def run_genetic_algorithm(self) -> None:
        """Run genetic algorithm optimization."""
        print("=" * 60)
        print("Starting Genetic Algorithm Optimization")
        print("=" * 60)

        start_time = time.time()

        ga_optimizer = GAHyperparameterOptimizer(
            trainer=self.trainer,
            train_data=(self.train_texts, self.train_labels),
            val_data=(self.val_texts, self.val_labels),
            **self.experiment_config["ga"],
        )

        ga_results = ga_optimizer.optimize()
        execution_time = time.time() - start_time

        ga_results["execution_time"] = execution_time
        ga_results["method"] = "Genetic Algorithm"
        self.results["ga"] = ga_results

        print(f"Genetic Algorithm completed in {execution_time:.2f} seconds")
        print(f"Best parameters found: {ga_results['best_params']}")
        print(f"Best validation accuracy: {ga_results['best_fitness']:.4f}")

    def run_particle_swarm_optimization(self) -> None:
        """Run particle swarm optimization."""
        print("=" * 60)
        print("Starting Particle Swarm Optimization")
        print("=" * 60)

        start_time = time.time()

        pso_optimizer = PSOHyperparameterOptimizer(
            trainer=self.trainer,
            train_data=(self.train_texts, self.train_labels),
            val_data=(self.val_texts, self.val_labels),
            **self.experiment_config["pso"],
        )

        pso_results = pso_optimizer.optimize()
        execution_time = time.time() - start_time

        pso_results["execution_time"] = execution_time
        pso_results["method"] = "Particle Swarm Optimization"
        self.results["pso"] = pso_results

        print(f"Particle Swarm Optimization completed in {execution_time:.2f} seconds")
        print(f"Best parameters found: {pso_results['best_params']}")
        print(f"Best validation accuracy: {pso_results['best_fitness']:.4f}")

    def run_bayesian_optimization(self) -> None:
        """Run Bayesian optimization."""
        print("=" * 60)
        print("Starting Bayesian Optimization")
        print("=" * 60)

        start_time = time.time()

        bayesian_optimizer = BayesianOptimizer(
            trainer=self.trainer,
            train_data=(self.train_texts, self.train_labels),
            val_data=(self.val_texts, self.val_labels),
            **self.experiment_config["bayesian"],
        )

        bayesian_results = bayesian_optimizer.optimize()
        execution_time = time.time() - start_time

        bayesian_results["execution_time"] = execution_time
        bayesian_results["method"] = "Bayesian Optimization"
        self.results["bayesian"] = bayesian_results

        print(f"Bayesian Optimization completed in {execution_time:.2f} seconds")
        print(f"Best parameters found: {bayesian_results['best_params']}")
        print(f"Best validation accuracy: {bayesian_results['best_fitness']:.4f}")

    def run_all_experiments(self) -> None:
        """Run all optimization methods and compare results."""
        print("Starting comprehensive hyperparameter optimization experiment...")
        print(f"Dataset size: {self.data_size}")

        experiment_start_time = time.time()

        # Run each optimization method
        for method_name, run_func in [
            ("Genetic Algorithm", self.run_genetic_algorithm),
            ("Particle Swarm Optimization", self.run_particle_swarm_optimization),
            ("Bayesian Optimization", self.run_bayesian_optimization),
        ]:
            try:
                run_func()
            except Exception as e:
                print(f"Error running {method_name}: {e}")

        total_experiment_time = time.time() - experiment_start_time

        self.generate_comparison_report()
        self.save_results()

        print(f"\nExperiment completed in {total_experiment_time:.2f} seconds total")
        print("Results saved to the 'results' directory")

    def generate_comparison_report(self) -> None:
        """Generate a comparison report of all optimization methods."""
        if not self.results:
            print("No results to compare - run optimizations first")
            return

        print("\n" + "=" * 80)
        print("HYPERPARAMETER OPTIMIZATION COMPARISON REPORT")
        print("=" * 80)

        best_accuracy = max(self.results.items(), key=lambda x: x[1]["best_fitness"])
        fastest_method = min(self.results.items(), key=lambda x: x[1]["execution_time"])

        print("\nBest Validation Accuracy:")
        print(f"  {best_accuracy[1]['method']}: {best_accuracy[1]['best_fitness']:.4f}")
        print(f"  Parameters: {best_accuracy[1]['best_params']}")

        print("\nFastest Execution:")
        print(f"  {fastest_method[1]['method']}: {fastest_method[1]['execution_time']:.2f} seconds")

        print("\nDetailed Results:")
        for results in self.results.values():
            print(f"  {results['method']}:")
            print(f"    Accuracy: {results['best_fitness']:.4f}")
            print(f"    Time: {results['execution_time']:.2f}s")
            efficiency = results["best_fitness"] / results["execution_time"]
            print(f"    Efficiency: {efficiency:.6f} accuracy/second")

        print("\nConvergence Analysis:")
        for results in self.results.values():
            if "fitness_history" in results and results["fitness_history"]:
                history = results["fitness_history"]
                improvement = history[-1] - history[0] if len(history) > 1 else 0
                print(
                    f"  {results['method']}: {improvement:.4f} improvement "
                    f"over {len(history)} iterations"
                )

        print("=" * 80)

    def save_results(self) -> None:
        """Save experiment results to JSON file."""
        os.makedirs("results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/hyperopt_results_complete_{timestamp}.json"

        save_data: dict[str, Any] = {}
        for method_name, results in self.results.items():
            save_data[method_name] = {}
            for key, value in results.items():
                if hasattr(value, "tolist"):
                    save_data[method_name][key] = value.tolist()
                else:
                    save_data[method_name][key] = value

        try:
            with open(filename, "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"Results saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main() -> None:
    """Entry point for running the experiment."""
    experiment = HyperparameterOptimizationExperiment(data_size="small")
    experiment.run_all_experiments()


if __name__ == "__main__":
    main()
