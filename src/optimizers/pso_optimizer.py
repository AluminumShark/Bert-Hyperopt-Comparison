"""Particle Swarm Optimization for BERT hyperparameter tuning.

This module implements PSO where each particle represents a set of
hyperparameters, and the swarm collectively searches for optimal configurations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from pyswarms import single as ps

if TYPE_CHECKING:
    from src.bert_classifier import BertTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PSOHyperparameterOptimizer:
    """Particle Swarm Optimization for finding optimal BERT hyperparameters.

    Each particle represents a set of hyperparameters, and the swarm
    collectively searches for the best combination.
    """

    def __init__(
        self,
        trainer: BertTrainer,
        train_data: tuple[list, list],
        val_data: tuple[list, list],
        n_particles: int = 15,
        n_iterations: int = 10,
    ) -> None:
        """Initialize the PSO optimizer.

        Args:
            trainer: The BERT trainer instance
            train_data: Training texts and labels as a tuple
            val_data: Validation texts and labels as a tuple
            n_particles: Number of particles in the swarm
            n_iterations: Number of iterations to run
        """
        self.trainer = trainer
        self.train_texts, self.train_labels = train_data
        self.val_texts, self.val_labels = val_data
        self.n_particles = n_particles
        self.n_iterations = n_iterations

        # Search bounds (PSO uses continuous space)
        self.param_bounds = {
            "learning_rate": (1e-5, 5e-5),
            "batch_size_idx": (0, 2.99),
            "epochs_idx": (0, 3.99),
            "dropout_rate": (0.1, 0.5),
            "max_length_idx": (0, 2.99),
        }

        # Discrete parameter mappings
        self.batch_size_options = [8, 16, 32]
        self.epochs_options = [2, 3, 4, 5]
        self.max_length_options = [64, 128, 256]

        # PSO behavior parameters
        self.pso_options = {
            "c1": 0.5,  # Cognitive parameter
            "c2": 0.3,  # Social parameter
            "w": 0.9,  # Inertia weight
        }

        self.fitness_history: list[float] = []

    def _decode_particle(self, particle: np.ndarray) -> dict[str, Any]:
        """Convert particle position to hyperparameters.

        Args:
            particle: Particle position array

        Returns:
            Dictionary of hyperparameters
        """
        batch_size = self.batch_size_options[int(particle[1]) % len(self.batch_size_options)]
        epochs = self.epochs_options[int(particle[2]) % len(self.epochs_options)]
        max_length = self.max_length_options[int(particle[4]) % len(self.max_length_options)]

        return {
            "learning_rate": particle[0],
            "batch_size": batch_size,
            "epochs": epochs,
            "dropout_rate": particle[3],
            "max_length": max_length,
        }

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

    def _fitness_function(self, particles: np.ndarray) -> np.ndarray:
        """Evaluate fitness of all particles.

        Args:
            particles: Array of particle positions

        Returns:
            Array of fitness values (negated accuracy for minimization)
        """
        fitness_values = []

        for particle in particles:
            try:
                params = self._decode_particle(particle)

                # Clean data
                train_texts, train_labels = self._clean_data(self.train_texts, self.train_labels)
                val_texts, val_labels = self._clean_data(self.val_texts, self.val_labels)

                # Train and evaluate
                results = self.trainer.train_and_evaluate(
                    train_texts, train_labels, val_texts, val_labels, **params
                )

                # PSO minimizes, so negate accuracy
                fitness = -results["accuracy"]
                fitness_values.append(fitness)

                logger.info("Particle params: %s", params)
                logger.info("Achieved accuracy: %.4f", results["accuracy"])

            except Exception as e:
                logger.exception("Error evaluating particle: %s", e)
                fitness_values.append(0.0)

        return np.array(fitness_values)

    def optimize(self) -> dict[str, Any]:
        """Run PSO optimization.

        Returns:
            Dictionary with best parameters and optimization history
        """
        logger.info("Starting PSO optimization...")
        logger.info("Particles: %d, Iterations: %d", self.n_particles, self.n_iterations)

        # Define bounds
        bounds = (
            np.array(
                [
                    self.param_bounds["learning_rate"][0],
                    self.param_bounds["batch_size_idx"][0],
                    self.param_bounds["epochs_idx"][0],
                    self.param_bounds["dropout_rate"][0],
                    self.param_bounds["max_length_idx"][0],
                ]
            ),
            np.array(
                [
                    self.param_bounds["learning_rate"][1],
                    self.param_bounds["batch_size_idx"][1],
                    self.param_bounds["epochs_idx"][1],
                    self.param_bounds["dropout_rate"][1],
                    self.param_bounds["max_length_idx"][1],
                ]
            ),
        )

        # Create optimizer
        optimizer = ps.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=5,
            options=self.pso_options,
            bounds=bounds,
        )

        # Run optimization
        best_cost, best_pos = optimizer.optimize(
            self._fitness_function,
            iters=self.n_iterations,
            verbose=True,
        )

        # Decode results
        best_params = self._decode_particle(best_pos)
        best_accuracy = -best_cost

        # Extract convergence history
        convergence_history = optimizer.cost_history
        fitness_history = [-cost for cost in convergence_history]

        logger.info("\nPSO Optimization Results:")
        logger.info("Best parameters: %s", best_params)
        logger.info("Best accuracy: %.4f", best_accuracy)

        return {
            "best_params": best_params,
            "best_fitness": best_accuracy,
            "fitness_history": fitness_history,
        }


if __name__ == "__main__":
    logger.info("PSO Optimizer module loaded successfully!")
