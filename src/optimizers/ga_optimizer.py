"""Genetic Algorithm optimizer for BERT hyperparameter tuning.

This module implements a genetic algorithm that evolves a population
of hyperparameter sets to find optimal configurations.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from deap import base, creator, tools

if TYPE_CHECKING:
    from src.bert_classifier import BertTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GAHyperparameterOptimizer:
    """Genetic Algorithm for finding optimal BERT hyperparameters.

    This approach evolves a population of hyperparameter sets over multiple
    generations, using selection pressure to find better combinations.
    """

    def __init__(
        self,
        trainer: BertTrainer,
        train_data: tuple[list, list],
        val_data: tuple[list, list],
        population_size: int = 20,
        generations: int = 10,
    ) -> None:
        """Initialize the genetic algorithm optimizer.

        Args:
            trainer: The BERT trainer to use for evaluating hyperparameters
            train_data: Training texts and labels as a tuple
            val_data: Validation texts and labels as a tuple
            population_size: Number of individuals in each generation
            generations: Number of generations to evolve
        """
        self.trainer = trainer
        self.train_texts, self.train_labels = train_data
        self.val_texts, self.val_labels = val_data
        self.population_size = population_size
        self.generations = generations

        # Define hyperparameter search space
        self.param_space = {
            "learning_rate": (1e-5, 5e-5),
            "batch_size": [8, 16, 32],
            "epochs": [2, 3, 4, 5],
            "dropout_rate": (0.1, 0.5),
            "max_length": [64, 128, 256],
        }

        self.toolbox: base.Toolbox | None = None
        self.fitness_history: list[float] = []

        self._setup_deap()

    def _setup_deap(self) -> None:
        """Configure DEAP for genetic algorithm optimization."""
        # Create fitness and individual classes if they don't exist
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Register attribute generators
        self.toolbox.register("learning_rate", random.uniform, *self.param_space["learning_rate"])
        self.toolbox.register("batch_size", random.choice, self.param_space["batch_size"])
        self.toolbox.register("epochs", random.choice, self.param_space["epochs"])
        self.toolbox.register("dropout_rate", random.uniform, *self.param_space["dropout_rate"])
        self.toolbox.register("max_length", random.choice, self.param_space["max_length"])

        # Register individual creator
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (
                self.toolbox.learning_rate,
                self.toolbox.batch_size,
                self.toolbox.epochs,
                self.toolbox.dropout_rate,
                self.toolbox.max_length,
            ),
            n=1,
        )

        # Register population creator
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

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

    def _evaluate_individual(self, individual: list) -> tuple[float]:
        """Evaluate a hyperparameter set by training BERT.

        Args:
            individual: List of hyperparameters

        Returns:
            Fitness value (accuracy) as a tuple
        """
        try:
            learning_rate, batch_size, epochs, dropout_rate, max_length = individual

            # Ensure correct types
            batch_size = int(batch_size)
            epochs = int(epochs)
            max_length = int(max_length)

            # Clean data
            train_texts, train_labels = self._clean_data(self.train_texts, self.train_labels)
            val_texts, val_labels = self._clean_data(self.val_texts, self.val_labels)

            logger.info(
                "Testing: lr=%.2e, batch=%d, epochs=%d, dropout=%.3f, max_len=%d",
                learning_rate,
                batch_size,
                epochs,
                dropout_rate,
                max_length,
            )

            results = self.trainer.train_and_evaluate(
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                dropout_rate=dropout_rate,
                max_length=max_length,
            )

            accuracy = results["accuracy"]
            logger.info("Individual accuracy: %.4f", accuracy)

            return (accuracy,)

        except Exception as e:
            logger.exception("Error evaluating individual: %s", e)
            return (0.0,)

    def _crossover(self, ind1: list, ind2: list) -> tuple[list, list]:
        """Create offspring by combining parent traits.

        Args:
            ind1: First parent
            ind2: Second parent

        Returns:
            Two offspring individuals
        """
        alpha = random.random()

        # Blend continuous parameters
        new_lr1 = alpha * ind1[0] + (1 - alpha) * ind2[0]
        new_lr2 = alpha * ind2[0] + (1 - alpha) * ind1[0]
        new_dr1 = alpha * ind1[3] + (1 - alpha) * ind2[3]
        new_dr2 = alpha * ind2[3] + (1 - alpha) * ind1[3]

        # Swap discrete parameters randomly
        if random.random() < 0.5:
            ind1[1], ind2[1] = ind2[1], ind1[1]
        if random.random() < 0.5:
            ind1[2], ind2[2] = ind2[2], ind1[2]
        if random.random() < 0.5:
            ind1[4], ind2[4] = ind2[4], ind1[4]

        # Apply blended values
        ind1[0], ind2[0] = new_lr1, new_lr2
        ind1[3], ind2[3] = new_dr1, new_dr2

        return ind1, ind2

    def _mutate(self, individual: list, mutation_rate: float = 0.2) -> tuple[list]:
        """Randomly modify an individual.

        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutating each parameter

        Returns:
            Mutated individual as a tuple
        """
        if random.random() < mutation_rate:
            individual[0] = random.uniform(*self.param_space["learning_rate"])

        if random.random() < mutation_rate:
            individual[1] = random.choice(self.param_space["batch_size"])

        if random.random() < mutation_rate:
            individual[2] = random.choice(self.param_space["epochs"])

        if random.random() < mutation_rate:
            individual[3] = random.uniform(*self.param_space["dropout_rate"])

        if random.random() < mutation_rate:
            individual[4] = random.choice(self.param_space["max_length"])

        return (individual,)

    def optimize(self) -> dict[str, Any]:
        """Run genetic algorithm optimization.

        Returns:
            Dictionary with best parameters and evolution history
        """
        logger.info("Starting genetic algorithm optimization...")
        logger.info("Population: %d, Generations: %d", self.population_size, self.generations)

        if self.toolbox is None:
            raise RuntimeError("DEAP toolbox not initialized")

        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        self.fitness_history = []

        # Evolution loop
        for generation in range(self.generations):
            logger.info("Generation %d/%d", generation + 1, self.generations)

            # Selection
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < 0.3:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            population[:] = offspring

            # Record best fitness
            fits = [ind.fitness.values[0] for ind in population]
            best_gen_fitness = max(fits)
            self.fitness_history.append(best_gen_fitness)

            logger.info("Generation %d best fitness: %.4f", generation + 1, best_gen_fitness)

        # Get best individual
        best_individual = tools.selBest(population, 1)[0]
        best_params = {
            "learning_rate": best_individual[0],
            "batch_size": int(best_individual[1]),
            "epochs": int(best_individual[2]),
            "dropout_rate": best_individual[3],
            "max_length": int(best_individual[4]),
        }

        logger.info("Genetic algorithm optimization completed!")
        logger.info("Best parameters: %s", best_params)
        logger.info("Best fitness: %.4f", best_individual.fitness.values[0])

        return {
            "best_params": best_params,
            "best_fitness": best_individual.fitness.values[0],
            "fitness_history": self.fitness_history,
        }


if __name__ == "__main__":
    logger.info("GA Optimizer module loaded successfully!")
