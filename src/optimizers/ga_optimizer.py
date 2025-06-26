import random
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing as mp
import logging

# Set up logging to track the evolution process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GAHyperparameterOptimizer:
    """
    Genetic Algorithm for finding optimal BERT hyperparameters.
    This approach evolves a population of hyperparameter sets over multiple
    generations, using selection pressure to find better combinations.
    """

    def __init__(self, trainer, train_data, val_data, population_size=20, generations=10):
        """
        Set up the genetic algorithm with training data and evolution parameters.
        
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

        # Define what hyperparameters we're optimizing
        self.param_space = {
            'learning_rate': (1e-5, 5e-5),      # Continuous range
            'batch_size': [8, 16, 32],           # Discrete choices
            'epochs': [2, 3, 4, 5],              # Discrete choices
            'dropout_rate': (0.1, 0.5),         # Continuous range
            'max_length': [64, 128, 256]         # Discrete choices
        }

        # Set up the genetic algorithm components
        self.setup_deap()
        
        # Track how fitness improves over generations
        self.fitness_history = []

    def setup_deap(self):
        """
        Configure DEAP (Distributed Evolutionary Algorithms in Python) for our problem.
        This sets up the genetic operators and data structures.
        """
        # Create the fitness function (we want to maximize accuracy)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Set up the toolbox with our genetic operators
        self.toolbox = base.Toolbox()

        # Define how to generate each hyperparameter
        self.toolbox.register("learning_rate", random.uniform, *self.param_space['learning_rate'])
        self.toolbox.register("batch_size", random.choice, self.param_space['batch_size'])
        self.toolbox.register("epochs", random.choice, self.param_space['epochs'])
        self.toolbox.register("dropout_rate", random.uniform, *self.param_space['dropout_rate'])
        self.toolbox.register("max_length", random.choice, self.param_space['max_length'])

        # Define how to create a complete individual (set of hyperparameters)
        self.toolbox.register("individual",
                              tools.initCycle,
                              creator.Individual,
                              (
                                self.toolbox.learning_rate,
                                self.toolbox.batch_size,
                                self.toolbox.epochs,
                                self.toolbox.dropout_rate,
                                self.toolbox.max_length
                              ),
                              n=1
                            )
        
        # Define how to create a population
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register the genetic operators
        self.toolbox.register('evaluate', self.evaluate_individual)
        self.toolbox.register('mate', self.crossover)
        self.toolbox.register('mutate', self.mutate)
        self.toolbox.register('select', tools.selTournament, tournsize=3)

    def evaluate_individual(self, individual):
        """
        Test how good a specific set of hyperparameters is by training BERT.
        
        Args:
            individual: List of hyperparameters [learning_rate, batch_size, epochs, dropout_rate, max_length]
            
        Returns:
            tuple: Fitness value (accuracy) as a tuple
        """
        try:
            learning_rate, batch_size, epochs, dropout_rate, max_length = individual

            # Make sure all parameters are the right type
            batch_size = int(batch_size)
            epochs = int(epochs)
            max_length = int(max_length)
            
            # Clean up the data before training
            train_texts = []
            train_labels = []
            val_texts = []
            val_labels = []
            
            # Process training data - only keep valid pairs
            for i in range(min(len(self.train_texts), len(self.train_labels))):
                text = self.train_texts[i]
                label = self.train_labels[i]
                
                if isinstance(text, str) and text.strip():
                    train_texts.append(text.strip())
                    try:
                        train_labels.append(int(label))
                    except (ValueError, TypeError):
                        train_labels.append(0)
            
            # Process validation data the same way
            for i in range(min(len(self.val_texts), len(self.val_labels))):
                text = self.val_texts[i]
                label = self.val_labels[i]
                
                if isinstance(text, str) and text.strip():
                    val_texts.append(text.strip())
                    try:
                        val_labels.append(int(label))
                    except (ValueError, TypeError):
                        val_labels.append(0)

            logger.info(f"Testing individual: lr={learning_rate:.2e}, batch_size={batch_size}, "
                       f"epochs={epochs}, dropout={dropout_rate:.3f}, max_length={max_length}")

            # Train the model with these hyperparameters
            results = self.trainer.train_and_evaluate(
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                dropout_rate=dropout_rate,
                max_length=max_length
            )

            accuracy = results['accuracy']
            logger.info(f"Individual achieved accuracy: {accuracy:.4f}")
            
            return (accuracy,)
            
        except Exception as e:
            logger.error(f"Error evaluating individual: {e}")
            return (0.0,)  # Return poor fitness for failed individuals
        
    def crossover(self, ind1, ind2):
        """
        Create two offspring by combining traits from two parent individuals.
        Uses different strategies for continuous vs discrete parameters.
        
        Args:
            ind1: First parent individual
            ind2: Second parent individual
            
        Returns:
            tuple: Two offspring individuals
        """
        # For continuous parameters (learning rate, dropout), use blending
        alpha = random.random()

        # Blend learning rates
        new_lr1 = alpha * ind1[0] + (1 - alpha) * ind2[0]
        new_lr2 = alpha * ind2[0] + (1 - alpha) * ind1[0]

        # Blend dropout rates
        new_dr1 = alpha * ind1[3] + (1 - alpha) * ind2[3]
        new_dr2 = alpha * ind2[3] + (1 - alpha) * ind1[3]

        # For discrete parameters, randomly swap
        if random.random() < 0.5:
            ind1[1], ind2[1] = ind2[1], ind1[1]  # Batch size
        if random.random() < 0.5:
            ind1[2], ind2[2] = ind2[2], ind1[2]  # Epochs
        if random.random() < 0.5:
            ind1[4], ind2[4] = ind2[4], ind1[4]  # Max length

        # Apply the blended continuous parameters
        ind1[0], ind2[0] = new_lr1, new_lr2
        ind1[3], ind2[3] = new_dr1, new_dr2

        return ind1, ind2
    
    def mutate(self, individual, mutation_rate=0.2):
        """
        Randomly modify an individual to introduce variation.
        Each parameter has a chance to be mutated.
        
        Args:
            individual: The individual to potentially mutate
            mutation_rate: Probability of mutating each parameter
            
        Returns:
            tuple: The (possibly modified) individual
        """
        # Mutate learning rate
        if random.random() < mutation_rate:
            individual[0] = random.uniform(*self.param_space['learning_rate'])
        
        # Mutate batch size
        if random.random() < mutation_rate:
            individual[1] = random.choice(self.param_space['batch_size'])
        
        # Mutate number of epochs
        if random.random() < mutation_rate:
            individual[2] = random.choice(self.param_space['epochs'])
        
        # Mutate dropout rate
        if random.random() < mutation_rate:
            individual[3] = random.uniform(*self.param_space['dropout_rate'])
        
        # Mutate max length
        if random.random() < mutation_rate:
            individual[4] = random.choice(self.param_space['max_length'])
        
        return individual,

    def optimize(self):
        """
        Run the genetic algorithm to evolve good hyperparameters.
        
        Returns:
            dict: Results including the best parameters found and evolution history
        """
        logger.info("Starting genetic algorithm optimization...")
        logger.info(f"Population size: {self.population_size}, Generations: {self.generations}")
        
        # Create the initial random population
        population = self.toolbox.population(n=self.population_size)
        
        # Evaluate the initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Track the best fitness in each generation
        self.fitness_history = []
        
        # Evolution loop
        for generation in range(self.generations):
            logger.info(f"Generation {generation + 1}/{self.generations}")
            
            # Select parents for the next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover to create new individuals
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:  # 70% crossover probability
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation to introduce variation
            for mutant in offspring:
                if random.random() < 0.3:  # 30% mutation probability
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate any new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace the old population
            population[:] = offspring
            
            # Record the best fitness this generation
            fits = [ind.fitness.values[0] for ind in population]
            best_gen_fitness = max(fits)
            self.fitness_history.append(best_gen_fitness)
            
            logger.info(f"Generation {generation + 1} best fitness: {best_gen_fitness:.4f}")
        
        # Find the overall best individual
        best_individual = tools.selBest(population, 1)[0]
        best_params = {
            'learning_rate': best_individual[0],
            'batch_size': int(best_individual[1]),
            'epochs': int(best_individual[2]),
            'dropout_rate': best_individual[3],
            'max_length': int(best_individual[4])
        }
        
        logger.info("Genetic algorithm optimization completed!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best fitness: {best_individual.fitness.values[0]:.4f}")
        
        return {
            'best_params': best_params,
            'best_fitness': best_individual.fitness.values[0],
            'fitness_history': self.fitness_history
        }