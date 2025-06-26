import numpy as np
import random
from pyswarms import single as ps
from pyswarms.utils.functions import single_obj as fx

class PSOHyperparameterOptimizer:
    """
    Particle Swarm Optimization for finding optimal BERT hyperparameters.
    Each particle represents a set of hyperparameters, and the swarm collectively
    searches for the best combination by sharing information about good solutions.
    """

    def __init__(self, trainer, train_data, val_data, n_particles=15, n_iterations=10):
        self.trainer = trainer
        self.train_texts, self.train_labels = train_data
        self.val_texts, self.val_labels = val_data
        self.n_particles = n_particles
        self.n_iterations = n_iterations

        # Define the search space - PSO works with continuous values,
        # so we'll map discrete parameters to continuous ranges
        self.param_bounds = {
            'learning_rate': (1e-5, 5e-5),
            'batch_size_idx': (0, 2.99),     # Will map to [8, 16, 32]
            'epochs_idx': (2, 3.99),         # Will map to [2, 3, 4, 5] 
            'dropout_rate': (0.1, 0.5),
            'max_length_idx': (0, 2.99)      # Will map to [64, 128, 256]
        }
        
        # Discrete parameter mappings
        self.batch_size_options = [8, 16, 32]
        self.epochs_options = [2, 3, 4, 5]
        self.max_length_options = [64, 128, 256]

        # PSO behavior parameters - these control how particles move
        self.pso_options = {
            'c1': 0.5,  # How much each particle trusts its own best result
            'c2': 0.3,  # How much each particle trusts the swarm's best result
            'w': 0.9,   # Inertia weight - how much momentum particles keep
        }

        self.fitness_history = []

    def decode_particle(self, particle):
        """
        Convert a particle's continuous position into actual hyperparameters.
        This handles the mapping from continuous PSO space to our discrete choices.
        """
        learning_rate = particle[0]
        batch_size = self.batch_size_options[int(particle[1] % len(self.batch_size_options))]
        epochs = self.epochs_options[int(particle[2] % len(self.epochs_options))]
        dropout_rate = particle[3]
        max_length = self.max_length_options[int(particle[4] % len(self.max_length_options))]

        return {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'dropout_rate': dropout_rate,
            'max_length': max_length
        }
    
    def fitness_function(self, particles):
        """
        Evaluate how good each particle's hyperparameters are.
        This is the core function that PSO uses to judge particle quality.
        """
        fitness_values = []

        for particle in particles:
            try:
                # Convert particle position to actual hyperparameters
                params = self.decode_particle(particle)

                # Clean up the training data to avoid issues
                train_texts = []
                train_labels = []
                val_texts = []
                val_labels = []
                
                # Process training data - keep only valid text-label pairs
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
                
                # Train the model with these hyperparameters
                results = self.trainer.train_and_evaluate(
                    train_texts,
                    train_labels,
                    val_texts,
                    val_labels,
                    **params
                )

                # PSO minimizes, so we use negative accuracy
                fitness = -results['accuracy']
                fitness_values.append(fitness)

                print(f"Particle parameters: {params}")
                print(f"Achieved accuracy: {results['accuracy']:.4f}")

            except Exception as e:
                print(f"Error evaluating particle: {e}")
                fitness_values.append(0.0)  # Poor fitness for failed evaluations

        return np.array(fitness_values)

    def optimize(self):
        """
        Run the particle swarm optimization to find the best hyperparameters.
        
        Returns:
            dict: Results including best parameters and convergence history
        """
        # Set up the search boundaries for all parameters
        bounds = (
            np.array([
                self.param_bounds['learning_rate'][0],
                self.param_bounds['batch_size_idx'][0],
                self.param_bounds['epochs_idx'][0],
                self.param_bounds['dropout_rate'][0],
                self.param_bounds['max_length_idx'][0]
            ]),
            np.array([
                self.param_bounds['learning_rate'][1],
                self.param_bounds['batch_size_idx'][1],
                self.param_bounds['epochs_idx'][1],
                self.param_bounds['dropout_rate'][1],
                self.param_bounds['max_length_idx'][1]
            ])
        )
        
        # Create the PSO optimizer
        optimizer = ps.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=5,  # We have 5 hyperparameters to optimize
            options=self.pso_options,
            bounds=bounds
        )

        # Run the optimization process
        best_cost, best_pos = optimizer.optimize(
            self.fitness_function,
            iters=self.n_iterations,
            verbose=True
        )

        # Convert the best result back to actual hyperparameters
        best_params = self.decode_particle(best_pos)
        best_accuracy = -best_cost  # Convert back from negative

        # Extract the convergence history
        convergence_history = optimizer.cost_history
        fitness_history = [-cost for cost in convergence_history]  # Convert to positive accuracies

        print(f"\nPSO Optimization Results:")
        print(f"Best parameters found: {best_params}")
        print(f"Best accuracy achieved: {best_accuracy:.4f}")

        return {
            'best_params': best_params,
            'best_fitness': best_accuracy,
            'fitness_history': fitness_history
        }
