import os
import json
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Import our custom modules
from bert_classifier import BertTrainer
from optimizers.ga_optimizer import GAHyperparameterOptimizer
from optimizers.pso_optimizer import PSOHyperparameterOptimizer
from optimizers.bayesian_optimizer import BayesianOptimizer

class HyperparameterOptimizationExperiment:
    """
    Complete framework for comparing different hyperparameter optimization methods.
    This class handles data loading, runs all optimization algorithms, and collects results.
    """

    def __init__(self, data_size='small'):
        """
        Set up the experiment with the specified dataset size.

        Args:
            data_size: Size of dataset to use - 'small' (1000 samples), 
                      'medium' (5000 samples), or 'large' (full dataset)
        """
        self.data_size = data_size
        self.results = {}
        self.trainer = BertTrainer()

        # Configure how many iterations each optimizer will run
        # These are kept small for quick testing - increase for production runs
        self.experiment_config = {
            'ga': {
                'population_size': 8,
                'generations': 5
            },
            'pso': {
                'n_particles': 8,
                'n_iterations': 5
            },
            'bayesian': {
                'n_calls': 15
            }
        }

        print(f"Experiment initialized with dataset size: {data_size}")
        self.load_data()

    def load_data(self):
        """
        Load and prepare the IMDB dataset for sentiment analysis.
        Falls back to synthetic data if the real dataset isn't available.
        """
        print("Loading IMDB dataset...")

        try:
            # Load the IMDB dataset from Hugging Face
            dataset = load_dataset("imdb")

            # Extract the text and labels
            train_texts = dataset['train']['text']
            train_labels = dataset['train']['label']
            test_texts = dataset['test']['text']
            test_labels = dataset['test']['label']

            # Decide how much data to use based on experiment size
            if self.data_size == 'small':
                sample_size = 1000
            elif self.data_size == 'medium':
                sample_size = 5000
            else:
                sample_size = len(train_texts)  # Use all data

            # Sample the data if needed
            if sample_size < len(train_texts):
                # Randomly select samples to keep things interesting
                indices = np.random.choice(len(train_texts), sample_size, replace=False)
                train_texts = [train_texts[i] for i in indices]
                train_labels = [train_labels[i] for i in indices]

            # Split training data into train and validation sets
            self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(
                train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
            )

            # Prepare test set (keeping it smaller for quicker evaluation)
            if sample_size < len(test_texts):
                test_indices = np.random.choice(len(test_texts), min(sample_size, 1000), replace=False)
                self.test_texts = [test_texts[i] for i in test_indices]
                self.test_labels = [test_labels[i] for i in test_indices]
            else:
                self.test_texts = test_texts[:1000]  # Limit test set size
                self.test_labels = test_labels[:1000]

            print(f"Dataset loaded successfully!")
            print(f"Training samples: {len(self.train_texts)}")
            print(f"Validation samples: {len(self.val_texts)}")
            print(f"Test samples: {len(self.test_texts)}")

        except Exception as e:
            print(f"Error loading IMDB dataset: {e}")
            print(f"Creating synthetic data for testing...")
            self.create_dummy_data()

    def create_dummy_data(self):
        """
        Create synthetic sentiment analysis data for testing when real data isn't available.
        This helps ensure the experiment can run even without internet access.
        """
        print("Creating synthetic sentiment data...")

        # Create some simple positive and negative examples
        positive_examples = [
            "This movie is excellent and amazing!",
            "I love this film, it's fantastic!",
            "Great acting and wonderful story!",
            "Best movie I've ever seen!",
            "Brilliant and entertaining film!"
        ] * 50  # Repeat to get more samples
        
        negative_examples = [
            "This movie is terrible and boring.",
            "I hate this film, it's awful!",
            "Poor acting and bad story.",
            "Worst movie I've ever seen!",
            "Horrible and disappointing film."
        ] * 50

        # Combine and shuffle the data
        all_texts = positive_examples + negative_examples
        all_labels = [1] * len(positive_examples) + [0] * len(negative_examples)

        # Randomly shuffle everything
        combined = list(zip(all_texts, all_labels))
        np.random.shuffle(combined)
        all_texts, all_labels = zip(*combined)

        # Split into train/validation/test sets
        train_split = int(len(all_texts) * 0.6)
        val_split = int(len(all_texts) * 0.8)

        self.train_texts = all_texts[:train_split]
        self.train_labels = all_labels[:train_split]
        self.val_texts = all_texts[train_split:val_split]
        self.val_labels = all_labels[train_split:val_split]
        self.test_texts = all_texts[val_split:]
        self.test_labels = all_labels[val_split:]

        print(f"Synthetic data created successfully!")
        print(f"Training samples: {len(self.train_texts)}")
        print(f"Validation samples: {len(self.val_texts)}")
        print(f"Test samples: {len(self.test_texts)}")

    def run_genetic_algorithm(self):
        """
        Run the genetic algorithm optimization and collect results.
        """
        print("=" * 60)
        print("Starting Genetic Algorithm Optimization")
        print("=" * 60)

        start_time = time.time()

        # Set up the genetic algorithm optimizer
        ga_optimizer = GAHyperparameterOptimizer(
            trainer=self.trainer,
            train_data=(self.train_texts, self.train_labels),
            val_data=(self.val_texts, self.val_labels),
            **self.experiment_config['ga']
        )

        # Run the optimization
        ga_results = ga_optimizer.optimize()

        # Record timing and add method info
        execution_time = time.time() - start_time
        ga_results['execution_time'] = execution_time
        ga_results['method'] = 'Genetic Algorithm'

        self.results['ga'] = ga_results

        print(f"Genetic Algorithm completed in {execution_time:.2f} seconds")
        print(f"Best parameters found: {ga_results['best_params']}")
        print(f"Best validation accuracy: {ga_results['best_fitness']:.4f}")

    def run_particle_swarm_optimization(self):
        """
        Run the particle swarm optimization and collect results.
        """
        print("=" * 60)
        print("Starting Particle Swarm Optimization")
        print("=" * 60)

        start_time = time.time()

        # Set up the PSO optimizer
        pso_optimizer = PSOHyperparameterOptimizer(
            trainer=self.trainer,
            train_data=(self.train_texts, self.train_labels),
            val_data=(self.val_texts, self.val_labels),
            **self.experiment_config['pso']
        )

        # Run the optimization
        pso_results = pso_optimizer.optimize()

        # Record timing and add method info
        execution_time = time.time() - start_time
        pso_results['execution_time'] = execution_time
        pso_results['method'] = 'Particle Swarm Optimization'

        self.results['pso'] = pso_results

        print(f"Particle Swarm Optimization completed in {execution_time:.2f} seconds")
        print(f"Best parameters found: {pso_results['best_params']}")
        print(f"Best validation accuracy: {pso_results['best_fitness']:.4f}")

    def run_bayesian_optimization(self):
        """
        Run the Bayesian optimization and collect results.
        """
        print("=" * 60)
        print("Starting Bayesian Optimization")
        print("=" * 60)

        start_time = time.time()

        # Set up the Bayesian optimizer
        bayesian_optimizer = BayesianOptimizer(
            trainer=self.trainer,
            train_data=(self.train_texts, self.train_labels),
            val_data=(self.val_texts, self.val_labels),
            **self.experiment_config['bayesian']
        )

        # Run the optimization
        bayesian_results = bayesian_optimizer.optimize()

        # Record timing and add method info
        execution_time = time.time() - start_time
        bayesian_results['execution_time'] = execution_time
        bayesian_results['method'] = 'Bayesian Optimization'

        self.results['bayesian'] = bayesian_results

        print(f"Bayesian Optimization completed in {execution_time:.2f} seconds")
        print(f"Best parameters found: {bayesian_results['best_params']}")
        print(f"Best validation accuracy: {bayesian_results['best_fitness']:.4f}")

    def run_all_experiments(self):
        """
        Run all three optimization methods and compare their results.
        This is the main function to call for a complete comparison.
        """
        print("Starting comprehensive hyperparameter optimization experiment...")
        print(f"Dataset size: {self.data_size}")
        
        experiment_start_time = time.time()

        # Run each optimization method
        try:
            self.run_genetic_algorithm()
        except Exception as e:
            print(f"Error running Genetic Algorithm: {e}")
            
        try:
            self.run_particle_swarm_optimization()
        except Exception as e:
            print(f"Error running Particle Swarm Optimization: {e}")
            
        try:
            self.run_bayesian_optimization()
        except Exception as e:
            print(f"Error running Bayesian Optimization: {e}")

        total_experiment_time = time.time() - experiment_start_time

        # Generate comparison report
        self.generate_comparison_report()
        
        # Save all results
        self.save_results()

        print(f"\nExperiment completed in {total_experiment_time:.2f} seconds total")
        print("Results saved to the 'results' directory")

    def generate_comparison_report(self):
        """
        Create a comprehensive comparison of all optimization methods.
        This analyzes which method performed best and why.
        """
        if not self.results:
            print("No results to compare - make sure to run the optimizations first")
            return

        print("\n" + "=" * 80)
        print("HYPERPARAMETER OPTIMIZATION COMPARISON REPORT")
        print("=" * 80)

        # Find the best performer in each category
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['best_fitness'])
        fastest_method = min(self.results.items(), key=lambda x: x[1]['execution_time'])

        print(f"\nBest Validation Accuracy:")
        print(f"  {best_accuracy[1]['method']}: {best_accuracy[1]['best_fitness']:.4f}")
        print(f"  Parameters: {best_accuracy[1]['best_params']}")

        print(f"\nFastest Execution:")
        print(f"  {fastest_method[1]['method']}: {fastest_method[1]['execution_time']:.2f} seconds")

        print(f"\nDetailed Results:")
        for method_name, results in self.results.items():
            print(f"  {results['method']}:")
            print(f"    Accuracy: {results['best_fitness']:.4f}")
            print(f"    Time: {results['execution_time']:.2f}s")
            print(f"    Efficiency: {results['best_fitness']/results['execution_time']:.6f} accuracy/second")

        # Analyze convergence patterns
        print(f"\nConvergence Analysis:")
        for method_name, results in self.results.items():
            if 'fitness_history' in results and results['fitness_history']:
                history = results['fitness_history']
                improvement = history[-1] - history[0] if len(history) > 1 else 0
                print(f"  {results['method']}: {improvement:.4f} improvement over {len(history)} iterations")

        print("=" * 80)

    def save_results(self):
        """
        Save all experimental results to a JSON file with timestamp.
        This makes it easy to analyze results later or share with others.
        """
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/hyperopt_results_complete_{timestamp}.json"

        # Prepare results for saving (handle any non-serializable data)
        save_data = {}
        for method_name, results in self.results.items():
            save_data[method_name] = {}
            for key, value in results.items():
                # Convert numpy arrays and other non-JSON types to lists
                if hasattr(value, 'tolist'):
                    save_data[method_name][key] = value.tolist()
                else:
                    save_data[method_name][key] = value

        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"Results saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {e}") 