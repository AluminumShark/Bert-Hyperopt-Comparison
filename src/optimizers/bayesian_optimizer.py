import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
import matplotlib.pyplot as plt

# Set up logging to track optimization progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """
    Bayesian optimization for finding the best BERT hyperparameters.
    Uses Gaussian processes to model the objective function and smart acquisition
    functions to decide which hyperparameters to try next.
    """
    def __init__(self,
                 trainer,
                 train_data: Tuple[List, List],
                 val_data: Tuple[List, List],
                 n_calls: int = 30,
                 n_initial_points: int = 8,
                 acq_func: str = 'EI',
                 random_state: int = 42,
                 ):
        """
        Set up the Bayesian optimizer with training data and search parameters.

        Args:
            trainer: The BERT trainer instance to use for model evaluation
            train_data: Training texts and labels as a tuple
            val_data: Validation texts and labels as a tuple
            n_calls: Total number of hyperparameter combinations to try
            n_initial_points: Number of random points to sample before using the model
            acq_func: Acquisition function ('EI', 'LCB', 'PI', etc.)
            random_state: Random seed for reproducible results
        """
        self.trainer = trainer
        self.train_texts, self.train_labels = train_data
        self.val_texts, self.val_labels = val_data
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        # Make sure we have a valid acquisition function
        self.acq_func = acq_func.upper() if isinstance(acq_func, str) else 'EI'
        if self.acq_func not in ['GP_HEDGE', 'EI', 'LCB', 'MES', 'PVRS', 'PI', 'EIPS', 'PIPS']:
            self.acq_func = 'EI'  # Default to Expected Improvement
        self.random_state = random_state

        # Define the search space for hyperparameters
        self.search_space = self._define_search_space()

        # Keep track of all evaluations and best results found so far
        self.evaluation_history = []
        self.best_params_history = []
        self.best_score_history = []

        logger.info(f"Initialized Bayesian optimizer with search space: {self.search_space}")
        logger.info(f"Planning {self.n_initial_points} initial random points + {self.n_calls} total evaluations")
        logger.info(f"Using acquisition function: {self.acq_func}")

    def _define_search_space(self):
        """
        Define the hyperparameter search space for BERT fine-tuning.
        This covers the most important parameters that affect model performance.
        """

        # Based on common BERT fine-tuning best practices
        search_space = [
            # Learning rate - log scale since small changes can have big effects
            Real(1e-5, 5e-5, name='learning_rate', prior='log-uniform'),

            # Batch size - discrete choices based on memory constraints
            Categorical([8, 16, 32], name='batch_size'),

            # Number of training epochs
            Integer(2, 5, name='epochs'),

            # Dropout rate to prevent overfitting
            Real(0.1, 0.5, name='dropout_rate'),

            # Maximum sequence length for tokenization
            Categorical([64, 128, 256], name='max_length'),

            # Learning rate warmup proportion
            Real(0.1, 0.5, name='warmup_ratio')
        ]

        logger.info(f"Search space defined with {len(search_space)} dimensions")
        
        for dim in search_space:
            dim_type = getattr(dim, 'dtype', type(dim).__name__)
            dim_bounds = getattr(dim, 'bounds', getattr(dim, 'categories', 'N/A'))
            logger.info(f"Parameter: {dim.name} - Type: {dim_type} - Range: {dim_bounds}")

        return search_space
    
    def _objective_function(self, **params):
        """
        The function we're trying to maximize - train BERT with given hyperparameters
        and return the validation accuracy.

        Args:
            params: Dictionary of hyperparameters to evaluate

        Returns:
            float: Negative validation accuracy (since skopt minimizes)
        """

        start_time = time.time()

        try:
            logger.info(f"Evaluating hyperparameter set #{len(self.evaluation_history) + 1}")
            logger.info(f"Parameters: {params}")

            # Clean up the data to avoid training issues
            train_texts = []
            train_labels = []
            val_texts = []
            val_labels = []
            
            # Process training data - only keep valid text-label pairs
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
            result = self.trainer.train_and_evaluate(
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                **params
            )

            accuracy = result['accuracy']
            f1_score = result.get('f1_score', 0.0)
            loss = result.get('loss', float('inf'))

            # Record how long this evaluation took
            eval_time = time.time() - start_time

            # Save this evaluation for later analysis
            evaluation_record = {
                'iteration': len(self.evaluation_history) + 1,
                'params': params,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'loss': loss,
                'eval_time': eval_time,
                'timestamp': time.time()
            }
            self.evaluation_history.append(evaluation_record)

            # Keep track of the best parameters found so far
            if not self.best_params_history or accuracy > max(self.best_score_history):
                self.best_params_history.append(params.copy())
                logger.info(f"New best accuracy: {accuracy:.4f} with params: {params}")
            else:
                self.best_params_history.append(self.best_params_history[-1])

            self.best_score_history.append(
                max(self.best_score_history + [accuracy])
            )

            logger.info(f"Results: accuracy={accuracy:.4f}, f1={f1_score:.4f}, loss={loss:.4f}, time={eval_time:.2f}s")

            # Return negative accuracy since skopt minimizes
            return -accuracy
        
        except Exception as e:
            logger.error(f"Error during hyperparameter evaluation: {e}")
            # Return a poor score to avoid stopping the optimization
            return 0.0
        
    def optimize(self) -> Dict[str, Any]:
        """
        Run the Bayesian optimization process to find the best hyperparameters.
        
        Returns:
            dict: Results including best parameters, best score, and optimization history
        """
        logger.info("Starting Bayesian optimization...")
        logger.info(f"Will evaluate {self.n_calls} hyperparameter combinations")

        start_time = time.time()

        try:
            # Run the optimization using scikit-optimize
            result = gp_minimize(
                func=self._objective_function,
                dimensions=self.search_space,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                acq_func=self.acq_func,
                random_state=self.random_state,
                n_jobs=1,  # Keep it single-threaded for stability
                verbose=False
            )

            total_time = time.time() - start_time
            best_accuracy = -result.fun  # Convert back from negative
            
            # Extract the best hyperparameters
            best_params = {}
            for i, param_name in enumerate([dim.name for dim in self.search_space]):
                best_params[param_name] = result.x[i]

            # Get convergence analysis
            convergence_analysis = self._analyze_convergence()

            optimization_results = {
                'best_params': best_params,
                'best_fitness': best_accuracy,
                'fitness_history': [-score for score in result.func_vals],  # Convert to positive
                'execution_time': total_time,
                'n_evaluations': len(result.func_vals),
                'convergence_analysis': convergence_analysis,
                'evaluation_history': self.evaluation_history
            }

            self._print_optimization_summary(optimization_results)
            
            logger.info(f"Bayesian optimization completed in {total_time:.2f} seconds")
            logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
            
            return optimization_results

        except Exception as e:
            logger.error(f"Error during Bayesian optimization: {e}")
            # Return a fallback result
            return {
                'best_params': {},
                'best_fitness': 0.0,
                'fitness_history': [],
                'execution_time': time.time() - start_time,
                'error': str(e)
            }

    def _analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze how well the optimization is converging to good solutions.
        
        Returns:
            dict: Convergence analysis including status and recommendations
        """
        if len(self.best_score_history) < 3:
            return {'status': 'insufficient_data', 'recommendation': 'Need more evaluations'}

        # Check if we're still improving significantly
        recent_improvement = self.best_score_history[-1] - self.best_score_history[-3]
        
        if recent_improvement > 0.01:  # Still improving by more than 1%
            status = 'improving'
        elif recent_improvement > 0.001:  # Small improvements
            status = 'slow_convergence'
        else:
            status = 'converged'

        # Calculate search efficiency
        search_efficiency = self._calculate_search_efficiency()
        
        return {
            'status': status,
            'recent_improvement': recent_improvement,
            'search_efficiency': search_efficiency,
            'recommendation': self._get_convergence_recommendation(status)
        }

    def _get_convergence_recommendation(self, status: str) -> str:
        """Get recommendations based on convergence status."""
        recommendations = {
            'improving': 'Continue optimization - good progress being made',
            'slow_convergence': 'Consider increasing exploration or trying different acquisition function',
            'converged': 'Optimization has likely found a good solution',
            'insufficient_data': 'Need more evaluations to assess convergence'
        }
        return recommendations.get(status, 'Unknown status')

    def _calculate_search_efficiency(self) -> float:
        """
        Calculate how efficiently we're exploring the search space.
        Higher values indicate we're finding good solutions quickly.
        """
        if not self.best_score_history:
            return 0.0
        
        # Measure how quickly we reached our current best
        final_score = self.best_score_history[-1]
        evaluations_to_best = len(self.best_score_history)
        
        return final_score / evaluations_to_best if evaluations_to_best > 0 else 0.0

    def _print_optimization_summary(self, results: Dict[str, Any]):
        """Print a nice summary of the optimization results."""
        logger.info("=" * 60)
        logger.info("BAYESIAN OPTIMIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Best validation accuracy: {results['best_fitness']:.4f}")
        logger.info(f"Total evaluations: {results['n_evaluations']}")
        logger.info(f"Execution time: {results['execution_time']:.2f} seconds")
        logger.info(f"Average time per evaluation: {results['execution_time']/results['n_evaluations']:.2f}s")
        
        logger.info("\nBest hyperparameters found:")
        for param, value in results['best_params'].items():
            if isinstance(value, float):
                logger.info(f"  {param}: {value:.6f}")
            else:
                logger.info(f"  {param}: {value}")
        
        # Show convergence analysis if available
        if 'convergence_analysis' in results:
            conv = results['convergence_analysis']
            logger.info(f"\nConvergence status: {conv['status']}")
            logger.info(f"Recommendation: {conv['recommendation']}")
        
        logger.info("=" * 60)

    def plot_optimization_progress(self, save_path:Optional[str] = None):
        """
        Plot Optimization Progress

        Args:
            save_path: Path to save the plot
        """

        if not self.evaluation_history:
            logger.warning("No evaluation history to plot")
            return

        if len(self.evaluation_history) < 2:
            logger.warning("Insufficient data to plot progress")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Bayesian Optimization Process Analysis', fontsize=16, fontweight='bold')
        
        # 1. Convergence Curve
        ax1 = axes[0, 0]
        iterations = range(1, len(self.best_score_history) + 1)
        ax1.plot(iterations, self.best_score_history, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Accuracy')
        ax1.set_title('Convergence Curve')
        ax1.grid(True, alpha=0.3)
        
        # 2. All Evaluation Results
        ax2 = axes[0, 1]
        all_scores = [record['accuracy'] for record in self.evaluation_history]
        ax2.scatter(range(1, len(all_scores) + 1), all_scores, alpha=0.6, s=30)
        ax2.plot(iterations, self.best_score_history, 'r-', linewidth=2, label='Best Trajectory')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('All Evaluation Results')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter Exploration History (Learning Rate)
        ax3 = axes[1, 0]
        lr_values = [record['params']['learning_rate'] for record in self.evaluation_history]
        colors = all_scores  # Color by accuracy
        scatter = ax3.scatter(range(1, len(lr_values) + 1), lr_values, 
                             c=colors, cmap='viridis', s=50, alpha=0.7)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Exploration Trajectory')
        ax3.set_yscale('log')
        plt.colorbar(scatter, ax=ax3, label='Accuracy')
        
        # 4. Evaluation Time Distribution
        ax4 = axes[1, 1]
        eval_times = [record['eval_time'] for record in self.evaluation_history]
        ax4.hist(eval_times, bins=10, alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(eval_times), color='red', linestyle='--', 
                   label=f'Average: {np.mean(eval_times):.1f}s')
        ax4.set_xlabel('Evaluation Time (seconds)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Evaluation Time Distribution')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Progress plot saved: {save_path}")
        
        plt.show()

    def get_feature_importance(self):
        """
        Analyze Feature Importance
        
        Note: This is based on GP model posterior analysis, not true feature importance
        But it can give intuition about which parameters have the most impact on results
        """
        
        try:
            from skopt.plots import plot_objective
            
            logger.info("Analyzing parameter importance...")
            
            # Use the last optimization result for analysis
            if hasattr(self, '_last_optimization_result'):
                result = self._last_optimization_result
                
                # Plot partial dependence plots
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle('Parameter Importance Analysis (Based on GP Posterior)', fontsize=16)
                
                param_names = [dim.name for dim in self.search_space]
                for i, param_name in enumerate(param_names):
                    if i < 6:  # Display maximum 6 parameters
                        ax = axes[i//3, i%3]
                        plot_objective(result, dimensions=[param_name], ax=ax)
                        ax.set_title(f'Impact of {param_name}')
                
                plt.tight_layout()
                plt.show()
                
            else:
                logger.warning("Need to run optimize() first to analyze feature importance")
                
        except ImportError:
            logger.warning("Need matplotlib to plot feature importance")
    
    def export_results(self, filepath: str):
        """Export detailed results for further analysis"""
        
        import json
        
        export_data = {
            'search_space': [
                {
                    'name': dim.name,
                    'type': type(dim).__name__,
                    'bounds': getattr(dim, 'bounds', None),
                    'categories': getattr(dim, 'categories', None)
                }
                for dim in self.search_space
            ],
            'optimization_config': {
                'n_calls': self.n_calls,
                'n_initial_points': self.n_initial_points,
                'acq_func': self.acq_func,
                'random_state': self.random_state
            },
            'evaluation_history': self.evaluation_history,
            'best_score_history': self.best_score_history,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results exported: {filepath}")

# Usage Examples and Best Practices
class BayesianOptimizationBestPractices:
    """
    Bayesian Optimization Best Practices Guide
    
    This class contains practical experience and techniques for real-world usage
    """
    
    @staticmethod
    def recommend_n_calls(search_space_size: int, time_budget_hours: float) -> int:
        """
        Recommend number of evaluations based on search space and time budget
        
        Rules of thumb:
        - 10-15 evaluations per continuous parameter
        - 5-8 evaluations per discrete parameter
        - Minimum of 15 total evaluations
        """
        
        base_calls = max(15, search_space_size * 8)
        time_limited_calls = int(time_budget_hours * 60 / 5)  # Assume 5 minutes per evaluation
        
        recommended = min(base_calls, time_limited_calls)
        
        logger.info(f"Recommended evaluations: {recommended}")
        logger.info(f"  Based on search space: {base_calls}")
        logger.info(f"  Based on time budget: {time_limited_calls}")
        
        return recommended
    
    @staticmethod
    def choose_acquisition_function(exploration_preference: str) -> str:
        """
        Choose acquisition function based on exploration preference
        
        Args:
            exploration_preference: 'conservative', 'balanced', 'aggressive'
        """
        
        mapping = {
            'conservative': 'LCB',  # Lower Confidence Bound - conservative, low risk selection
            'balanced': 'EI',       # Expected Improvement - balanced, expected improvement
            'aggressive': 'PI'      # Probability of Improvement - aggressive, high risk high reward
        }
        
        return mapping.get(exploration_preference, 'EI')
    
    @staticmethod
    def validate_search_space(search_space, expected_evaluation_time: float):
        """
        Validate the reasonableness of search space design
        """
        
        warnings = []
        
        # Check dimension count
        if len(search_space) > 15:
            warnings.append("Search space dimensionality too high (>15), may affect GP performance")
        
        # Check parameter ranges
        for dim in search_space:
            if hasattr(dim, 'bounds'):
                low, high = dim.bounds
                if high / low > 1000:  # Range too large
                    warnings.append(f"Parameter {dim.name} range too large, consider using log scale")
        
        # Check time budget
        total_time_hours = len(search_space) * 10 * expected_evaluation_time / 3600
        if total_time_hours > 24:
            warnings.append(f"Estimated total time {total_time_hours:.1f} hours, consider reducing evaluations")
        
        if warnings:
            logger.warning("Search space design warnings:")
            for warning in warnings:
                logger.warning(f"  WARNING: {warning}")
        else:
            logger.info("Search space design is reasonable")

# Quick Test Script
if __name__ == "__main__":
    # This would be a simple test script
    logger.info("Bayesian Optimizer module loaded successfully!")
    logger.info("Usage instructions:")
    logger.info("  1. Import BayesianOptimizer")
    logger.info("  2. Prepare trainer and data")
    logger.info("  3. Create optimizer instance")
    logger.info("  4. Call optimize() method")
    logger.info("  5. Analyze results and visualize")