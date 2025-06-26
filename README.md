# BERT Hyperparameter Optimization Comparison

A comprehensive comparison of different hyperparameter optimization methods for BERT fine-tuning on sentiment analysis tasks. This project implements and compares three popular optimization algorithms: Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Bayesian Optimization.

## What This Project Does

This project helps you find the best hyperparameters for training BERT models by comparing three different optimization approaches. Instead of manually trying different combinations (which could take forever), these algorithms intelligently search through the hyperparameter space to find configurations that give you the best performance.

**Why This Matters:** Finding good hyperparameters can be the difference between a model that barely works and one that achieves state-of-the-art results. But with so many possible combinations, you need smart search strategies.

## Key Features

- **Three Optimization Methods**: Compare genetic algorithms, particle swarm optimization, and Bayesian optimization side-by-side
- **Real Dataset**: Uses the IMDB movie review dataset for sentiment analysis (with fallback to synthetic data)
- **Comprehensive Analysis**: Detailed visualizations showing convergence, parameter distributions, and performance comparisons
- **Easy to Use**: Run complete experiments with just a few lines of code
- **Extensible Design**: Easy to add new optimization methods or adapt to different tasks

## Project Structure

```
Bert-Hyperopt-Comparison/
├── src/
│   ├── bert_classifier.py           # BERT model and training pipeline
│   ├── experiment.py                # Main experiment runner
│   ├── visualize.py                 # Results visualization tools
│   └── optimizers/
│       ├── bayesian_optimizer.py    # Bayesian optimization implementation
│       ├── ga_optimizer.py          # Genetic algorithm implementation
│       └── pso_optimizer.py         # Particle swarm optimization
├── results/                         # Experiment results (JSON files)
├── visualizations/                  # Generated charts and analysis
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## How It Works

### The BERT Classifier (`src/bert_classifier.py`)
- **Custom Dataset Handler**: Safely loads and validates text data
- **BERT Model**: Uses pre-trained BERT with a classification head
- **Training Pipeline**: Complete training and evaluation with proper error handling

### The Optimization Algorithms

#### Genetic Algorithm (`src/optimizers/ga_optimizer.py`)
Think of this like evolution in action. We start with a population of random hyperparameter combinations, then repeatedly:
- Select the best performers as "parents"
- Create new combinations by mixing parent parameters (crossover)
- Randomly mutate some parameters to explore new possibilities
- Keep the best performers for the next generation

**Best for**: Exploring diverse parameter combinations when you don't have strong prior beliefs about what works.

#### Particle Swarm Optimization (`src/optimizers/pso_optimizer.py`)
Imagine a swarm of particles flying through the hyperparameter space, where each particle represents a different configuration. Particles:
- Remember their own best position
- Are attracted to the swarm's overall best position  
- Have momentum that helps them explore new areas

**Best for**: Balanced exploration and exploitation with good convergence properties.

#### Bayesian Optimization (`src/optimizers/bayesian_optimizer.py`)
This is the "smart" approach that builds a probabilistic model of how hyperparameters affect performance:
- Uses past results to predict which new combinations are most promising
- Balances trying promising areas vs exploring unknown regions
- Gets smarter with each evaluation

**Best for**: When evaluations are expensive and you want to make every trial count.

### Experiment Framework (`src/experiment.py`)

The main `HyperparameterOptimizationExperiment` class handles everything:
- Loads and preprocesses the IMDB dataset (or creates synthetic data)
- Runs all three optimization methods
- Collects timing and performance metrics
- Generates comparison reports
- Saves results for later analysis

### Visualization Suite (`src/visualize.py`)

Creates publication-ready charts including:
- **Convergence plots**: See how each method improves over time
- **Parameter analysis**: Understand which hyperparameters matter most
- **Performance comparison**: Compare accuracy, speed, and efficiency
- **Summary reports**: Executive-level overview of results

## What Hyperparameters Are Optimized

| Parameter | Type | Range | Why It Matters |
|-----------|------|-------|----------------|
| `learning_rate` | Continuous | 1e-5 to 5e-5 | Controls how fast the model learns (too high = unstable, too low = slow) |
| `batch_size` | Discrete | [8, 16, 32] | Affects training stability and GPU memory usage |
| `epochs` | Discrete | [2, 3, 4, 5] | How many times to go through the training data |
| `dropout_rate` | Continuous | 0.1 to 0.5 | Prevents overfitting by randomly ignoring some neurons |
| `max_length` | Discrete | [64, 128, 256] | Maximum text length to process (longer = more context but slower) |
| `warmup_ratio` | Continuous | 0.1 to 0.5 | Gradually increases learning rate at the start (Bayesian only) |

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Bert-Hyperopt-Comparison

# Install dependencies
pip install -r requirements.txt
```

### Running the Complete Experiment

```python
from src.experiment import HyperparameterOptimizationExperiment

# Run a quick comparison with a small dataset
experiment = HyperparameterOptimizationExperiment(data_size='small')
experiment.run_all_experiments()

# Results are automatically saved and visualizations are generated
```

### Running Individual Optimizers

```python
from src.bert_classifier import BertTrainer
from src.optimizers.ga_optimizer import GAHyperparameterOptimizer

# Set up your data and trainer
trainer = BertTrainer()
train_texts = ["This movie is great!", "This movie is terrible.", ...]
train_labels = [1, 0, ...]  # 1 = positive, 0 = negative
val_texts = ["Amazing film!", "Boring movie.", ...]
val_labels = [1, 0, ...]

# Run genetic algorithm optimization
ga_optimizer = GAHyperparameterOptimizer(
    trainer=trainer,
    train_data=(train_texts, train_labels),
    val_data=(val_texts, val_labels),
    population_size=10,
    generations=5
)

results = ga_optimizer.optimize()
print(f"Best accuracy: {results['best_fitness']:.4f}")
print(f"Best parameters: {results['best_params']}")
```

### Generating Visualizations

```python
from src.visualize import ExperimentVisualizer

# Create visualizations from saved results
visualizer = ExperimentVisualizer()
visualizer.create_all_visualizations()

# Individual plots
visualizer.plot_convergence_comparison(save_path='convergence.png')
visualizer.create_performance_radar_chart(save_path='radar.png')
```

## Understanding the Results

### What to Look For

1. **Convergence Speed**: Which method finds good solutions fastest?
2. **Final Performance**: Which method achieves the highest accuracy?
3. **Stability**: Which method gives consistent results across runs?
4. **Efficiency**: Which method gives the best accuracy per unit of time?

### Typical Findings

- **Bayesian Optimization**: Often achieves the best final performance with fewer evaluations
- **Genetic Algorithm**: Good at exploring diverse solutions, sometimes finds unexpected good combinations
- **Particle Swarm**: Balanced performance with good convergence properties

### Tips for Real-World Use

- Start with Bayesian optimization if you have limited time/compute budget
- Use genetic algorithms if you want to explore many diverse solutions
- Try particle swarm optimization as a middle ground
- Always run multiple independent trials to check consistency

## Customization

### Adding New Optimization Methods

1. Create a new optimizer class in `src/optimizers/`
2. Implement the same interface as existing optimizers
3. Add it to the experiment configuration in `src/experiment.py`

### Using Different Datasets

1. Modify the `load_data()` method in `HyperparameterOptimizationExperiment`
2. Ensure your data has the same format (texts and binary labels)
3. Adjust the synthetic data generation if needed

### Changing Hyperparameter Ranges

1. Update the parameter spaces in each optimizer class
2. Make sure the ranges make sense for your specific use case
3. Consider the computational cost of your parameter choices

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- Transformers library (for BERT)
- scikit-optimize (for Bayesian optimization)
- DEAP (for genetic algorithms)
- PySwarms (for particle swarm optimization)
- Various other dependencies (see requirements.txt)

## Contributing

Feel free to:
- Add new optimization algorithms
- Improve the visualization tools
- Add support for different datasets or tasks
- Fix bugs or improve documentation

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{bert-hyperopt-comparison,
  title={BERT Hyperparameter Optimization Comparison},
  author={Levi},
  year={2025},
  url={https://github.com/your-username/Bert-Hyperopt-Comparison}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
