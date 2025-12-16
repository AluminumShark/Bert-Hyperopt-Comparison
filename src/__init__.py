"""BERT Hyperparameter Optimization Comparison Package.

This package provides tools for comparing different hyperparameter optimization
methods (Genetic Algorithm, Particle Swarm Optimization, and Bayesian Optimization)
for BERT fine-tuning on sentiment analysis tasks.
"""

from src.bert_classifier import BERTClassifier, BertTrainer, IMDBDataset
from src.experiment import HyperparameterOptimizationExperiment
from src.visualize import ExperimentVisualizer

__version__ = "1.0.0"
__all__ = [
    "BERTClassifier",
    "BertTrainer",
    "IMDBDataset",
    "HyperparameterOptimizationExperiment",
    "ExperimentVisualizer",
]
