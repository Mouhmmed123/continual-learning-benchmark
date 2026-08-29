"""Continual-learning benchmark datasets, models, metrics, and strategies."""

from cl_benchmark.datasets import (
    PermuteTransform,
    Task,
    TaskStream,
    get_benchmark,
)
from cl_benchmark.evaluator import Evaluator
from cl_benchmark.models import MLP, ContinualModel, SimpleConvNet
from cl_benchmark.strategies import (
    EWC,
    BaseStrategy,
    ExemplarBuffer,
    ExperienceReplay,
    Naive,
    Rehearsal,
)
from cl_benchmark.utils import seed_everything
from cl_benchmark.viz import plot_accuracy_trajectories, plot_evaluation_matrix

__all__ = [
    "MLP",
    "SimpleConvNet",
    "ContinualModel",
    "Task",
    "TaskStream",
    "PermuteTransform",
    "get_benchmark",
    "Evaluator",
    "BaseStrategy",
    "EWC",
    "Naive",
    "Rehearsal",
    "ExperienceReplay",
    "ExemplarBuffer",
    "seed_everything",
    "plot_evaluation_matrix",
    "plot_accuracy_trajectories",
]
