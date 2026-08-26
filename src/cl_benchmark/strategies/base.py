from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.optim import Optimizer
from cl_benchmark.datasets import Task


class BaseStrategy(ABC):
    """Abstract base class for all continual learning strategies."""

    def __init__(
        self, model: nn.Module, optimizer: Optimizer, device: str = "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = str(device)

    def before_training_task(self, task: Task) -> None:
        pass

    def after_training_task(self, task: Task) -> None:
        pass

    def before_backward(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    @abstractmethod
    def train_epoch(self, task: Task) -> float:
        pass

    @abstractmethod
    def evaluate(self, task: Task) -> float:
        pass