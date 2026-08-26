"""Event-driven base strategy interface for continual learning algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from cl_benchmark.datasets import Task, TaskStream


class BaseStrategy(ABC):
    """Abstract base class establishing the lifecycle for continual learning strategies.

    Lifecycle hooks enable advanced CL methods (e.g., EWC, LwF, GEM, Rehearsal)
    to inject custom loss penalties, memory replay, and gradient constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the strategy with model, optimizer, loss criterion, and device.

        Args:
            model: Continual learning model or PyTorch module.
            optimizer: PyTorch optimizer instance.
            criterion: Loss function (defaults to nn.CrossEntropyLoss()).
            device: Compute device ('cpu', 'cuda', 'mps', or torch.device).
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    # --- Strategy Lifecycle Hooks ---

    def before_training_task(self, task: Task) -> None:
        """Hook executed immediately before starting training on a task.

        Useful for registering dynamic heads, initializing task-specific buffers,
        or snapshotting parameters.
        """
        pass

    def before_training_epoch(self, epoch: int, task: Task) -> None:
        """Hook executed at the start of each training epoch."""
        pass

    def before_backward(self, loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """Hook executed before backward pass to augment loss with regularization terms.

        Args:
            loss: Task classification loss tensor.
            **kwargs: Optional batch context (e.g. batch_idx, task, inputs).

        Returns:
            Augmented total loss tensor to backpropagate.
        """
        return loss

    def after_backward(self, task: Task, **kwargs) -> None:
        """Hook executed after loss.backward() and before optimizer.step().

        Useful for gradient modification (e.g., GEM projection, gradient clipping).
        """
        pass

    def after_training_epoch(self, epoch: int, task: Task) -> None:
        """Hook executed at the end of each training epoch."""
        pass

    def after_training_task(self, task: Task) -> None:
        """Hook executed immediately after completing training on a task.

        Useful for Fisher matrix computation (EWC) or updating replay buffers.
        """
        pass

    # --- Training & Evaluation Execution ---

    def train_task(self, task: Task, epochs: int = 1) -> list[float]:
        """Train the model on a task across epochs executing all lifecycle hooks.

        Args:
            task: Task descriptor containing task_id and data loaders.
            epochs: Number of training epochs to run.

        Returns:
            List of average loss values per epoch.
        """
        self.before_training_task(task)
        epoch_losses: list[float] = []

        for epoch in range(epochs):
            self.before_training_epoch(epoch, task)
            avg_loss = self.train_epoch(task)
            epoch_losses.append(avg_loss)
            self.after_training_epoch(epoch, task)

        self.after_training_task(task)
        return epoch_losses

    @abstractmethod
    def train_epoch(self, task: Task) -> float:
        """Train the model for a single epoch on the given task.

        Args:
            task: Active training task.

        Returns:
            Average loss across all batches in the epoch.
        """
        pass

    def evaluate(self, task: Task) -> float:
        """Evaluate classification accuracy on the test set of the given task.

        Args:
            task: Task containing test data loader.

        Returns:
            Top-1 accuracy as a float in range [0.0, 1.0].
        """
        self.model.eval()
        correct = 0
        total = 0

        # Import locally to avoid circular dependencies
        from cl_benchmark.models import ContinualModel

        with torch.no_grad():
            for x, y in task.test_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if isinstance(self.model, ContinualModel):
                    logits = self.model(x, task_id=task.task_id)
                    targets = self.model.to_local_targets(y, task_id=task.task_id)
                else:
                    logits = self.model(x)
                    targets = y

                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += y.size(0)

        return correct / total if total > 0 else 0.0

    def evaluate_all(
        self, stream: TaskStream, max_task_id: int | None = None
    ) -> dict[int, float]:
        """Evaluate the model across tasks up to max_task_id in the stream.

        Args:
            stream: Task stream benchmark.
            max_task_id: Highest task index to evaluate. If None, evaluates all tasks.

        Returns:
            Dictionary mapping task_id to its test accuracy.
        """
        results = {}
        for task in stream:
            if max_task_id is not None and task.task_id > max_task_id:
                break
            results[task.task_id] = self.evaluate(task)
        return results
