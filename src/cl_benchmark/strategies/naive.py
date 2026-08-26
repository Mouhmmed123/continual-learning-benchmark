"""Naive fine-tuning continual learning baseline strategy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer

from cl_benchmark.datasets import Task
from cl_benchmark.models import ContinualModel
from cl_benchmark.strategies.base import BaseStrategy


class Naive(BaseStrategy):
    """Naive fine-tuning continual learning strategy (sequential training)."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize Naive fine-tuning strategy."""
        super().__init__(
            model=model, optimizer=optimizer, criterion=criterion, device=device
        )

    def train_epoch(self, task: Task) -> float:
        """Train the model for one epoch on the current task.

        Args:
            task: Current training task.

        Returns:
            Average loss across training batches.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(task.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if isinstance(self.model, ContinualModel):
                logits = self.model(x, task_id=task.task_id)
                targets = self.model.to_local_targets(y, task_id=task.task_id)
            else:
                logits = self.model(x)
                targets = y

            loss = self.criterion(logits, targets)
            loss = self.before_backward(loss, batch_idx=batch_idx, task=task, x=x, y=y)
            loss.backward()
            self.after_backward(task, batch_idx=batch_idx)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)
