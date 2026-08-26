"""Experience replay (Rehearsal) continual learning strategy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer

from cl_benchmark.datasets import Task
from cl_benchmark.models import ContinualModel
from cl_benchmark.strategies.base import BaseStrategy
from cl_benchmark.strategies.buffer import ExemplarBuffer


class Rehearsal(BaseStrategy):
    """Experience Replay (Rehearsal) strategy using a fixed-capacity exemplar buffer.

    Interleaves current task data with random exemplars sampled from previous tasks
    to mitigate catastrophic forgetting.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        buffer: ExemplarBuffer | None = None,
        buffer_size: int = 500,
        replay_batch_size: int = 32,
        criterion: nn.Module | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize Rehearsal strategy.

        Args:
            model: Continual learning model or PyTorch module.
            optimizer: PyTorch optimizer instance.
            buffer: Optional pre-configured ExemplarBuffer.
            buffer_size: Capacity of the exemplar buffer if buffer is None.
            replay_batch_size: Number of replay exemplars sampled per step.
            criterion: Loss function.
            device: Compute device.
        """
        super().__init__(
            model=model, optimizer=optimizer, criterion=criterion, device=device
        )
        self.buffer = (
            buffer if buffer is not None else ExemplarBuffer(max_size=buffer_size)
        )
        self.replay_batch_size = replay_batch_size

    def _compute_loss(
        self, x: torch.Tensor, y: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        """Compute cross-entropy loss under either single or multi-head mode."""
        if isinstance(self.model, ContinualModel):
            logits = self.model(x, task_id=task_id)
            targets = self.model.to_local_targets(y, task_id=task_id)
        else:
            logits = self.model(x)
            targets = y
        return self.criterion(logits, targets)

    def train_epoch(self, task: Task) -> float:
        """Train one epoch using interleaved stream data and replay memory exemplars.

        Args:
            task: Active training task.

        Returns:
            Average total loss across training batches.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(task.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # 1. Current task loss
            loss_curr = self._compute_loss(x, y, task_id=task.task_id)

            # 2. Replay loss (if buffer contains samples)
            loss_replay = torch.tensor(0.0, device=self.device)
            if len(self.buffer) > 0:
                bx, by, bt = self.buffer.sample(
                    self.replay_batch_size, device=self.device, non_blocking=True
                )
                if isinstance(self.model, ContinualModel) and self.model.multi_head:
                    # Multi-head: compute replay loss per task subset
                    task_losses = []
                    for t_val in bt.unique():
                        t_id = int(t_val.item())
                        mask = bt == t_val
                        if mask.any():
                            task_losses.append(
                                self._compute_loss(bx[mask], by[mask], task_id=t_id)
                            )
                    if task_losses:
                        loss_replay = torch.stack(task_losses).mean()
                else:
                    loss_replay = self._compute_loss(bx, by, task_id=task.task_id)

            total_step_loss = loss_curr + loss_replay
            total_step_loss = self.before_backward(
                total_step_loss,
                batch_idx=batch_idx,
                task=task,
                x=x,
                y=y,
                loss_curr=loss_curr,
                loss_replay=loss_replay,
            )
            total_step_loss.backward()
            self.after_backward(task, batch_idx=batch_idx)
            self.optimizer.step()

            # 3. Ingest current stream samples into reservoir buffer
            self.buffer.add_samples(x, y, task_id=task.task_id)

            total_loss += total_step_loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)


# Convenient alias
ExperienceReplay = Rehearsal
