"""Elastic Weight Consolidation (EWC) continual learning strategy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from cl_benchmark.datasets import Task
from cl_benchmark.models import ContinualModel
from cl_benchmark.strategies.base import BaseStrategy


class EWC(BaseStrategy):
    """Elastic Weight Consolidation strategy regularizing parameter drift.

    Penalizes updates to critical parameters based on the diagonal Fisher
    Information Matrix estimated from previous tasks (Kirkpatrick et al., 2017).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        ewc_lambda: float = 400.0,
        criterion: nn.Module | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize EWC strategy.

        Args:
            model: Continual learning model or PyTorch module.
            optimizer: PyTorch optimizer instance.
            ewc_lambda: Regularization penalty weight (default: 400.0).
            criterion: Loss function (defaults to nn.CrossEntropyLoss()).
            device: Compute device.
        """
        super().__init__(
            model=model, optimizer=optimizer, criterion=criterion, device=device
        )
        self.ewc_lambda = float(ewc_lambda)
        # Store list of tuples: (fisher_dict, optimal_params_dict) per completed task
        self.saved_tasks: list[
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        ] = []

    def compute_fisher_and_params(
        self, task: Task
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute empirical Fisher diagonal and snapshot current optimal weights.

        Args:
            task: Task to compute empirical Fisher matrix across.

        Returns:
            Tuple of (fisher_dict, params_dict) mapping parameter names to tensors.
        """
        self.model.eval()
        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        optimal_params: dict[str, torch.Tensor] = {
            n: p.detach().clone().to(self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        total_samples = 0

        for x, y in task.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            batch_size = x.size(0)
            total_samples += batch_size

            self.model.zero_grad()

            if isinstance(self.model, ContinualModel):
                logits = self.model(x, task_id=task.task_id)
                targets = self.model.to_local_targets(y, task_id=task.task_id)
            else:
                logits = self.model(x)
                targets = y

            # Empirical Fisher uses squared gradients of the negative log-likelihood
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(log_probs, targets)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += (p.grad.detach() ** 2) * batch_size

        if total_samples > 0:
            for n in fisher:
                fisher[n] /= total_samples

        self.model.zero_grad()
        return fisher, optimal_params

    def after_training_task(self, task: Task) -> None:
        """Estimate Fisher information matrix after task training concludes.

        Args:
            task: Completed training task.
        """
        fisher, optimal_params = self.compute_fisher_and_params(task)
        self.saved_tasks.append((fisher, optimal_params))

    def penalty(self) -> torch.Tensor:
        """Compute the cumulative EWC quadratic penalty loss.

        Returns:
            Scalar tensor holding the quadratic penalty value.
        """
        if not self.saved_tasks:
            return torch.tensor(0.0, device=self.device)

        loss_penalty = torch.tensor(0.0, device=self.device)
        named_params = dict(self.model.named_parameters())

        for task_fisher, task_params in self.saved_tasks:
            for name, param in named_params.items():
                if name in task_fisher and name in task_params and param.requires_grad:
                    f = task_fisher[name].to(param.device)
                    opt = task_params[name].to(param.device)
                    loss_penalty = loss_penalty + (f * (param - opt) ** 2).sum()

        return loss_penalty

    def before_backward(self, loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """Augment task loss with the EWC quadratic penalty before backpropagation.

        Args:
            loss: Primary cross-entropy loss tensor.
            **kwargs: Optional batch context.

        Returns:
            Total loss with EWC quadratic penalty added.
        """
        if self.saved_tasks and self.ewc_lambda > 0:
            ewc_loss = (self.ewc_lambda / 2.0) * self.penalty()
            return loss + ewc_loss
        return loss

    def train_epoch(self, task: Task) -> float:
        """Train one epoch on the current task incorporating EWC penalty.

        Args:
            task: Active training task.

        Returns:
            Average total step loss across batches.
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

            base_loss = self.criterion(logits, targets)
            step_loss = self.before_backward(
                base_loss, batch_idx=batch_idx, task=task, x=x, y=y
            )
            step_loss.backward()
            self.after_backward(task, batch_idx=batch_idx)
            self.optimizer.step()

            total_loss += step_loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)
