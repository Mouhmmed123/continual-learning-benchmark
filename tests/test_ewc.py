"""Unit tests for Elastic Weight Consolidation (EWC) strategy."""

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cl_benchmark.datasets import Task, TaskStream
from cl_benchmark.models import MLP, ContinualModel
from cl_benchmark.strategies.ewc import EWC


def _make_model(num_classes: int = 2, task_id: int = 0) -> ContinualModel:
    """Build a ContinualModel with one pre-registered multi-head task."""
    backbone = MLP(input_dim=784, hidden_dim=32)
    model = ContinualModel(backbone=backbone, multi_head=True)
    classes = list(range(task_id * num_classes, task_id * num_classes + num_classes))
    model.add_task_head(task_id=task_id, classes=classes, multi_head=True)
    return model


def _create_synthetic_task(
    task_id: int, num_samples: int = 32, num_classes: int = 2
) -> Task:
    """Create a synthetic task with dummy image tensors."""
    classes = [task_id * num_classes + c for c in range(num_classes)]
    x = torch.randn(num_samples, 1, 28, 28)
    y = torch.tensor(classes * (num_samples // num_classes))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    return Task(
        task_id=task_id,
        train_loader=loader,
        test_loader=loader,
        classes=classes,
    )


def test_ewc_fisher_diagonal_properties():
    """Fisher information matrix diagonals must be non-negative and shape-matched."""
    model = _make_model(num_classes=2, task_id=0)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    strategy = EWC(model=model, optimizer=optimizer, ewc_lambda=100.0, device="cpu")

    task = _create_synthetic_task(task_id=0, num_samples=32, num_classes=2)
    fisher, optimal_params = strategy.compute_fisher_and_params(task)

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in fisher, f"'{name}' missing from Fisher dict."
            assert name in optimal_params, f"'{name}' missing from optimal params."
            assert fisher[name].shape == param.shape
            assert optimal_params[name].shape == param.shape
            assert (fisher[name] >= 0.0).all(), (
                f"Fisher diagonal for '{name}' contains negative values."
            )


def test_ewc_penalty_strict_growth():
    """EWC penalty must be 0 at theta* and strictly increase with weight drift."""
    model = _make_model(num_classes=2, task_id=0)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    strategy = EWC(model=model, optimizer=optimizer, ewc_lambda=200.0, device="cpu")

    task = _create_synthetic_task(task_id=0, num_samples=32, num_classes=2)

    # Train task and compute Fisher (populates saved_tasks via after_training_task)
    strategy.train_task(task, epochs=1)

    # At optimal params, penalty must be zero
    assert strategy.penalty().item() == pytest.approx(0.0, abs=1e-6)

    # Perturb model weights slightly
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05)

    penalty_small = strategy.penalty().item()
    assert penalty_small > 0.0, (
        "Penalty should be positive after parameter perturbation."
    )

    # Perturb model weights further and verify strictly larger penalty
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.10)

    penalty_large = strategy.penalty().item()
    assert penalty_large > penalty_small, (
        "Penalty must strictly increase with larger parameter drift."
    )


def test_ewc_end_to_end_continual_training():
    """EWC trains sequentially across multiple tasks with ContinualModel."""
    backbone = MLP(input_dim=784, hidden_dim=32)
    model = ContinualModel(backbone=backbone, multi_head=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    strategy = EWC(model=model, optimizer=optimizer, ewc_lambda=100.0, device="cpu")

    tasks = [
        _create_synthetic_task(t_id, num_samples=16, num_classes=2)
        for t_id in range(2)
    ]
    stream = TaskStream(tasks)

    for task in stream:
        model.add_task_head(task.task_id, task.classes, multi_head=True)
        tracked = {
            p
            for group in optimizer.param_groups
            for p in group["params"]
        }
        new_params = [p for p in model.parameters() if p not in tracked]
        if new_params:
            optimizer.add_param_group({"params": new_params})

        losses = strategy.train_task(task, epochs=1)
        assert len(losses) == 1
        assert isinstance(losses[0], float)

    # Saved Fisher dicts must match number of completed tasks
    assert len(strategy.saved_tasks) == 2

    # Evaluates cleanly on all seen tasks
    acc_0 = strategy.evaluate(tasks[0])
    acc_1 = strategy.evaluate(tasks[1])
    assert 0.0 <= acc_0 <= 1.0
    assert 0.0 <= acc_1 <= 1.0
