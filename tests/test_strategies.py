import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cl_benchmark.datasets import Task, TaskStream
from cl_benchmark.models import MLP, ContinualModel
from cl_benchmark.strategies.buffer import ExemplarBuffer
from cl_benchmark.strategies.naive import Naive
from cl_benchmark.strategies.rehearsal import Rehearsal


def _create_synthetic_stream() -> TaskStream:
    """Create a 2-task synthetic stream with 2 classes each."""
    tasks = []
    for t_id in range(2):
        classes = [t_id * 2, t_id * 2 + 1]
        x_train = torch.randn(32, 1, 28, 28)
        y_train = torch.tensor([classes[0], classes[1]] * 16)
        ds_train = TensorDataset(x_train, y_train)

        x_test = torch.randn(16, 1, 28, 28)
        y_test = torch.tensor([classes[0], classes[1]] * 8)
        ds_test = TensorDataset(x_test, y_test)

        tasks.append(
            Task(
                task_id=t_id,
                train_loader=DataLoader(ds_train, batch_size=8, shuffle=True),
                test_loader=DataLoader(ds_test, batch_size=8, shuffle=False),
                classes=classes,
            )
        )
    return TaskStream(tasks)


def test_naive_strategy_multi_head():
    """Verify Naive strategy trains across tasks with multi-head ContinualModel."""
    stream = _create_synthetic_stream()
    backbone = MLP(input_dim=784, hidden_dim=64)
    model = ContinualModel(
        backbone, num_classes_per_task=2, num_tasks=2, multi_head=True
    )
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    strategy = Naive(model=model, optimizer=optimizer, device="cpu")

    for task in stream:
        losses = strategy.train_task(task, epochs=2)
        assert len(losses) == 2
        acc = strategy.evaluate(task)
        assert 0.0 <= acc <= 1.0

    eval_results = strategy.evaluate_all(stream)
    assert len(eval_results) == 2
    assert 0 in eval_results and 1 in eval_results


def test_rehearsal_strategy_multi_head():
    """Verify Rehearsal strategy accumulates replay exemplars and replays data."""

    stream = _create_synthetic_stream()
    backbone = MLP(input_dim=784, hidden_dim=64)
    model = ContinualModel(
        backbone, num_classes_per_task=2, num_tasks=2, multi_head=True
    )
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    buffer = ExemplarBuffer(max_size=50)

    strategy = Rehearsal(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        replay_batch_size=8,
        device="cpu",
    )

    # Train Task 0
    strategy.train_task(stream[0], epochs=1)
    assert len(strategy.buffer) > 0

    # Train Task 1 (should interleave Task 0 samples from buffer)
    strategy.train_task(stream[1], epochs=1)
    assert len(strategy.buffer) > 0

    acc_0 = strategy.evaluate(stream[0])
    acc_1 = strategy.evaluate(stream[1])
    assert 0.0 <= acc_0 <= 1.0
    assert 0.0 <= acc_1 <= 1.0


def test_rehearsal_strategy_single_head():
    """Verify Rehearsal strategy functions correctly under single-head mode."""
    stream = _create_synthetic_stream()
    backbone = MLP(input_dim=784, hidden_dim=64)
    model = ContinualModel(
        backbone, num_classes_per_task=2, num_tasks=2, multi_head=False
    )
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    buffer = ExemplarBuffer(max_size=50)

    strategy = Rehearsal(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        replay_batch_size=8,
        device="cpu",
    )

    for task in stream:
        strategy.train_task(task, epochs=1)

    eval_results = strategy.evaluate_all(stream)
    assert len(eval_results) == 2
