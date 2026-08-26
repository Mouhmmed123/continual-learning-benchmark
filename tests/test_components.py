import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cl_benchmark.datasets import Task
from cl_benchmark.evaluator import Evaluator
from cl_benchmark.models import MLP, ContinualModel
from cl_benchmark.strategies.base import BaseStrategy
from cl_benchmark.strategies.buffer import ExemplarBuffer


# --- Dummy Concrete Strategy for BaseStrategy Testing ---
class DummyStrategy(BaseStrategy):
    """Concrete subclass of BaseStrategy for lifecycle testing."""

    def train_epoch(self, task: Task) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch_idx, (x, y) in enumerate(task.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            out = self.model(x, task_id=task.task_id)
            y_local = self.model.to_local_targets(y, task.task_id)
            loss = nn.functional.cross_entropy(out, y_local)
            loss = self.before_backward(loss, batch_idx=batch_idx, task=task)
            loss.backward()
            self.after_backward(task, batch_idx=batch_idx)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)


# --- Test Suite ---


def test_continual_model_dynamic_heads():
    sample_input = torch.randn(4, 1, 28, 28)
    backbone = MLP(input_dim=784, hidden_dim=128)
    model = ContinualModel(backbone, sample_input=sample_input)

    assert model.feature_dim == 128

    # Multi-head setup
    model.add_task_head(task_id=0, classes=[0, 1], multi_head=True)
    model.add_task_head(task_id=1, classes=[2, 3], multi_head=True)

    out_t0 = model(sample_input, task_id=0, multi_head=True)
    out_t1 = model(sample_input, task_id=1, multi_head=True)

    assert out_t0.shape == (4, 2)
    assert out_t1.shape == (4, 2)

    # Label remapping offset
    targets_t1 = torch.tensor([2, 3, 2, 3])
    local_targets = model.to_local_targets(targets_t1, task_id=1)
    assert torch.equal(local_targets, torch.tensor([0, 1, 0, 1]))


def test_exemplar_buffer_capacity_and_sampling():
    buffer = ExemplarBuffer(max_size=10)
    x = torch.randn(15, 3, 32, 32)
    y = torch.randint(0, 10, (15,))

    buffer.add_samples(x, y, task_id=0)

    assert len(buffer) == 10
    assert buffer.x.shape == (10, 3, 32, 32)
    assert buffer.y.shape == (10,)
    assert buffer.t.shape == (10,)

    x_buf, y_buf, t_buf = buffer.sample(batch_size=4)
    assert x_buf.shape == (4, 3, 32, 32)
    assert y_buf.shape == (4,)
    assert t_buf.shape == (4,)

    # Add second batch to test ongoing reservoir sampling
    x2 = torch.randn(10, 3, 32, 32)
    y2 = torch.randint(10, 20, (10,))
    buffer.add_samples(x2, y2, task_id=1)
    assert len(buffer) == 10
    assert buffer.num_seen == 25

    # Test clear
    buffer.clear()
    assert len(buffer) == 0
    with pytest.raises(ValueError, match="Buffer is empty"):
        buffer.sample(batch_size=2)


def test_evaluator_metrics_calculation():
    evaluator = Evaluator(num_tasks=3)

    # Handcrafted Accuracy Matrix R (3 x 3)
    # R[0] after Task 0: [0.80, 0.10, 0.10]
    # R[1] after Task 1: [0.70, 0.85, 0.15]
    # R[2] after Task 2: [0.60, 0.75, 0.90]
    evaluator.update(training_task_id=0, test_task_id=0, accuracy=0.80)
    evaluator.update(training_task_id=0, test_task_id=1, accuracy=0.10)
    evaluator.update(training_task_id=0, test_task_id=2, accuracy=0.10)

    evaluator.update(training_task_id=1, test_task_id=0, accuracy=0.70)
    evaluator.update(training_task_id=1, test_task_id=1, accuracy=0.85)
    evaluator.update(training_task_id=1, test_task_id=2, accuracy=0.15)

    evaluator.update(training_task_id=2, test_task_id=0, accuracy=0.60)
    evaluator.update(training_task_id=2, test_task_id=1, accuracy=0.75)
    evaluator.update(training_task_id=2, test_task_id=2, accuracy=0.90)

    # Expected Avg Acc = (0.60 + 0.75 + 0.90) / 3 = 0.75
    avg_acc = evaluator.average_accuracy()
    assert pytest.approx(avg_acc, abs=1e-4) == 0.75

    # Expected BWT = ((0.60 - 0.80) + (0.75 - 0.85)) / 2 = (-0.20 + -0.10) / 2 = -0.15
    bwt = evaluator.backward_transfer()
    assert pytest.approx(bwt, abs=1e-4) == -0.15

    # Expected FWT with zero baseline: (0.10 + 0.15) / 2 = 0.125
    fwt = evaluator.forward_transfer()
    assert pytest.approx(fwt, abs=1e-4) == 0.125

    # Expected Forgetting Measure:
    # Task 0 drop = 0.80 - 0.60 = 0.20; Task 1 drop = 0.85 - 0.75 = 0.10 -> mean = 0.15
    fm = evaluator.forgetting_measure()
    assert pytest.approx(fm, abs=1e-4) == 0.15

    # Summary metrics dictionary
    metrics = evaluator.get_metrics()
    assert "average_accuracy" in metrics
    assert "backward_transfer" in metrics
    assert "forward_transfer" in metrics
    assert "forgetting_measure" in metrics


def test_evaluator_single_task_edge_case():
    evaluator = Evaluator(num_tasks=1)
    evaluator.update(training_task_id=0, test_task_id=0, accuracy=0.95)

    assert evaluator.average_accuracy() == 0.95
    assert evaluator.backward_transfer() == 0.0
    assert evaluator.forward_transfer() == 0.0
    assert evaluator.forgetting_measure() == 0.0


def test_base_strategy_lifecycle_execution():
    sample_input = torch.randn(8, 1, 28, 28)
    backbone = MLP(input_dim=784, hidden_dim=64)
    model = ContinualModel(backbone, sample_input=sample_input)
    model.add_task_head(task_id=0, classes=[0, 1], multi_head=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    strategy = DummyStrategy(model=model, optimizer=optimizer, device="cpu")

    # Mock Task Data
    x_train = torch.randn(16, 1, 28, 28)
    y_train = torch.tensor([0, 1] * 8)
    ds_train = TensorDataset(x_train, y_train)
    loader_train = DataLoader(ds_train, batch_size=4)

    task = Task(
        task_id=0,
        train_loader=loader_train,
        test_loader=loader_train,
        classes=[0, 1],
    )

    # Execute Lifecycle
    losses = strategy.train_task(task, epochs=2)
    acc = strategy.evaluate(task)

    assert len(losses) == 2
    assert isinstance(losses[0], float)
    assert 0.0 <= acc <= 1.0
