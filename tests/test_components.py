import pytest
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from cl_benchmark.models import MLP, ContinualModel
from cl_benchmark.strategies.base import BaseStrategy
from cl_benchmark.strategies.buffer import ExemplarBuffer
from cl_benchmark.evaluator import Evaluator
from cl_benchmark.datasets import Task


# --- Dummy Concrete Strategy for BaseStrategy Testing ---
class DummyStrategy(BaseStrategy):
    """Concrete subclass of BaseStrategy for lifecycle testing."""
    def train_epoch(self, task: Task) -> float:
        total_loss = 0.0
        for x, y in task.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x, task_id=task.task_id)
            y_local = self.model.to_local_targets(y, task.task_id)
            loss = nn.functional.cross_entropy(out, y_local)
            loss = self.before_backward(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(task.train_loader), 1)

    def evaluate(self, task: Task) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in task.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x, task_id=task.task_id)
                y_local = self.model.to_local_targets(y, task.task_id)
                preds = out.argmax(dim=-1)
                correct += (preds == y_local).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0


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
    
    assert len(buffer.x) <= 10
    assert len(buffer.y) <= 10
    
    x_buf, y_buf, t_buf = buffer.sample(batch_size=4)
    assert x_buf.shape == (4, 3, 32, 32)
    assert y_buf.shape == (4,)
    assert t_buf.shape == (4,)


def test_evaluator_metrics_calculation():
    evaluator = Evaluator(num_tasks=3)
    
    # Handcrafted Accuracy Matrix R (3 x 3)
    # R[0] after Task 0: [0.80, 0.00, 0.00]
    # R[1] after Task 1: [0.70, 0.85, 0.00]
    # R[2] after Task 2: [0.60, 0.75, 0.90]
    evaluator.update(training_task_id=0, test_task_id=0, accuracy=0.80)
    evaluator.update(training_task_id=1, test_task_id=0, accuracy=0.70)
    evaluator.update(training_task_id=1, test_task_id=1, accuracy=0.85)
    evaluator.update(training_task_id=2, test_task_id=0, accuracy=0.60)
    evaluator.update(training_task_id=2, test_task_id=1, accuracy=0.75)
    evaluator.update(training_task_id=2, test_task_id=2, accuracy=0.90)

    # Expected Avg Acc = (0.60 + 0.75 + 0.90) / 3 = 0.75
    avg_acc = evaluator.average_accuracy()
    assert pytest.approx(avg_acc, abs=1e-4) == 0.75

    # Expected BWT = ((0.60 - 0.80) + (0.75 - 0.85)) / 2 = (-0.20 + -0.10) / 2 = -0.15
    bwt = evaluator.backward_transfer()
    assert pytest.approx(bwt, abs=1e-4) == -0.15


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
    
    task = Task(task_id=0, train_loader=loader_train, test_loader=loader_train, classes=[0, 1])

    # Execute Lifecycle
    strategy.before_training_task(task)
    loss = strategy.train_epoch(task)
    strategy.after_training_task(task)
    acc = strategy.evaluate(task)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0