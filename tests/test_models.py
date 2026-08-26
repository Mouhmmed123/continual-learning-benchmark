import pytest
import torch

from cl_benchmark.models import MLP, ContinualModel, SimpleConvNet


def test_mlp_single_and_multi_head():
    """Verify MLP backbone works with single-head and multi-head ContinualModel."""
    backbone = MLP(input_dim=784, hidden_dim=256)
    dummy_input = torch.randn(16, 1, 28, 28)

    # Test Single-Head (5 tasks * 2 classes = 10 classes)
    single_model = ContinualModel(
        backbone, num_classes_per_task=2, num_tasks=5, multi_head=False
    )
    out_single = single_model(dummy_input)
    assert out_single.shape == (16, 10), f"Expected (16, 10), got {out_single.shape}"

    # Test Multi-Head (2 output logits for active task head)
    multi_model = ContinualModel(
        backbone, num_classes_per_task=2, num_tasks=5, multi_head=True
    )
    out_multi_t0 = multi_model(dummy_input, task_id=0)
    assert out_multi_t0.shape == (16, 2), f"Expected (16, 2), got {out_multi_t0.shape}"


def test_convnet_forward():
    """Verify SimpleConvNet processes 3-channel RGB image batches."""
    backbone = SimpleConvNet(hidden_dim=512)
    dummy_input = torch.randn(8, 3, 32, 32)
    model = ContinualModel(
        backbone, num_classes_per_task=2, num_tasks=5, multi_head=True
    )
    out = model(dummy_input, task_id=2)
    assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"


def test_dynamic_head_addition_and_weight_preservation():
    """Verify dynamic head registration preserves weights when expanding shared head."""
    backbone = MLP(input_dim=784, hidden_dim=128)
    model = ContinualModel(backbone, multi_head=False)

    # Add task 0 with classes [0, 1]
    model.add_task_head(task_id=0, classes=[0, 1], multi_head=False)
    assert model.heads["shared"].out_features == 2

    # Fill weights with recognizable values
    with torch.no_grad():
        model.heads["shared"].weight.fill_(1.5)
        model.heads["shared"].bias.fill_(0.5)

    # Expand shared head to include task 1 with classes [2, 3] (total 4 classes)
    model.add_task_head(task_id=1, classes=[2, 3], multi_head=False)
    assert model.heads["shared"].out_features == 4

    # Verify existing weights were preserved
    assert torch.allclose(model.heads["shared"].weight[:2], torch.tensor(1.5))
    assert torch.allclose(model.heads["shared"].bias[:2], torch.tensor(0.5))


def test_non_contiguous_class_remapping():
    """Verify target remapping works correctly for non-contiguous class partitions."""
    backbone = MLP(input_dim=784, hidden_dim=64)
    model = ContinualModel(backbone, multi_head=True)

    # Task with arbitrary non-contiguous classes: [1, 5, 8]
    model.add_task_head(task_id=0, classes=[1, 5, 8], multi_head=True)

    targets = torch.tensor([1, 8, 5, 8, 1])
    local_targets = model.to_local_targets(targets, task_id=0)

    # Expected mapping: 1 -> 0, 5 -> 1, 8 -> 2
    assert torch.equal(local_targets, torch.tensor([0, 2, 1, 2, 0]))

    # Test error raised on out-of-task target
    invalid_targets = torch.tensor([1, 9, 5])
    with pytest.raises(ValueError, match="Targets contain class IDs"):
        model.to_local_targets(invalid_targets, task_id=0)
