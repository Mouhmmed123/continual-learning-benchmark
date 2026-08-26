import pytest
import torch

from cl_benchmark.datasets import get_benchmark


def test_split_mnist():
    """Verify split-MNIST creates five class-disjoint image streams."""
    stream = get_benchmark("split_mnist", num_tasks=5, batch_size=32)
    assert len(stream) == 5, f"Expected 5 tasks, got {len(stream)}"
    all_seen_classes = []
    for task in stream:
        images, labels = next(iter(task.train_loader))
        assert images.shape == (32, 1, 28, 28)
        assert set(labels.numpy()).issubset(set(task.classes))
        all_seen_classes.extend(task.classes)

    # Verify complete class disjointness across tasks
    assert len(all_seen_classes) == len(set(all_seen_classes)) == 10


def test_split_cifar10():
    """Verify split-CIFAR10 returns channel-first RGB image batches."""
    stream = get_benchmark("split_cifar10", num_tasks=5, batch_size=16)
    assert len(stream) == 5
    images, _ = next(iter(stream[0].train_loader))
    assert images.shape == (16, 3, 32, 32)


def test_permuted_mnist():
    """Verify separate permuted-MNIST tasks produce different pixel layouts."""
    stream = get_benchmark("permuted_mnist", num_tasks=3, batch_size=32)
    assert len(stream) == 3
    img0, _ = next(iter(stream[0].train_loader))
    img1, _ = next(iter(stream[1].train_loader))
    assert not torch.equal(img0, img1)


def test_benchmark_error_handling():
    """Verify benchmark errors on invalid names or indivisible task counts."""
    with pytest.raises(ValueError, match="Unsupported benchmark protocol"):
        get_benchmark("invalid_dataset_name")

    with pytest.raises(ValueError, match="Cannot evenly divide"):
        get_benchmark("split_mnist", num_tasks=3)
