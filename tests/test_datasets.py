import torch

from cl_benchmark.datasets import get_benchmark


def test_split_mnist():
    """Verify split-MNIST creates five class-disjoint image streams."""
    stream = get_benchmark("split_mnist", num_tasks=5, batch_size=32)
    assert len(stream) == 5, f"Expected 5 tasks, got {len(stream)}"
    for task in stream:
        images, labels = next(iter(task.train_loader))
        assert images.shape == (32, 1, 28, 28)
        assert set(labels.numpy()).issubset(set(task.classes))
    print("✓ Split-MNIST: Pass")

def test_split_cifar10():
    """Verify split-CIFAR10 returns channel-first RGB image batches."""
    stream = get_benchmark("split_cifar10", num_tasks=5, batch_size=16)
    assert len(stream) == 5
    images, _ = next(iter(stream[0].train_loader))
    assert images.shape == (16, 3, 32, 32)
    print("✓ Split-CIFAR10: Pass")

def test_permuted_mnist():
    """Verify separate permuted-MNIST tasks produce different pixel layouts."""
    stream = get_benchmark("permuted_mnist", num_tasks=3, batch_size=32)
    assert len(stream) == 3
    img0, _ = next(iter(stream[0].train_loader))
    img1, _ = next(iter(stream[1].train_loader))
    assert not torch.equal(img0, img1)
    print("✓ Permuted-MNIST: Pass")

if __name__ == "__main__":
    test_split_mnist()
    test_split_cifar10()
    test_permuted_mnist()
    print("\nAll dataset tests passed successfully!")