"""Dataset stream builders for continual learning benchmark protocols."""

from __future__ import annotations

from typing import Iterator, Sequence, Type

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


class Task:
    """Hold loaders and class metadata for one task in a benchmark stream.

    Loaders yield image batches shaped `[B, C, H, W]` and label batches `[B]`.
    """

    def __init__(
        self,
        task_id: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        classes: Sequence[int],
    ) -> None:
        """Initialize a task with its loaders and class IDs.

        Args:
            task_id: Unique integer index for the task (0-indexed).
            train_loader: DataLoader yielding training batches `(x, y)`.
            test_loader: DataLoader yielding evaluation batches `(x, y)`.
            classes: Sequence of global integer class labels contained in this task.
        """
        self.task_id = task_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = [int(c) for c in classes]


class TaskStream:
    """Provide indexed and sequential access to a benchmark's tasks."""

    def __init__(self, tasks: Sequence[Task]) -> None:
        """Initialize the stream from tasks in execution order.

        Args:
            tasks: Sequence of Task instances.
        """
        self.tasks = list(tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Task:
        return self.tasks[idx]

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)


class PermuteTransform:
    """Apply a fixed pixel permutation while preserving image tensor shape.

    Input and output tensors have shape `[C, H, W]` (or `[B, C, H, W]`); the permutation
    is applied to the flattened spatial elements.
    """

    def __init__(self, permutation: torch.Tensor) -> None:
        """Initialize transform with a 1D permutation index tensor.

        Args:
            permutation: 1D torch.Tensor containing a pixel permutation.
        """
        self.permutation = permutation

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return input with spatial elements reordered according to permutation.

        Args:
            tensor: Image tensor shaped `[C, H, W]` or `[B, C, H, W]`.

        Returns:
            Permuted image tensor with the original shape preserved.
        """
        shape = tensor.shape
        flattened = tensor.reshape(-1)
        permuted = flattened[self.permutation]
        return permuted.view(shape)


def _get_dataset_targets(dataset: Dataset) -> np.ndarray:
    """Extract targets as a NumPy array cleanly without NumPy 2.0 warnings."""
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.detach().cpu().numpy()
        return np.asarray(targets)
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
        if isinstance(labels, torch.Tensor):
            return labels.detach().cpu().numpy()
        return np.asarray(labels)
    raise AttributeError("Dataset does not expose targets or labels attribute.")


def _build_split_stream(
    dataset_cls: Type[Dataset],
    transform: transforms.Compose | None,
    num_tasks: int,
    batch_size: int,
    root: str = "./data",
    num_workers: int = 0,
    pin_memory: bool = False,
) -> TaskStream:
    """Build class-disjoint task loaders from a torchvision dataset class.

    Args:
        dataset_cls: Torchvision dataset class (e.g., MNIST, CIFAR10, CIFAR100).
        transform: Transformations applied to dataset samples.
        num_tasks: Number of class-disjoint tasks to construct.
        batch_size: Mini-batch size for training and evaluation.
        root: Directory where raw/processed dataset is stored.
        num_workers: Number of DataLoader subprocesses.
        pin_memory: Whether DataLoader should pin memory for fast GPU transfer.

    Returns:
        TaskStream containing class-disjoint tasks.
    """
    train_ds = dataset_cls(root=root, train=True, download=True, transform=transform)
    test_ds = dataset_cls(root=root, train=False, download=True, transform=transform)

    train_targets = _get_dataset_targets(train_ds)
    test_targets = _get_dataset_targets(test_ds)
    unique_classes = np.sort(np.unique(train_targets))
    num_classes = len(unique_classes)

    if num_classes % num_tasks != 0:
        raise ValueError(
            f"Cannot evenly divide {num_classes} classes into {num_tasks} tasks. "
            f"Please choose num_tasks that divides {num_classes}."
        )

    classes_per_task = num_classes // num_tasks
    tasks: list[Task] = []

    for i in range(num_tasks):
        task_classes = unique_classes[i * classes_per_task : (i + 1) * classes_per_task]

        train_idx = np.where(np.isin(train_targets, task_classes))[0]
        test_idx = np.where(np.isin(test_targets, task_classes))[0]

        train_loader = DataLoader(
            Subset(train_ds, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            Subset(test_ds, test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        tasks.append(
            Task(
                task_id=i,
                train_loader=train_loader,
                test_loader=test_loader,
                classes=task_classes.tolist(),
            )
        )

    return TaskStream(tasks)


def _build_permuted_mnist(
    num_tasks: int,
    batch_size: int,
    seed: int = 42,
    root: str = "./data",
    num_workers: int = 0,
    pin_memory: bool = False,
) -> TaskStream:
    """Build Permuted-MNIST tasks with distinct pixel permutations."""
    rng = np.random.RandomState(seed)
    num_pixels = 28 * 28
    tasks: list[Task] = []

    for i in range(num_tasks):
        perm = torch.from_numpy(rng.permutation(num_pixels))
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                PermuteTransform(perm),
            ]
        )

        train_ds = torchvision.datasets.MNIST(
            root=root, train=True, download=True, transform=transform
        )
        test_ds = torchvision.datasets.MNIST(
            root=root, train=False, download=True, transform=transform
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        tasks.append(
            Task(
                task_id=i,
                train_loader=train_loader,
                test_loader=test_loader,
                classes=list(range(10)),
            )
        )

    return TaskStream(tasks)


def get_benchmark(
    name: str,
    num_tasks: int = 5,
    batch_size: int = 64,
    root: str = "./data",
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> TaskStream:
    """Return a named continual learning benchmark stream `[B, C, H, W]`.

    Supported benchmark names:
    - 'split_mnist': 10-class MNIST split into class-disjoint tasks.
    - 'split_cifar10': 10-class CIFAR-10 split into class-disjoint tasks.
    - 'split_cifar100': 100-class CIFAR-100 split into class-disjoint tasks.
    - 'permuted_mnist': 10-class MNIST with random pixel permutations.

    Args:
        name: Name of the benchmark protocol.
        num_tasks: Total number of tasks in the continual stream.
        batch_size: Batch size for train and test DataLoaders.
        root: Local directory path to store dataset files.
        num_workers: Number of background worker processes for DataLoaders.
        pin_memory: Whether DataLoaders should pin GPU memory.
        seed: Random seed for permutations (used in Permuted-MNIST).

    Returns:
        TaskStream object containing sequential Task instances.

    Raises:
        ValueError: If benchmark name is unsupported or num_tasks is invalid.
    """
    name = name.lower().strip()

    if name == "split_mnist":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return _build_split_stream(
            torchvision.datasets.MNIST,
            transform=tf,
            num_tasks=num_tasks,
            batch_size=batch_size,
            root=root,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    elif name == "split_cifar10":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        return _build_split_stream(
            torchvision.datasets.CIFAR10,
            transform=tf,
            num_tasks=num_tasks,
            batch_size=batch_size,
            root=root,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    elif name == "split_cifar100":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        return _build_split_stream(
            torchvision.datasets.CIFAR100,
            transform=tf,
            num_tasks=num_tasks,
            batch_size=batch_size,
            root=root,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    elif name == "permuted_mnist":
        return _build_permuted_mnist(
            num_tasks=num_tasks,
            batch_size=batch_size,
            seed=seed,
            root=root,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    else:
        valid = ["split_mnist", "split_cifar10", "split_cifar100", "permuted_mnist"]
        raise ValueError(
            f"Unsupported benchmark protocol: '{name}'. Supported options: {valid}."
        )
