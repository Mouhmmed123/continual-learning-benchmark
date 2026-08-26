"""Dataset stream builders for continual-learning benchmark protocols."""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class Task:
    """Hold loaders and class metadata for one task in a benchmark stream.

    Loaders yield image batches shaped `[B, C, H, W]` and label batches `[B]`.
    """

    def __init__(
        self,
        task_id: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        classes: list,
    ):
        """Initialize a task with its loaders and class IDs."""
        self.task_id = task_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = classes


class TaskStream:
    """Provide indexed and sequential access to a benchmark's tasks."""

    def __init__(self, tasks: list[Task]):
        """Initialize the stream from tasks in execution order."""
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Task:
        return self.tasks[idx]

    def __iter__(self):
        return iter(self.tasks)


class PermuteTransform:
    """Apply a fixed pixel permutation while preserving image shape.

    Input and output tensors have shape `[C, H, W]` (or any shape with the same
    number of elements); the permutation is applied to the flattened tensor.
    """

    def __init__(self, permutation: torch.Tensor):
        """Initialize the transform with a pixel-index permutation."""
        self.permutation = permutation

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the input with pixels reordered according to the permutation."""
        shape = tensor.shape
        # Flatten only for indexing; restoring `shape` keeps downstream loaders
        # compatible with image models expecting channel-first tensors.
        flattened = tensor.view(-1)
        permuted = flattened[self.permutation]
        return permuted.view(shape)


def _build_split_stream(
    dataset_cls, transform, num_tasks: int, batch_size: int, root: str = "./data"
):
    """Build class-disjoint task loaders from a torchvision dataset class."""
    train_ds = dataset_cls(
        root=root, train=True, download=True, transform=transform
    )
    test_ds = dataset_cls(
        root=root, train=False, download=True, transform=transform
    )

    train_targets = np.array(train_ds.targets)
    test_targets = np.array(test_ds.targets)
    unique_classes = np.unique(train_targets)

    classes_per_task = len(unique_classes) // num_tasks
    tasks = []

    for i in range(num_tasks):
        task_classes = unique_classes[
            i * classes_per_task : (i + 1) * classes_per_task
        ]

        train_idx = np.where(np.isin(train_targets, task_classes))[0]
        test_idx = np.where(np.isin(test_targets, task_classes))[0]

        train_loader = DataLoader(
            Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            Subset(test_ds, test_idx), batch_size=batch_size, shuffle=False
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
    num_tasks: int, batch_size: int, seed: int = 42, root: str = "./data"
):
    """Build MNIST tasks sharing labels but using distinct pixel permutations."""
    rng = np.random.RandomState(seed)
    num_pixels = 28 * 28
    tasks = []

    for i in range(num_tasks):
        perm = torch.from_numpy(rng.permutation(num_pixels))
        transform = transforms.Compose(
            [transforms.ToTensor(), PermuteTransform(perm)]
        )

        train_ds = torchvision.datasets.MNIST(
            root=root, train=True, download=True, transform=transform
        )
        test_ds = torchvision.datasets.MNIST(
            root=root, train=False, download=True, transform=transform
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False
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
    name: str, num_tasks: int = 5, batch_size: int = 64, root: str = "./data"
) -> TaskStream:
    """Return a named benchmark stream whose image batches are `[B, C, H, W]`."""
    name = name.lower()

    if name == "split_mnist":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return _build_split_stream(
            torchvision.datasets.MNIST, tf, num_tasks, batch_size, root
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
            torchvision.datasets.CIFAR10, tf, num_tasks, batch_size, root
        )

    elif name == "permuted_mnist":
        return _build_permuted_mnist(num_tasks, batch_size, root=root)

    else:
        raise ValueError(f"Unsupported benchmark protocol: {name}")