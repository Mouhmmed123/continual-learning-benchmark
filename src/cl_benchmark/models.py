"""Neural network architectures and continual learning model wrappers."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """2-layer Multi-Layer Perceptron feature extractor.

    Input tensor shape: `[B, input_dim]` or `[B, C, H, W]` (flattened automatically).
    Output tensor shape: `[B, hidden_dim]`.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256) -> None:
        """Initialize the MLP backbone.

        Args:
            input_dim: Number of input features or flattened image pixels.
            hidden_dim: Number of hidden units in linear layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input batch.

        Args:
            x: Input tensor shaped `[B, ...]` where flattened dimension is `input_dim`.

        Returns:
            Feature tensor shaped `[B, hidden_dim]`.
        """
        return self.net(x)


class SimpleConvNet(nn.Module):
    """3-layer Convolutional Network feature extractor for 32x32 images.

    Input tensor shape: `[B, 3, 32, 32]`.
    Output tensor shape: `[B, hidden_dim]`.
    """

    def __init__(self, hidden_dim: int = 512) -> None:
        """Initialize the ConvNet backbone.

        Args:
            hidden_dim: Output feature dimensionality after fully-connected projection.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from RGB image batch.

        Args:
            x: Input image tensor shaped `[B, 3, 32, 32]`.

        Returns:
            Feature tensor shaped `[B, hidden_dim]`.
        """
        return self.fc(self.features(x))


class ContinualModel(nn.Module):
    """Continual learning wrapper supporting multi-head and single-head outputs.

    Provides target remapping (`to_local_targets`) for multi-head classification
    and supports progressive task registration without architectural surgery.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int | None = None,
        sample_input: torch.Tensor | None = None,
        num_classes_per_task: int | None = None,
        num_tasks: int | None = None,
        multi_head: bool = True,
    ) -> None:
        """Initialize the continual model wrapper.

        Args:
            backbone: Feature extractor mapping input `[B, ...]` to `[B, D]`.
            feature_dim: Optional explicit feature dimension.
            sample_input: Optional sample tensor used to infer feature dimension.
            num_classes_per_task: Optional classes per task to pre-init heads.
            num_tasks: Optional total tasks to pre-init heads.
            multi_head: Default head configuration (True=task heads, False=shared).
        """
        super().__init__()
        self.backbone = backbone
        self.multi_head = multi_head

        if feature_dim is not None:
            self.feature_dim = feature_dim
        elif sample_input is not None:
            with torch.no_grad():
                feat = backbone(sample_input)
                self.feature_dim = feat.shape[1] if feat.ndim > 1 else feat.shape[0]
        else:
            self.feature_dim = getattr(backbone, "out_dim", 256)

        self.heads = nn.ModuleDict()
        self.task_classes: dict[int, list[int]] = {}
        self.task_class_maps: dict[int, dict[int, int]] = {}

        # Pre-initialize heads if task structure is provided
        if num_classes_per_task is not None and num_tasks is not None:
            if multi_head:
                for t in range(num_tasks):
                    task_cls = list(
                        range(t * num_classes_per_task, (t + 1) * num_classes_per_task)
                    )
                    self.add_task_head(task_id=t, classes=task_cls, multi_head=True)
            else:
                total_classes = num_classes_per_task * num_tasks
                self.add_task_head(
                    task_id=0, classes=list(range(total_classes)), multi_head=False
                )

    def _get_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        """Return the device and dtype of the backbone parameters."""
        for p in self.backbone.parameters():
            return p.device, p.dtype
        return torch.device("cpu"), torch.float32

    def add_task_head(
        self,
        task_id: int,
        classes: Sequence[int],
        multi_head: bool | None = None,
    ) -> None:
        """Register or expand an output classification head for a given task.

        Args:
            task_id: Task index identifier.
            classes: Sequence of unique global class IDs belonging to this task.
            multi_head: True for task-specific head, False for expanding shared head.
        """
        if multi_head is None:
            multi_head = self.multi_head

        cls_list = list(sorted(set(int(c) for c in classes))) if classes else [0]
        self.task_classes[task_id] = cls_list
        self.task_class_maps[task_id] = {c: idx for idx, c in enumerate(cls_list)}

        dev, dt = self._get_device_and_dtype()
        task_str = str(task_id)

        if multi_head:
            if task_str not in self.heads:
                head = nn.Linear(self.feature_dim, len(cls_list), device=dev, dtype=dt)
                self.heads[task_str] = head
        else:
            required_classes = max(cls_list) + 1 if cls_list else 1
            if "shared" not in self.heads:
                self.heads["shared"] = nn.Linear(
                    self.feature_dim, required_classes, device=dev, dtype=dt
                )
            elif self.heads["shared"].out_features < required_classes:
                old_head = self.heads["shared"]
                new_head = nn.Linear(
                    self.feature_dim,
                    required_classes,
                    device=old_head.weight.device,
                    dtype=old_head.weight.dtype,
                )
                with torch.no_grad():
                    new_head.weight[: old_head.out_features] = old_head.weight
                    new_head.bias[: old_head.out_features] = old_head.bias
                self.heads["shared"] = new_head

    def forward(
        self,
        x: torch.Tensor,
        task_id: int = 0,
        multi_head: bool | None = None,
    ) -> torch.Tensor:
        """Forward pass through backbone and appropriate classification head.

        Args:
            x: Input tensor shaped `[B, ...]`.
            task_id: Task ID specifying which head to evaluate in multi-head setting.
            multi_head: Override for multi-head routing.

        Returns:
            Logit tensor shaped `[B, num_task_classes]` or `[B, num_total_classes]`.
        """
        is_multi = self.multi_head if multi_head is None else multi_head
        features = self.backbone(x)

        if is_multi:
            task_str = str(task_id)
            if task_str not in self.heads:
                raise KeyError(
                    f"Task head '{task_str}' has not been registered. "
                    f"Call add_task_head first."
                )
            return self.heads[task_str](features)

        if "shared" not in self.heads:
            raise KeyError(
                "Shared head has not been registered. Call add_task_head first."
            )
        return self.heads["shared"](features)

    def to_local_targets(
        self,
        targets: torch.Tensor,
        task_id: int,
        multi_head: bool | None = None,
    ) -> torch.Tensor:
        """Map global class labels to local task labels for multi-head loss.

        In multi-head mode, global class IDs (e.g. [4, 5]) map to local indices [0, 1].
        In single-head mode, targets remain global IDs [B].

        Args:
            targets: Global target label tensor shaped `[B]`.
            task_id: Active task index.
            multi_head: Override for multi-head behavior.

        Returns:
            Mapped target tensor shaped `[B]`.

        Raises:
            ValueError: If a target does not belong to the classes registered for task.
        """
        is_multi = self.multi_head if multi_head is None else multi_head

        if not is_multi:
            return targets

        if task_id not in self.task_class_maps:
            raise KeyError(
                f"Task ID {task_id} not registered in class metadata. "
                f"Call add_task_head first."
            )

        mapping = self.task_class_maps[task_id]
        local_targets = torch.empty_like(targets)

        # Fast vectorized replacement per class
        for global_cls, local_idx in mapping.items():
            mask = targets == global_cls
            local_targets[mask] = local_idx

        # Check for any unmapped targets
        unmapped_mask = torch.ones_like(targets, dtype=torch.bool)
        for global_cls in mapping:
            unmapped_mask &= targets != global_cls

        if unmapped_mask.any():
            invalid_classes = targets[unmapped_mask].unique().tolist()
            raise ValueError(
                f"Targets contain class IDs {invalid_classes} not belonging to "
                f"Task {task_id} (allowed: {list(mapping.keys())})."
            )

        return local_targets
