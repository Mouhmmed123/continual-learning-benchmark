import torch
import torch.nn as nn


class MLP(nn.Module):
    """2-layer Multi-Layer Perceptron feature extractor."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256):
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
        return self.net(x)


class SimpleConvNet(nn.Module):
    """3-layer Convolutional Network for 32x32 color images."""

    def __init__(self, hidden_dim: int = 512):
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
        return self.fc(self.features(x))


class ContinualModel(nn.Module):
    """Continual learning wrapper supporting dynamic head registration and target remapping."""

    def __init__(self, backbone: nn.Module, sample_input: torch.Tensor = None):
        super().__init__()
        self.backbone = backbone

        if sample_input is not None:
            with torch.no_grad():
                self.feature_dim = backbone(sample_input).shape[1]
        else:
            self.feature_dim = getattr(backbone, "out_dim", 256)

        self.heads = nn.ModuleDict()
        self.task_class_offsets = {}

    def add_task_head(
        self, task_id: int, classes: list[int], multi_head: bool = True
    ):
        task_str = str(task_id)
        if task_str in self.heads:
            return

        self.task_class_offsets[task_id] = min(classes) if classes else 0

        if multi_head:
            self.heads[task_str] = nn.Linear(self.feature_dim, len(classes))
        else:
            max_class = max(classes) + 1
            if "shared" not in self.heads:
                self.heads["shared"] = nn.Linear(self.feature_dim, max_class)
            elif self.heads["shared"].out_features < max_class:
                old_head = self.heads["shared"]
                new_head = nn.Linear(self.feature_dim, max_class)
                with torch.no_grad():
                    new_head.weight[: old_head.out_features] = old_head.weight
                    new_head.bias[: old_head.out_features] = old_head.bias
                self.heads["shared"] = new_head

    def forward(
        self, x: torch.Tensor, task_id: int = 0, multi_head: bool = True
    ) -> torch.Tensor:
        features = self.backbone(x)
        if multi_head:
            return self.heads[str(task_id)](features)
        return self.heads["shared"](features)

    def to_local_targets(
        self, targets: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        offset = self.task_class_offsets.get(task_id, 0)
        return targets - offset