"""Replay memory buffers for continual learning strategies."""

from __future__ import annotations

import torch


class ExemplarBuffer:
    """Fixed-capacity memory buffer with reservoir sampling and non-blocking transfers.

    Pre-allocates tensor storage on CPU host to prevent reallocation churn.
    Supports asynchronous pinned memory transfers to GPU devices.
    """

    def __init__(self, max_size: int = 500, pin_memory: bool = False) -> None:
        """Initialize exemplar replay buffer.

        Args:
            max_size: Maximum number of sample exemplars to store across all tasks.
            pin_memory: Whether to allocate memory in page-locked CPU storage
                for faster host-to-device asynchronous DMA transfers.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self.max_size = max_size
        self.pin_memory = pin_memory and torch.cuda.is_available()

        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._t: torch.Tensor | None = None
        self._current_size: int = 0
        self.num_seen: int = 0

    @property
    def x(self) -> torch.Tensor | None:
        """Active stored input tensors shaped `[N, ...]` where `N <= max_size`."""
        return self._x[: self._current_size] if self._x is not None else None

    @property
    def y(self) -> torch.Tensor | None:
        """Active stored labels shaped `[N]` where `N <= max_size`."""
        return self._y[: self._current_size] if self._y is not None else None

    @property
    def t(self) -> torch.Tensor | None:
        """Active stored task IDs shaped `[N]` where `N <= max_size`."""
        return self._t[: self._current_size] if self._t is not None else None

    def _init_storage(self, sample_x: torch.Tensor) -> None:
        """Pre-allocate fixed CPU tensor memory on first sample ingestion."""
        data_shape = (self.max_size, *sample_x.shape[1:])
        self._x = torch.empty(data_shape, dtype=sample_x.dtype, device="cpu")
        self._y = torch.empty((self.max_size,), dtype=torch.long, device="cpu")
        self._t = torch.empty((self.max_size,), dtype=torch.long, device="cpu")

        if self.pin_memory:
            self._x = self._x.pin_memory()
            self._y = self._y.pin_memory()
            self._t = self._t.pin_memory()

    def add_samples(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> None:
        """Add a batch of samples into the buffer using unbiased reservoir sampling.

        Args:
            x: Batch tensor of inputs shaped `[B, ...]`.
            y: Batch tensor of labels shaped `[B]`.
            task_id: Global task integer ID.
        """
        if x.size(0) == 0:
            return

        if self._x is None:
            self._init_storage(x)

        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu().to(torch.long)
        batch_size = x_cpu.size(0)

        for i in range(batch_size):
            self.num_seen += 1
            if self._current_size < self.max_size:
                idx = self._current_size
                self._x[idx] = x_cpu[i]
                self._y[idx] = y_cpu[i]
                self._t[idx] = task_id
                self._current_size += 1
            else:
                # Reservoir sampling: select random index from [0, num_seen - 1]
                r = torch.randint(0, self.num_seen, (1,)).item()
                if r < self.max_size:
                    self._x[r] = x_cpu[i]
                    self._y[r] = y_cpu[i]
                    self._t[r] = task_id

    def sample(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        non_blocking: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a mini-batch uniformly from stored exemplars.

        Args:
            batch_size: Number of samples to draw.
            device: Target device to transfer sampled batch to.
            non_blocking: Whether to perform non-blocking copy when moving to GPU.

        Returns:
            Tuple of (x, y, t) tensors shaped `[min(B, N), ...]`, `[min(B, N)]`.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self._current_size == 0 or self._x is None:
            raise ValueError("Buffer is empty.")

        sample_size = min(batch_size, self._current_size)
        indices = torch.randperm(self._current_size)[:sample_size]

        bx = self._x[indices]
        by = self._y[indices]
        bt = self._t[indices]

        if device is not None:
            dev = torch.device(device)
            bx = bx.to(dev, non_blocking=non_blocking)
            by = by.to(dev, non_blocking=non_blocking)
            bt = bt.to(dev, non_blocking=non_blocking)

        return bx, by, bt

    def clear(self) -> None:
        """Reset the buffer and release stored exemplars."""
        self._current_size = 0
        self.num_seen = 0

    def __len__(self) -> int:
        """Return the number of valid exemplars currently stored."""
        return self._current_size
