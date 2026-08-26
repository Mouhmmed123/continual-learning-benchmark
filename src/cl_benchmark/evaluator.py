"""Evaluation metrics and tracking for continual learning benchmark protocols."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class Evaluator:
    """Tracks evaluation matrix R in R^(T x T) and computes continual learning metrics.

    Let R[i, j] denote the test accuracy on task j after training on task i.
    Supports computing:
    - Average Accuracy (A_k)
    - Backward Transfer (BWT_k)
    - Forward Transfer (FWT)
    - Forgetting Measure (FM_k)
    """

    def __init__(self, num_tasks: int) -> None:
        """Initialize evaluator with task count.

        Args:
            num_tasks: Total number of tasks in the continual benchmark stream.
        """
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be at least 1, got {num_tasks}")

        self.num_tasks = num_tasks
        self.R = np.zeros((num_tasks, num_tasks), dtype=np.float64)

    def update(self, training_task_id: int, test_task_id: int, accuracy: float) -> None:
        """Record accuracy for test_task_id after training on training_task_id.

        Args:
            training_task_id: Index of current/last trained task (0-indexed).
            test_task_id: Index of evaluated test task (0-indexed).
            accuracy: Test accuracy value in range [0.0, 1.0].
        """
        if not (0 <= training_task_id < self.num_tasks):
            raise IndexError(
                f"training_task_id {training_task_id} out of bounds "
                f"for num_tasks={self.num_tasks}"
            )
        if not (0 <= test_task_id < self.num_tasks):
            raise IndexError(
                f"test_task_id {test_task_id} out of bounds "
                f"for num_tasks={self.num_tasks}"
            )

        self.R[training_task_id, test_task_id] = float(accuracy)

    def average_accuracy(self, task_id: int | None = None) -> float:
        """Compute Average Accuracy after training up to task_id.

        Formula: A_k = (1 / (k + 1)) * sum_{j=0}^{k} R[k, j].
        If task_id is None, defaults to evaluating after the final task (k = T - 1).

        Args:
            task_id: Task index after which to evaluate (0-indexed).

        Returns:
            Mean accuracy across tasks observed up to task_id.
        """
        k = self.num_tasks - 1 if task_id is None else task_id
        if not (0 <= k < self.num_tasks):
            raise IndexError(
                f"task_id {k} out of bounds for num_tasks={self.num_tasks}"
            )

        return float(np.mean(self.R[k, : k + 1]))

    def backward_transfer(self, task_id: int | None = None) -> float:
        """Compute Backward Transfer (BWT) after training up to task_id.

        Formula: BWT_k = (1 / k) * sum_{j=0}^{k-1} (R[k, j] - R[j, j]).
        Measures the average change in accuracy on previously learned tasks.

        Args:
            task_id: Task index after which to evaluate (0-indexed). Defaults to T - 1.

        Returns:
            Backward transfer metric. Returns 0.0 if k == 0 or T <= 1.
        """
        k = self.num_tasks - 1 if task_id is None else task_id
        if not (0 <= k < self.num_tasks):
            raise IndexError(
                f"task_id {k} out of bounds for num_tasks={self.num_tasks}"
            )

        if k == 0:
            return 0.0

        final_accs = self.R[k, :k]
        diag_accs = np.diag(self.R)[:k]
        return float(np.mean(final_accs - diag_accs))

    def forward_transfer(
        self, random_baseline: Sequence[float] | np.ndarray | None = None
    ) -> float:
        """Compute Forward Transfer (FWT) across all tasks.

        Formula: FWT = (1 / (T - 1)) * sum_{j=1}^{T-1} (R[j-1, j] - b_j),
        where b_j is the zero-shot / random baseline accuracy for task j.

        Args:
            random_baseline: Optional baseline accuracy array shaped [T].

        Returns:
            Forward transfer score. Returns 0.0 if T <= 1.
        """
        if self.num_tasks <= 1:
            return 0.0

        T = self.num_tasks
        fwt_diffs = []
        for j in range(1, T):
            b_j = float(random_baseline[j]) if random_baseline is not None else 0.0
            fwt_diffs.append(self.R[j - 1, j] - b_j)

        return float(np.mean(fwt_diffs))

    def forgetting_measure(self, task_id: int | None = None) -> float:
        """Compute Forgetting Measure (FM) after training up to task_id.

        Formula: FM_k = (1 / k) * sum_{j=0}^{k-1} (max_{l} R[l, j] - R[k, j]).
        Measures the maximum performance degradation on past tasks.

        Args:
            task_id: Task index after which to evaluate. Defaults to T - 1.

        Returns:
            Mean forgetting across previous tasks. Returns 0.0 if k == 0 or T <= 1.
        """
        k = self.num_tasks - 1 if task_id is None else task_id
        if not (0 <= k < self.num_tasks):
            raise IndexError(
                f"task_id {k} out of bounds for num_tasks={self.num_tasks}"
            )

        if k == 0:
            return 0.0

        forgetting_vals = []
        for j in range(k):
            peak_acc = np.max(self.R[:k, j])
            current_acc = self.R[k, j]
            forgetting_vals.append(max(0.0, peak_acc - current_acc))

        return float(np.mean(forgetting_vals))

    def get_metrics(
        self,
        task_id: int | None = None,
        random_baseline: Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute and return all evaluation metrics as a summary dictionary.

        Args:
            task_id: Task index to evaluate. Defaults to final task T - 1.
            random_baseline: Optional random baseline accuracy per task.

        Returns:
            Dictionary containing accuracy, bwt, fwt, and forgetting metrics.
        """
        return {
            "average_accuracy": self.average_accuracy(task_id=task_id),
            "backward_transfer": self.backward_transfer(task_id=task_id),
            "forward_transfer": self.forward_transfer(random_baseline=random_baseline),
            "forgetting_measure": self.forgetting_measure(task_id=task_id),
        }
