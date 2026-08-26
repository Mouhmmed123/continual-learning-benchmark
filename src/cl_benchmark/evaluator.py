import numpy as np


class Evaluator:
    """Tracks the evaluation matrix R in R^(T x T) and computes metrics."""

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.R = np.zeros((num_tasks, num_tasks), dtype=np.float64)

    def update(
        self, training_task_id: int, test_task_id: int, accuracy: float
    ):
        self.R[training_task_id, test_task_id] = accuracy

    def average_accuracy(self) -> float:
        return float(np.mean(self.R[self.num_tasks - 1, :]))

    def backward_transfer(self) -> float:
        if self.num_tasks <= 1:
            return 0.0
        T = self.num_tasks
        final_accs = self.R[T - 1, : T - 1]
        diag_accs = np.diag(self.R)[: T - 1]
        return float(np.mean(final_accs - diag_accs))

    def forward_transfer(self) -> float:
        if self.num_tasks <= 1:
            return 0.0
        T = self.num_tasks
        fwt_sum = sum(self.R[i - 1, i] for i in range(1, T))
        return float(fwt_sum / (T - 1))