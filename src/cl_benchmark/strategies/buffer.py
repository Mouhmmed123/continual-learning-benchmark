import torch


class ExemplarBuffer:
    """Fixed-capacity memory buffer supporting reservoir sampling."""

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self.t: torch.Tensor | None = None
        self.num_seen: int = 0

    def add_samples(self, x: torch.Tensor, y: torch.Tensor, task_id: int):
        t = torch.full(
            (x.size(0),), task_id, dtype=torch.long, device=x.device
        )

        if self.x is None:
            n_add = min(x.size(0), self.max_size)
            self.x = x[:n_add].clone().cpu()
            self.y = y[:n_add].clone().cpu()
            self.t = t[:n_add].clone().cpu()
            self.num_seen = n_add
        else:
            for i in range(x.size(0)):
                self.num_seen += 1
                if len(self.x) < self.max_size:
                    self.x = torch.cat([self.x, x[i : i + 1].cpu()], dim=0)
                    self.y = torch.cat([self.y, y[i : i + 1].cpu()], dim=0)
                    self.t = torch.cat([self.t, t[i : i + 1].cpu()], dim=0)
                else:
                    idx = torch.randint(0, self.num_seen, (1,)).item()
                    if idx < self.max_size:
                        self.x[idx] = x[i].cpu()
                        self.y[idx] = y[i].cpu()
                        self.t[idx] = t[i].cpu()

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.x is None or len(self.x) == 0:
            raise ValueError("Buffer is empty.")

        sample_size = min(batch_size, len(self.x))
        indices = torch.randperm(len(self.x))[:sample_size]
        return self.x[indices], self.y[indices], self.t[indices]

    def __len__(self) -> int:
        return len(self.x) if self.x is not None else 0