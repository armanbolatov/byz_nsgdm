"""LM worker for Byzantine-robust training."""

import torch
from collections import defaultdict
from typing import Tuple


class LMWorker:
    def __init__(
        self,
        momentum: float,
        data_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.momentum = momentum
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.running = {}
        self.metrics = {}
        self.state = defaultdict(dict)

    def add_metric(self, name: str, callback):
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")
        self.metrics[name] = callback

    def add_metrics(self, metrics: dict):
        for name in metrics:
            self.add_metric(name, metrics[name])

    def __str__(self) -> str:
        return "LMWorker"

    def train_epoch_start(self) -> None:
        self.running["train_loader_iterator"] = iter(self.data_loader)
        self.model.train()

    def compute_gradient(self) -> Tuple[float, int]:
        results = {}

        x, y = self.running["train_loader_iterator"].__next__()
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        logits, loss = self.model(x, targets=y)
        loss.backward()
        self._save_grad()

        self.running["x"] = x
        self.running["y"] = y

        results["loss"] = loss.item()
        results["length"] = x.size(0)
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric(logits, y)
        return results

    def get_gradient(self) -> torch.Tensor:
        return self._get_saved_grad()

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            end = beg + len(p.grad.view(-1))
            x = gradient[beg:end].reshape_as(p.grad.data)
            p.grad.data = x.clone().detach()
            beg = end

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(p.grad).detach()
                else:
                    param_state["momentum_buffer"].mul_(self.momentum).add_(p.grad)

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                layer_gradients.append(param_state["momentum_buffer"].data.view(-1))
        return torch.cat(layer_gradients)


class LMByzantineWorker(LMWorker):
    def configure(self, simulator):
        self.simulator = simulator
        simulator.register_omniscient_callback(self.omniscient_callback)

    def omniscient_callback(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return "LMByzantineWorker"


class LMBitFlippingWorker(LMByzantineWorker):
    def get_gradient(self) -> torch.Tensor:
        return -super().get_gradient()

    def omniscient_callback(self):
        pass

    def __str__(self) -> str:
        return "LMBitFlippingWorker"


class LMMimicWorker(LMByzantineWorker):
    def __init__(self, warmup: int = 50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup = warmup
        self.iteration = 0
        self.attack_gradient = None

    def get_gradient(self) -> torch.Tensor:
        if self.iteration < self.warmup:
            return super().get_gradient()
        else:
            return self.attack_gradient

    def omniscient_callback(self):
        self.iteration += 1
        if self.iteration >= self.warmup:
            honest_grads = []
            for w in self.simulator.workers:
                if not isinstance(w, LMByzantineWorker):
                    honest_grads.append(w.get_gradient())
            if honest_grads:
                mean_grad = torch.stack(honest_grads).mean(dim=0)
                self.attack_gradient = -2 * mean_grad

    def __str__(self) -> str:
        return "LMMimicWorker"


class LMALIEWorker(LMByzantineWorker):
    def __init__(self, n: int, m: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.m = m
        self.attack_gradient = None

    def get_gradient(self) -> torch.Tensor:
        if self.attack_gradient is not None:
            return self.attack_gradient
        return super().get_gradient()

    def omniscient_callback(self):
        honest_grads = []
        for w in self.simulator.workers:
            if not isinstance(w, LMByzantineWorker):
                honest_grads.append(w.get_gradient())
        if honest_grads:
            stacked = torch.stack(honest_grads)
            mean_grad = stacked.mean(dim=0)
            # ALIE: shift by factor based on Byzantine fraction
            factor = 2.0 * self.m / (self.n - self.m)
            self.attack_gradient = mean_grad - factor * mean_grad

    def __str__(self) -> str:
        return "LMALIEWorker"
