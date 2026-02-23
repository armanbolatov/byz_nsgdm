import torch
import torch.distributed as dist
import numpy as np

from codes.worker import ByzantineWorker


class ALittleIsEnoughAttack(ByzantineWorker):
    """ALIE (A Little Is Enough) attack.

    Byzantine workers send v_i = mean - z * std,
    where std is the coordinate-wise standard deviation of good
    workers' gradients.

    Args:
        n (int): Total number of workers
        m (int): Number of Byzantine workers
        z (float): Scaling factor (default 1.0)
    """

    def __init__(self, n, m, z=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z = z

    def get_gradient(self):
        return self._gradient

    def omniscient_callback(self):
        # Loop over good workers and accumulate their gradients
        gradients = []
        for w in self.simulator.workers:
            if not isinstance(w, ByzantineWorker):
                gradients.append(w.get_gradient())

        stacked_gradients = torch.stack(gradients, 1)
        mu = torch.mean(stacked_gradients, 1)
        std = torch.std(stacked_gradients, 1)

        # ALIE: v_i = mean - z * std
        self._gradient = mu - self.z * std

    def set_gradient(self, gradient) -> None:
        raise NotImplementedError

    def apply_gradient(self) -> None:
        raise NotImplementedError
