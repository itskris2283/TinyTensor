import numpy as np

from tinytensor.tensor import Tensor, tensor


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, params: list[Tensor], lr: float = 0.01, momentum: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        if momentum > 0.0:
            self.velocities = [np.zeros(p.shape, dtype=np.float32) for p in params]
        else:
            self.velocities = []

    def step(self, grads: list[Tensor]) -> None:
        """Update parameters with computed gradients."""
        for i, (p, g) in enumerate(zip(self.params, grads, strict=False)):
            p_np = p.numpy()
            g_np = g.numpy()

            if self.momentum > 0.0:
                self.velocities[i] = self.momentum * self.velocities[i] + g_np
                update = self.velocities[i]
            else:
                update = g_np

            p_np -= self.lr * update
            # Write updated values back into tensor buffer
            new_p = tensor(p_np, dtype=p.dtype)
            p.view(p.shape)  # touch
            # Direct buffer copy
            np.copyto(p.numpy(), new_p.numpy())
