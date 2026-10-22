import numpy as np
from nn.layers import MaskedLinear


class MaskedAdam:
    """
    Adam with per-parameter step counters (t) and strict state isolation for masked weights.
    When a weight is masked (mask == 0):
      - Its weight, m, v, and step counter t are forced to 0.
      - Momentum cannot leak into masked weights.
      - If revived later (mask 0 -> 1), its bias correction starts cleanly at t=1.
    """

    def __init__(
        self, layers: list[MaskedLinear], lr=1e-3, betas=(0.9, 0.999), eps=1e-8
    ):
        self.layers = layers
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.state = {}
        for l in layers:
            for p in (l.W, l.b):
                self.state[id(p)] = {
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                    "t": np.zeros_like(p.data, dtype=np.int64),
                }

    def step(self):
        for l in self.layers:
            self._update_param(l.W, l.mask)
            self._update_param(l.b, np.ones_like(l.b.data))

    def _update_param(self, p, mask: np.ndarray):
        st = self.state[id(p)]
        active = mask > 0

        # Zero out gradient and state on masked indices immediately
        g = np.where(active, p.grad, 0.0)
        st["t"] = np.where(active, st["t"] + 1, 0)
        st["m"] = np.where(active, self.b1 * st["m"] + (1.0 - self.b1) * g, 0.0)
        st["v"] = np.where(active, self.b2 * st["v"] + (1.0 - self.b2) * (g**2), 0.0)

        # Compute bias-corrected moments safely only on active entries
        t_safe = np.maximum(st["t"], 1)
        m_hat = st["m"] / (1.0 - self.b1**t_safe)
        v_hat = st["v"] / (1.0 - self.b2**t_safe)

        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        p.data = np.where(active, p.data - update, 0.0)
