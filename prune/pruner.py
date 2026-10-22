import numpy as np
from nn.layers import MaskedLinear
from nn.optim import MaskedAdam


class SelfPruner:
    def __init__(
        self,
        layers: list[MaskedLinear],
        optim: MaskedAdam,
        target_sparsity: float,
        t_start: int,
        t_end: int,
        mode: str = "saliency",
        regrow_frac: float = 0.1,
    ):
        self.layers = layers
        self.optim = optim
        self.s_f = target_sparsity
        self.t_0 = t_start
        self.t_end = t_end
        self.mode = mode
        self.regrow_frac = regrow_frac
        # Accumulate moving average of |W * dL/dW| over mini-batches
        self.ema_saliency = [np.zeros_like(l.W.data) for l in layers]

    def accumulate_saliency(self, beta=0.9):
        for i, l in enumerate(self.layers):
            if self.mode == "saliency":
                score = np.abs(l.W.data * l.W.grad)
            else:
                score = np.abs(l.W.data)
            self.ema_saliency[i] = beta * self.ema_saliency[i] + (1.0 - beta) * score

    def cubic_sparsity(self, step: int) -> float:
        """Zhu & Gupta (2017) cubic sparsity schedule."""
        if step < self.t_0:
            return 0.0
        if step >= self.t_end:
            return self.s_f
        progress = (step - self.t_0) / max(1, (self.t_end - self.t_0))
        return self.s_f * (1.0 - (1.0 - progress) ** 3)

    def step_prune(self, step: int):
        if step < self.t_0 or step > self.t_end or self.s_f <= 0.0:
            return
        s_t = self.cubic_sparsity(step)
        total_params = sum(l.W.data.size for l in self.layers)
        k_keep = int(round(total_params * (1.0 - s_t)))
        if k_keep <= 0:
            return

        # 1. Global Importance Scoring across all layers
        all_scores = np.concatenate(
            [
                (self.ema_saliency[i] * l.mask).ravel()
                if self.mode == "saliency"
                else (np.abs(l.W.data) * l.mask).ravel()
                for i, l in enumerate(self.layers)
            ]
        )

        # Determine number of connections to drop & regrow (RigL style)
        do_regrow = (
            (self.mode == "saliency")
            and (self.regrow_frac > 0.0)
            and (step < self.t_end)
        )
        n_regrow = (
            int(k_keep * self.regrow_frac * (1.0 - step / self.t_end))
            if do_regrow
            else 0
        )
        k_survive = k_keep - n_regrow

        # Top k_survive active connections survive
        new_flat_mask = np.zeros_like(all_scores)
        if k_survive > 0:
            survive_idx = np.argpartition(all_scores, -k_survive)[-k_survive:]
            new_flat_mask[survive_idx] = 1.0

        # 2. Regrowth: revive pruned connections with highest counterfactual |grad|
        if n_regrow > 0:
            grad_scores = np.concatenate(
                [
                    (
                        np.abs(l.W.grad)
                        * (1.0 - new_flat_mask[self._slice(i)].reshape(l.W.grad.shape))
                    ).ravel()
                    for i, l in enumerate(self.layers)
                ]
            )
            revive_idx = np.argpartition(grad_scores, -n_regrow)[-n_regrow:]
            new_flat_mask[revive_idx] = 1.0

        # 3. Apply masks and scrub optimizer state on newly pruned entries
        offset = 0
        for l in self.layers:
            sz = l.W.data.size
            m = new_flat_mask[offset : offset + sz].reshape(l.W.data.shape)
            offset += sz
            l.mask = m
            l.W.data *= m
            st = self.optim.state[id(l.W)]
            st["m"] *= m
            st["v"] *= m
            st["t"] = np.where(m > 0, st["t"], 0)

    def _slice(self, layer_idx: int):
        start = sum(self.layers[i].W.data.size for i in range(layer_idx))
        return slice(start, start + self.layers[layer_idx].W.data.size)
