import numpy as np
from engine.tensor import Tensor


class MaskedLinear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        # He Initialization for ReLU stability: std = sqrt(2 / fan_in)
        std = np.sqrt(2.0 / in_features)
        self.W = Tensor(
            rng.normal(0.0, std, (in_features, out_features)), requires_grad=True
        )
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)
        self.mask = np.ones((in_features, out_features), dtype=np.float64)
        # CSR sparse representation for Part 4 real cost measurement
        self.csr_data, self.csr_indices, self.csr_indptr = None, None, None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward applies mask cleanly so masked weights contribute 0.
        Backward passes counterfactual gradient dL/dW_eff directly to self.W.grad
        so importance & regrowth criteria can inspect inactive connection gradients,
        while the optimizer enforces zero update/state on masked weights.
        """
        eff_W_data = self.W.data * self.mask
        out_matmul = Tensor(
            x.data @ eff_W_data, (x, self.W), x.requires_grad or self.W.requires_grad
        )

        def _backward():
            if x.requires_grad:
                x.grad += out_matmul.grad @ eff_W_data.T
            if self.W.requires_grad:
                self.W.grad += x.data.T @ out_matmul.grad

        out_matmul._backward = _backward
        return out_matmul + self.b

    def compile_sparse(self):
        """Convert active weights (W * mask) to CSR format for zero-skipping inference."""
        W_eff = self.W.data * self.mask
        data, indices, indptr = [], [], [0]
        for row in W_eff:
            nz = np.nonzero(row)[0]
            data.extend(row[nz])
            indices.extend(nz)
            indptr.append(len(data))
        self.csr_data = np.array(data, dtype=np.float64)
        self.csr_indices = np.array(indices, dtype=np.int32)
        self.csr_indptr = np.array(indptr, dtype=np.int32)

    def forward_sparse(self, x_np: np.ndarray) -> np.ndarray:
        """True sparse forward pass iterating only over non-zero weights."""
        N, In = x_np.shape
        Out = self.b.data.shape[1]
        out = np.broadcast_to(self.b.data, (N, Out)).copy()
        for i in range(In):
            start, end = self.csr_indptr[i], self.csr_indptr[i + 1]
            if start < end:
                cols = self.csr_indices[start:end]
                vals = self.csr_data[start:end]
                out[:, cols] += x_np[:, i : i + 1] * vals
        return out


class MLP:
    def __init__(self, dims: list[int], rng: np.random.Generator):
        self.layers = [
            MaskedLinear(dims[i], dims[i + 1], rng) for i in range(len(dims) - 1)
        ]

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x

    def forward_sparse(self, x_np: np.ndarray) -> np.ndarray:
        for i, layer in enumerate(self.layers):
            x_np = layer.forward_sparse(x_np)
            if i < len(self.layers) - 1:
                x_np = np.maximum(0.0, x_np)
        return x_np

    def compile_sparse(self):
        for l in self.layers:
            l.compile_sparse()

    def zero_grad(self):
        for l in self.layers:
            l.W.zero_grad()
            l.b.zero_grad()

    def sparsity_stats(self):
        total = sum(l.mask.size for l in self.layers)
        active = sum(int(np.sum(l.mask)) for l in self.layers)
        return 1.0 - (active / total), active, total
