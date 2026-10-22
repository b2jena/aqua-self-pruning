import numpy as np


def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Sum out broadcasted dimensions so gradient matches target_shape."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._prev = set(_children)
        self._backward = lambda: None

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self):
        if not self.requires_grad:
            return
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            (self, other),
            self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            (self, other),
            self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            (self, other),
            self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(
                    -out.grad * self.data / (other.data**2), other.data.shape
                )

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            (self, other),
            self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims), (self,), self.requires_grad
        )

        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    axes = [axis] if isinstance(axis, int) else axis
                    for ax in sorted(axes):
                        g = np.expand_dims(g, axis=ax)
                self.grad += np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = (
            self.data.size
            if axis is None
            else np.prod(
                [
                    self.data.shape[i]
                    for i in ([axis] if isinstance(axis, int) else axis)
                ]
            )
        )
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1.0 - t**2)

        out._backward = _backward
        return out

    def softmax_cross_entropy(self, targets: np.ndarray):
        """Numerically stable log-softmax + NLL loss for integer class targets (N,)."""
        shifted = self.data - np.max(self.data, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        N = self.data.shape[0]
        loss_val = (
            -np.sum(np.log(np.clip(probs[np.arange(N), targets], 1e-15, 1.0))) / N
        )
        out = Tensor(loss_val, (self,), self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = probs.copy()
                grad[np.arange(N), targets] -= 1.0
                self.grad += out.grad * (grad / N)

        out._backward = _backward
        return out
