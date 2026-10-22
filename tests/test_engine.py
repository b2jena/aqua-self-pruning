import os
import sys

# Ensure project root is in sys.path for direct pytest calls
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from engine.tensor import Tensor
from nn.layers import MaskedLinear
from nn.optim import MaskedAdam


def numerical_grad(f, x: Tensor, eps=1e-6):
    num_g = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x.data[idx]
        x.data[idx] = orig + eps
        pos = float(f().data)
        x.data[idx] = orig - eps
        neg = float(f().data)
        x.data[idx] = orig
        num_g[idx] = (pos - neg) / (2.0 * eps)
        it.iternext()
    return num_g


def test_all_ops_and_broadcasting():
    rng = np.random.default_rng(42)
    X = Tensor(rng.normal(size=(4, 5)), requires_grad=True)
    W = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
    b = Tensor(rng.normal(size=(3,)), requires_grad=True)  # (4,3) + (3,) broadcast
    scale = Tensor(rng.uniform(0.5, 1.5, size=(4, 1)), requires_grad=True)
    targets = np.array([0, 2, 1, 0])

    def forward():
        z = ((X @ W + b) * scale) / 1.5 - 0.2
        h = z.relu() + z.tanh()
        loss = h.softmax_cross_entropy(targets) + (h.sum() * 0.01) + (h.mean() * 0.05)
        return loss

    X.zero_grad()
    W.zero_grad()
    b.zero_grad()
    scale.zero_grad()
    loss = forward()
    loss.backward()
    for param in [X, W, b, scale]:
        ng = numerical_grad(forward, param)
        np.testing.assert_allclose(param.grad, ng, rtol=1e-5, atol=1e-6)


def test_masked_weight_and_revival_integrity():
    """
    Verifies:
    1. Masked weight contributes strictly 0 to forward pass.
    2. Optimizer state (m, v, t) does not drift while weight is masked.
    3. Reviving a weight resets bias-correction cleanly without exploding updates.
    """
    rng = np.random.default_rng(7)
    layer = MaskedLinear(4, 3, rng)
    optim = MaskedAdam([layer], lr=1e-2)
    x = Tensor(rng.normal(size=(5, 4)))
    y = np.array([0, 1, 2, 0, 1])

    # Step 1: Normal step so moments (m, v, t) are populated
    loss = layer.forward(x).softmax_cross_entropy(y)
    loss.backward()
    optim.step()
    assert optim.state[id(layer.W)]["t"][0, 0] == 1

    # Step 2: Mask weight (0, 0) and verify forward invariance to W[0, 0]
    layer.mask[0, 0] = 0.0
    layer.W.data[0, 0] = 999.0  # Even if corrupted before step, forward ignores it
    out1 = layer.forward(x).data.copy()
    layer.W.data[0, 0] = -999.0
    out2 = layer.forward(x).data.copy()
    np.testing.assert_allclose(out1, out2, atol=1e-12)

    # Step 3: Run 5 steps while masked -> verify W[0,0], m[0,0], v[0,0], t[0,0] are strictly 0
    for _ in range(5):
        layer.W.zero_grad()
        layer.b.zero_grad()
        layer.forward(x).softmax_cross_entropy(y).backward()
        optim.step()
        st = optim.state[id(layer.W)]
        assert layer.W.data[0, 0] == 0.0
        assert st["m"][0, 0] == 0.0 and st["v"][0, 0] == 0.0 and st["t"][0, 0] == 0

    # Step 4: Revive connection (0, 0) -> t[0, 0] starts at 1, finite update
    layer.mask[0, 0] = 1.0
    layer.W.zero_grad()
    layer.b.zero_grad()
    layer.forward(x).softmax_cross_entropy(y).backward()
    optim.step()
    st = optim.state[id(layer.W)]
    assert st["t"][0, 0] == 1
    assert np.isfinite(layer.W.data[0, 0]) and layer.W.data[0, 0] != 0.0
