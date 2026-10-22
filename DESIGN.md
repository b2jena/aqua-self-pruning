# Design & Theoretical Justification

## 1. Derivation of the Importance Criterion
Suppose we remove connection $w_{ij}$ by setting it to zero. By a first-order Taylor expansion of the loss $\mathcal{L}(W)$ around the current weights $W$:
$$\mathcal{L}(W - w_{ij} e_{ij}) - \mathcal{L}(W) \approx -w_{ij} \frac{\partial \mathcal{L}}{\partial w_{ij}} + \mathcal{O}(w_{ij}^2)$$
Since we seek connections whose removal minimally perturbs the loss in either direction (and because mini-batch stochastic gradients fluctuate in sign), we bound the expected absolute perturbation via an exponential moving average of the first-order saliency:
$$S_{ij}^{(t)} = \beta S_{ij}^{(t-1)} + (1 - \beta) \left\vert{} w_{ij} \frac{\partial \mathcal{L}}{\partial w_{ij}} \right\vert{}$$
Unlike pure magnitude pruning ($\vert{}w_{ij}\vert{}$), which assumes an isotropic loss landscape where all parameters have identical curvature/sensitivity, Taylor saliency preserves small weights sitting on steep loss cliffs ($\left\vert{}\frac{\partial \mathcal{L}}{\partial w_{ij}}\right\vert{} \gg 0$) while pruning larger weights in flat plateaus ($\frac{\partial \mathcal{L}}{\partial w_{ij}} \approx 0$).

## 2. The Gradient of a Masked Weight & Optimizer State Integrity
In `MaskedLinear.forward`, the forward pass computes $Y = X (W \odot M) + b$, ensuring pruned connections contribute strictly zero. In the backward pass, we compute:
$$\frac{\partial \mathcal{L}}{\partial W_{\text{eff}}} = X^\top \frac{\partial \mathcal{L}}{\partial Y}$$
**Why we do NOT multiply `W.grad` by the mask inside the backward node:**
1. **Counterfactual Sensitivity for Regrowth (RigL):** For a masked weight ($M_{ij}=0$), $X^\top \frac{\partial \mathcal{L}}{\partial Y}$ is the exact directional derivative of the loss if connection $(i,j)$ were infinitesimally restored at zero. Zeroing `W.grad` inside the graph blinds the pruner and makes gradient-based regrowth impossible.
2. **Strict Mask Isolation in `MaskedAdam`:** Instead of relying on a zero gradient (where Adam's momentum $m_t = \beta_1 m_{t-1}$ would still decay and leak updates into masked weights!), `MaskedAdam` explicitly forces $w_{ij} = 0, m_{ij} = 0, v_{ij} = 0$, and resets the per-weight step counter $t_{ij} = 0$ whenever $M_{ij} = 0$. When a connection is revived ($M_{ij}: 0 \to 1$), $t_{ij}$ increments from $0 \to 1$, preventing division by a near-zero $\sqrt{\hat{v}_t}$ from stale steps.

## 3. Autodiff Engine Bottlenecks & Optimizations
1. **Memory Allocation in Backward Passes:** Every operator allocates new NumPy arrays and closures. *Fix:* Pre-allocate a tape with Reusable Scratch Buffers (arena allocation) and fuse element-wise ops (e.g. `Linear + ReLU`).
2. **Dense-to-Sparse Crossover:** In pure Python, iterating over CSR rows has loop overhead at low sparsity (<75%), whereas at high sparsity (>=90%), CSR skips 90% of multiply-adds. *Fix:* Lower the CSR inner loop to a Cython/Numba/SIMD kernel or block-sparse (BCSR 16x16 tiles) to saturate tensor cores.

## 4. Serving Self-Pruned Models in Multi-Tenant Inference at Scale
1. **Shared Frozen Backbone + Sparse Tenant Adapters:** Store the dense/semi-sparse base model once in GPU HBM; represent tenant-specific self-pruned deltas as compressed CSR/BCSR masks + values.