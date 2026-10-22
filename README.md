# AQUA Self-Pruning Network

Pure Python + NumPy reverse-mode autodiff engine, mask-aware Adam optimizer, and dynamic self-pruning MLP with first-order Taylor saliency and RigL gradient regrowth.

*Note on Scikit-learn:* `scikit-learn` is used strictly in `train/run.py` to load the standard 8x8 `digits` dataset and split/scale features. All autodiff, layers, optimizers, pruning, and sparse CSR inference are built from scratch in NumPy.

## Setup & Commands

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

# 1. Run Part 1 finite-difference gradient check & masked-weight optimizer tests
pytest tests/test_engine.py -v

# 2. Reproduce Part 2 baseline dense training
python train/run.py part2

# 3. Reproduce Part 3 self-pruning (90% target sparsity)
python train/run.py part3

# 4. Reproduce Part 4 Pareto sweep (0%, 50%, 75%, 90%, 95% across 5 seeds)
python train/run.py part4