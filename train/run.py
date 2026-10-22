import argparse, json, time
import os
import sys

# Ensure project root is in sys.path when running from within train/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from engine.tensor import Tensor
from nn.layers import MLP
from nn.optim import MaskedAdam
from prune.pruner import SelfPruner


def load_data(seed=42):
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    y = digits.target
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def train_model(
    seed=42, epochs=40, batch_size=64, lr=3e-3, sparsity=0.0, mode="saliency"
):
    rng = np.random.default_rng(seed)
    X_tr, X_te, y_tr, y_te = load_data(seed)
    model = MLP([64, 128, 64, 10], rng)
    optim = MaskedAdam(model.layers, lr=lr)

    steps_per_epoch = int(np.ceil(len(X_tr) / batch_size))
    total_steps = epochs * steps_per_epoch
    pruner = SelfPruner(
        model.layers,
        optim,
        sparsity,
        t_start=int(0.15 * total_steps),
        t_end=int(0.75 * total_steps),
        mode=mode,
    )

    history = []
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(len(X_tr))
        ep_loss = 0.0
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Tensor(X_tr[idx]), y_tr[idx]
            model.zero_grad()
            logits = model.forward(xb)
            loss = logits.softmax_cross_entropy(yb)
            loss.backward()

            pruner.accumulate_saliency()
            optim.step()
            if step % 10 == 0:
                pruner.step_prune(step)
            ep_loss += float(loss.data) * len(idx)
            step += 1

        preds = np.argmax(model.forward(Tensor(X_te)).data, axis=1)
        acc = float(np.mean(preds == y_te))
        sp, active, total = model.sparsity_stats()
        history.append(
            {"epoch": ep + 1, "loss": ep_loss / len(X_tr), "acc": acc, "sparsity": sp}
        )

    # Benchmark Dense vs True CSR Sparse Inference Cost
    model.compile_sparse()
    X_bench = np.repeat(X_te, 20, axis=0)
    t0 = time.perf_counter()
    _ = model.forward(Tensor(X_bench)).data
    dense_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    sparse_out = model.forward_sparse(X_bench)
    sparse_ms = (time.perf_counter() - t0) * 1000.0
    np.testing.assert_allclose(
        model.forward(Tensor(X_te)).data, model.forward_sparse(X_te), atol=1e-10
    )

    sp, active, total = model.sparsity_stats()
    return {
        "final_acc": history[-1]["acc"],
        "achieved_sparsity": sp,
        "active_params": active,
        "total_params": total,
        "dense_flops_per_sample": 2 * total,
        "sparse_flops_per_sample": 2 * active,
        "dense_ms": round(dense_ms, 2),
        "sparse_ms": round(sparse_ms, 2),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["part2", "part3", "part4"])
    args = parser.parse_args()

    if args.task == "part2":
        res = train_model(sparsity=0.0)
        print(
            f"[Part 2] Dense Acc: {res['final_acc'] * 100:.2f}% | Active Params: {res['active_params']}"
        )
    elif args.task == "part3":
        res = train_model(sparsity=0.90, mode="saliency")
        print(
            f"[Part 3] 90% Saliency Pruned Acc: {res['final_acc'] * 100:.2f}% | "
            f"Sparsity: {res['achieved_sparsity'] * 100:.1f}% | "
            f"FLOPs/sample: {res['sparse_flops_per_sample']} (vs {res['dense_flops_per_sample']} dense) | "
            f"Dense: {res['dense_ms']}ms vs CSR Sparse: {res['sparse_ms']}ms"
        )
    elif args.task == "part4":
        sparsities = [0.0, 0.50, 0.75, 0.90, 0.95]
        seeds = [42, 43, 44, 45, 46]
        results = {"saliency": {}, "magnitude": {}}
        for mode in ["saliency", "magnitude"]:
            for sp in sparsities:
                runs = [train_model(seed=s, sparsity=sp, mode=mode) for s in seeds]
                accs = [r["final_acc"] for r in runs]
                results[mode][str(sp)] = {
                    "mean_acc": float(np.mean(accs)),
                    "std_acc": float(np.std(accs)),
                    "sparse_flops": runs[0]["sparse_flops_per_sample"],
                    "dense_ms": runs[0]["dense_ms"],
                    "sparse_ms": runs[0]["sparse_ms"],
                }
                print(
                    f"{mode:10s} | Sparsity {sp * 100:4.0f}% | Acc: {np.mean(accs) * 100:.2f}% ± {np.std(accs) * 100:.2f}%"
                )

        with open("pareto_results.json", "w") as f:
            json.dump(results, f, indent=2)

        plt.figure(figsize=(7, 4.5))
        for mode, marker in [("saliency", "o"), ("magnitude", "s")]:
            means = [results[mode][str(s)]["mean_acc"] * 100 for s in sparsities]
            stds = [results[mode][str(s)]["std_acc"] * 100 for s in sparsities]
            plt.errorbar(
                [s * 100 for s in sparsities],
                means,
                yerr=stds,
                marker=marker,
                capsize=4,
                label=mode.title(),
            )
        plt.xlabel("Sparsity (% Connections Removed)")
        plt.ylabel("Test Accuracy (%)")
        plt.title("Sparsity–Accuracy Pareto Curve (5 Seeds)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("pareto_curve.png", dpi=150)
        print("Saved pareto_results.json and pareto_curve.png")
