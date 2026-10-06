"""
Redraw HW1 Q1 figures from Spark-run costs (and optional GCS assignments).

Usage:
  python plot_q1_results.py
"""

import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
SUMMARY = OUT / "q1_summary.txt"
DATA = BASE / "data.txt"
ASSIGN_C1 = OUT / "assignments_L2_c1.txt"
ASSIGN_C2 = OUT / "assignments_L2_c2.txt"
RANDOM_SEED = 42


def parse_costs(summary_text: str) -> dict[str, list[float]]:
    """Parse 'L1/c1: ... costs=[...]' lines from Spark summary."""
    costs: dict[str, list[float]] = {}
    for line in summary_text.splitlines():
        m = re.match(r"(L[12])/(c[12]):", line.strip())
        if not m:
            continue
        start = line.find("costs=")
        if start < 0:
            continue
        bracket = line[start + len("costs=") :].strip()
        if not bracket.startswith("["):
            continue
        end = bracket.find("]")
        if end < 0:
            continue
        arr = bracket[: end + 1]
        key = f"{m.group(1)}_{m.group(2)}"
        costs[key] = [float(x) for x in ast.literal_eval(arr)]
    return costs


def plot_costs(costs: dict[str, list[float]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for norm in (1, 2):
        c1 = costs[f"L{norm}_c1"]
        c2 = costs[f"L{norm}_c2"]
        iters = list(range(1, len(c1) + 1))
        for name, series in (("c1", c1), ("c2", c2)):
            plt.figure(figsize=(8, 5))
            plt.plot(iters, series, marker="o")
            plt.xlabel("Iteration")
            plt.ylabel("Within-cluster cost")
            plt.title(f"K-means cost — L{norm}, init={name}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            path = OUT / f"cost_L{norm}_{name}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"Saved {path.name}")

        plt.figure(figsize=(8, 5))
        plt.plot(iters, c1, marker="o", label="c1 (random init)")
        plt.plot(iters, c2, marker="s", label="c2 (farthest init)")
        plt.xlabel("Iteration")
        plt.ylabel("Within-cluster cost")
        plt.title(f"K-means within-cluster cost (L{norm})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = OUT / f"cost_L{norm}_compare.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path.name}")


def load_assignments(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    labels: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            labels.append(int(float(line)))
    return labels


def plot_tsne() -> None:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not installed; skip t-SNE")
        return

    if not DATA.exists():
        print("data.txt missing; skip t-SNE")
        return

    points = np.loadtxt(DATA)
    for init_name, assign_path in (("c1", ASSIGN_C1), ("c2", ASSIGN_C2)):
        labels = load_assignments(assign_path)
        if labels is None:
            print(f"Missing {assign_path.name}; skip tsne_L2_{init_name}.png")
            continue
        if len(labels) != len(points):
            raise ValueError(
                f"{assign_path.name} has {len(labels)} labels, data has {len(points)}"
            )
        embedded = TSNE(
            n_components=2,
            random_state=RANDOM_SEED,
            perplexity=30,
            init="pca",
            learning_rate="auto",
        ).fit_transform(points)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embedded[:, 0],
            embedded[:, 1],
            c=labels,
            cmap="tab10",
            s=8,
            alpha=0.75,
        )
        plt.colorbar(scatter, label="Cluster")
        plt.title(f"t-SNE of L2 clustering (init={init_name})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        path = OUT / f"tsne_L2_{init_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path.name}")


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(f"Missing {SUMMARY}")
    costs = parse_costs(SUMMARY.read_text(encoding="utf-8"))
    needed = ["L1_c1", "L1_c2", "L2_c1", "L2_c2"]
    missing = [k for k in needed if k not in costs]
    if missing:
        raise SystemExit(f"Could not parse costs for: {missing}")
    plot_costs(costs)
    plot_tsne()
    print("Done.")


if __name__ == "__main__":
    main()
