"""HW1 Q1: Iterative K-means on Spark (k=10, 20 iterations)."""

import operator
import sys

import numpy as np
from pyspark import SparkContext

MAX_ITER = 20
BUCKET = "eecs-e6893-big-data-analysis-hw1"
DATA_PATH = f"gs://{BUCKET}/hw1/data.txt"
C1_PATH = f"gs://{BUCKET}/hw1/c1.txt"
C2_PATH = f"gs://{BUCKET}/hw1/c2.txt"
OUTPUT_URI = f"gs://{BUCKET}/hw1/output/q1"


def closest(p: np.ndarray, centroids: list[np.ndarray], norm: int) -> int:
    return min(
        ((i, float(np.linalg.norm(p - c, ord=norm))) for i, c in enumerate(centroids)),
        key=operator.itemgetter(1),
    )[0]


def point_cost(p: np.ndarray, centroid: np.ndarray, norm: int) -> float:
    if norm == 2:
        return float(np.sum((p - centroid) ** 2))
    return float(np.linalg.norm(p - centroid, ord=1))


def kmeans(data, centroids: list[np.ndarray], norm: int) -> tuple[list[float], list[int]]:
    current = [c.copy() for c in centroids]
    costs: list[float] = []
    for _ in range(MAX_ITER):
        cents = current
        assigned = data.map(lambda p, cents=cents, n=norm: (closest(p, cents, n), (p, 1)))
        costs.append(
            float(
                assigned.map(
                    lambda kv, cents=cents, n=norm: point_cost(kv[1][0], cents[kv[0]], n)
                ).sum()
            )
        )
        new_map = (
            assigned.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
            .mapValues(lambda sc: sc[0] / sc[1])
            .collectAsMap()
        )
        current = [
            np.array(new_map[i], dtype=float) if i in new_map else old.copy()
            for i, old in enumerate(current)
        ]
    assignments = data.map(lambda p, cents=current, n=norm: closest(p, cents, n)).collect()
    return costs, assignments


def save_lines(sc: SparkContext, uri: str, lines: list[str]) -> None:
    sc.parallelize(lines, 1).saveAsTextFile(uri)


def load_vectors(sc: SparkContext, path: str) -> object:
    return sc.textFile(path).map(
        lambda line: np.array([float(x) for x in line.split()], dtype=float)
    )


def main() -> None:
    args = sys.argv[1:]
    data_path, c1_path, c2_path, output_uri = (
        args if len(args) == 4 else [DATA_PATH, C1_PATH, C2_PATH, OUTPUT_URI]
    )

    sc = SparkContext(appName="HW1-Q1")
    sc.setLogLevel("WARN")

    data = load_vectors(sc, data_path).cache()
    c1 = load_vectors(sc, c1_path).collect()
    c2 = load_vectors(sc, c2_path).collect()

    for norm in (1, 2):
        by_init: dict[str, list[float]] = {}
        for name, init in (("c1", c1), ("c2", c2)):
            costs, assignments = kmeans(data, init, norm)
            by_init[name] = costs
            print(f"L{norm}/{name}: final={costs[-1]:.6f}; costs={[round(c, 6) for c in costs]}")
            save_lines(sc, f"{output_uri}/costs_L{norm}_{name}", [str(c) for c in costs])
            if norm == 2:
                save_lines(
                    sc, f"{output_uri}/assignments_L2_{name}", [str(x) for x in assignments]
                )
        save_lines(
            sc,
            f"{output_uri}/costs_L{norm}_compare",
            ["iteration,c1,c2"]
            + [f"{i + 1},{by_init['c1'][i]},{by_init['c2'][i]}" for i in range(MAX_ITER)],
        )

    sc.stop()


if __name__ == "__main__":
    main()
