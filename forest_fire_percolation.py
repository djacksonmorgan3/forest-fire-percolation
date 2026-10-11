"""Forest-fire percolation simulation with repeated trials and logistic regression."""
import argparse
import csv
import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


class Forest:
    """Random forest grid: 1 is a tree, 0 is empty."""

    def __init__(self, height, width, density, rng=None):
        if height < 1 or width < 1:
            raise ValueError("Height and width must be positive.")
        if not 0 <= density <= 1:
            raise ValueError("Density must be between 0 and 1.")
        self.height, self.width, self.density = height, width, density
        rng = rng or random
        self.grid = [
            [int(rng.random() < density) for _ in range(width)]
            for _ in range(height)
        ]

    @classmethod
    def from_grid(cls, grid):
        if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("Grid must be non-empty and rectangular.")
        obj = cls.__new__(cls)
        obj.height, obj.width = len(grid), len(grid[0])
        obj.grid = [row[:] for row in grid]
        obj.density = sum(map(sum, grid)) / (obj.height * obj.width)
        return obj

    def _neighbors(self, row, col):
        for r, c in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
            if 0 <= r < self.height and 0 <= c < self.width:
                yield r, c

    def depth_first_search(self):
        stack = [(0, c) for c in range(self.width) if self.grid[0][c] == 1]
        seen = set(stack)
        while stack:
            row, col = stack.pop()
            if row == self.height - 1:
                return True
            for r, c in self._neighbors(row, col):
                if self.grid[r][c] == 1 and (r, c) not in seen:
                    seen.add((r, c))
                    stack.append((r, c))
        return False

    def breadth_first_search(self):
        queue = deque((0, c) for c in range(self.width) if self.grid[0][c] == 1)
        seen = set(queue)
        while queue:
            row, col = queue.popleft()
            if row == self.height - 1:
                return True
            for r, c in self._neighbors(row, col):
                if self.grid[r][c] == 1 and (r, c) not in seen:
                    seen.add((r, c))
                    queue.append((r, c))
        return False


def make_densities(step):
    if not 0 < step <= 1:
        raise ValueError("Step must be greater than 0 and at most 1.")
    count = round(1 / step)
    if not np.isclose(count * step, 1):
        raise ValueError("Step must divide evenly into 1.0, e.g. 0.01 or 0.02.")
    return np.linspace(0, 1, count + 1)


def run_simulation(trials=1000, height=20, width=20, step=0.01, seed=42):
    """Run independent trials at every density and return aggregated results."""
    if trials < 1:
        raise ValueError("Trials must be at least 1.")
    rng = random.Random(seed)
    records, trial_rows = [], []
    for density in make_densities(step):
        dfs_count = bfs_count = 0
        for _ in range(trials):
            forest = Forest(height, width, float(density), rng)
            dfs = forest.depth_first_search()
            bfs = forest.breadth_first_search()
            if dfs != bfs:
                raise RuntimeError("DFS and BFS disagreed on the same forest.")
            dfs_count += int(dfs)
            bfs_count += int(bfs)
            trial_rows.append((float(density), int(dfs)))
        records.append({
            "density": float(density),
            "trials": trials,
            "dfs_successes": dfs_count,
            "bfs_successes": bfs_count,
            "dfs_probability": dfs_count / trials,
            "bfs_probability": bfs_count / trials,
        })
        print(f"density={density:.2f} DFS={dfs_count}/{trials} ({dfs_count/trials:.3f})")
    return records, trial_rows


def fit_model(trial_rows, seed=42):
    """Fit logistic regression to trial-level density/outcome observations."""
    X = np.array([row[0] for row in trial_rows]).reshape(-1, 1)
    y = np.array([row[1] for row in trial_rows])
    if len(np.unique(y)) != 2:
        raise ValueError("Only one outcome class occurred; increase trials or vary the setup.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "test_rows": len(y_test),
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "brier_score": brier_score_loss(y_test, prob),
        "intercept": float(model.intercept_[0]),
        "density_coefficient": float(model.coef_[0][0]),
    }
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1000, help="Trials per density (default 1000)")
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, trial_rows = run_simulation(
        args.trials, args.height, args.width, args.step, args.seed
    )
    model, metrics = fit_model(trial_rows, args.seed)

    csv_path = args.output_dir / "forest_fire_simulation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    x = np.linspace(0, 1, 500).reshape(-1, 1)
    y = model.predict_proba(x)[:, 1]
    plt.figure(figsize=(10, 6))
    plt.plot([r["density"] for r in records], [r["dfs_probability"] for r in records],
             "o", markersize=3, label="Observed simulation rate (DFS)")
    plt.plot([r["density"] for r in records], [r["bfs_probability"] for r in records],
             "--", label="Observed simulation rate (BFS)")
    plt.plot(x.ravel(), y, linewidth=2.5, label="Logistic regression estimate")
    plt.xlabel("Forest density")
    plt.ylabel("Probability fire reaches the bottom")
    plt.title("Forest Fire Percolation: Simulation vs. Logistic Regression")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plot_path = args.output_dir / "forest_fire_logistic_regression.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\nHeld-out test results (simulated data)")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Brier score: {metrics['brier_score']:.3f}")
    print(f"Intercept: {metrics['intercept']:.3f}")
    print(f"Density coefficient: {metrics['density_coefficient']:.3f}")
    print(f"Saved results: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print("These metrics evaluate predictions of simulated outcomes, not real-world fires.")


if __name__ == "__main__":
    main()
