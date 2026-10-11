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


def wilson_interval(successes, trials, z=1.96):
    """Return a 95% Wilson confidence interval for a binomial proportion."""
    if trials < 1:
        raise ValueError("Trials must be at least 1.")
    p_hat = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p_hat + z2 / (2 * trials)) / denominator
    margin = (
        z
        * ((p_hat * (1 - p_hat) / trials + z2 / (4 * trials * trials)) ** 0.5)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def estimate_empirical_threshold(records):
    """Linearly interpolate the density where observed spread probability crosses 0.5."""
    ordered = sorted(records, key=lambda record: record["density"])
    for left, right in zip(ordered, ordered[1:]):
        p_left = left["dfs_probability"]
        p_right = right["dfs_probability"]
        if p_left == 0.5:
            return left["density"]
        if (p_left < 0.5 <= p_right) or (p_left > 0.5 >= p_right):
            if p_right == p_left:
                return (left["density"] + right["density"]) / 2
            fraction = (0.5 - p_left) / (p_right - p_left)
            return left["density"] + fraction * (right["density"] - left["density"])
    # If no observed crossing occurs, return the nearest observed density as a fallback.
    return min(ordered, key=lambda record: abs(record["dfs_probability"] - 0.5))["density"]


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
            "dfs_ci_lower": wilson_interval(dfs_count, trials)[0],
            "dfs_ci_upper": wilson_interval(dfs_count, trials)[1],
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


def calculate_metrics(y_true, probabilities):
    """Calculate classification and probability metrics from predicted probabilities."""
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "test_rows": int(len(y_true)),
        "accuracy": accuracy_score(y_true, predictions),
        "brier_score": brier_score_loss(y_true, probabilities),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, probabilities)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def evaluate_unseen_densities(trial_rows, seed=42, holdout_every=5):
    """Train on most density levels and evaluate only on entirely held-out levels."""
    X = np.array([row[0] for row in trial_rows], dtype=float)
    y = np.array([row[1] for row in trial_rows], dtype=int)
    unique_densities = np.unique(X)
    if len(unique_densities) < holdout_every + 2:
        raise ValueError("Not enough density levels for unseen-density evaluation.")

    held_out_densities = set(unique_densities[::holdout_every].tolist())
    test_mask = np.array([density in held_out_densities for density in X])
    train_mask = ~test_mask

    if len(np.unique(y[train_mask])) < 2:
        raise ValueError("Training data has only one outcome class; cannot fit logistic regression.")
    if not np.any(test_mask):
        raise ValueError("The unseen-density test set is empty.")

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X[train_mask].reshape(-1, 1), y[train_mask])
    probabilities = model.predict_proba(X[test_mask].reshape(-1, 1))[:, 1]
    metrics = calculate_metrics(y[test_mask], probabilities)
    metrics["train_rows"] = int(train_mask.sum())
    metrics["held_out_density_levels"] = len(held_out_densities)
    metrics["held_out_density_min"] = float(min(held_out_densities))
    metrics["held_out_density_max"] = float(max(held_out_densities))
    return metrics


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
    unseen_metrics = evaluate_unseen_densities(trial_rows, args.seed)
    empirical_threshold = estimate_empirical_threshold(records)
    logistic_threshold = (
        -metrics["intercept"] / metrics["density_coefficient"]
        if metrics["density_coefficient"] != 0
        else float("nan")
    )

    csv_path = args.output_dir / "forest_fire_simulation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    densities = np.array([r["density"] for r in records])
    dfs_rates = np.array([r["dfs_probability"] for r in records])
    x = np.linspace(0, 1, 500).reshape(-1, 1)
    y = model.predict_proba(x)[:, 1]
    plt.figure(figsize=(10, 6))
    lower_errors = dfs_rates - np.array([r["dfs_ci_lower"] for r in records])
    upper_errors = np.array([r["dfs_ci_upper"] for r in records]) - dfs_rates
    plt.errorbar(
        densities, dfs_rates, yerr=np.vstack((lower_errors, upper_errors)),
        fmt="o", markersize=3, capsize=2, label="Observed rate (DFS, 95% Wilson CI)"
    )
    plt.plot([r["density"] for r in records], [r["bfs_probability"] for r in records],
             "--", linewidth=1.5, label="Observed simulation rate (BFS)")
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
    print(f"Empirical 50% spread threshold (interpolated): {empirical_threshold:.3f}")
    print(f"Logistic regression 50% threshold: {logistic_threshold:.3f}")
    print("\nUnseen-density evaluation (entire density levels held out)")
    print(f"Training rows: {unseen_metrics['train_rows']}")
    print(f"Test rows: {unseen_metrics['test_rows']}")
    print(f"Held-out density levels: {unseen_metrics['held_out_density_levels']}")
    print(
        "Held-out density range: "
        f"{unseen_metrics['held_out_density_min']:.2f} to "
        f"{unseen_metrics['held_out_density_max']:.2f}"
    )
    print(f"Accuracy: {unseen_metrics['accuracy']:.3f}")
    print(f"ROC AUC: {unseen_metrics['roc_auc']:.3f}")
    print(f"Brier score: {unseen_metrics['brier_score']:.3f}")
    print(f"Saved results: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print("These metrics evaluate predictions of simulated outcomes, not real-world fires.")


if __name__ == "__main__":
    main()
