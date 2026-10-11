# Forest Fire Percolation and Logistic Regression

A Python simulation exploring how forest density affects the probability that fire can travel from the top row to the bottom row through connected trees.

## Features
- Random forest grids where each cell independently contains a tree with probability equal to the density.
- Repeated Monte Carlo trials at each density (default: 1,000).
- DFS and BFS to determine whether a connected tree path reaches the bottom.
- Logistic regression using density as the predictor and fire-spread success (yes/no) as the target.
- Held-out evaluation with accuracy, ROC AUC, and Brier score.
- CSV results and a plot comparing observed simulation rates with the fitted logistic regression curve.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows: `.venv\\Scripts\\activate`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run
Default run: 1,000 trials per density, 20 x 20 grids, densities from 0.00 to 1.00 in increments of 0.01.

```bash
python forest_fire_percolation.py
```

Quick test run:

```bash
python forest_fire_percolation.py --trials 100 --step 0.02 --seed 42
```

Options: `--trials`, `--height`, `--width`, `--step`, `--seed`, and `--output-dir`. The step must divide evenly into 1.0.

## Output
- `results/forest_fire_simulation_results.csv`
- `results/forest_fire_logistic_regression.png`

## Interpretation and limitations
Logistic regression estimates the probability of fire spreading from density. It does not replace DFS/BFS and does not predict an exact fire path. The model is trained and evaluated on simulated outcomes, so its metrics describe this simulation—not real-world fire behavior. The simulation uses four-directional movement only and does not model wind, moisture, terrain, or tree species.


## Generalization to Unseen Density Levels

In addition to a random trial-level split, the program holds out complete density levels and evaluates logistic regression on those levels. This tests prediction at density values that were not used for training, while keeping the held-out levels within the overall simulated density range. Results still describe this simulated setup, not real-world wildfire behavior.
## Model Evaluation

The logistic regression model was evaluated on simulated forest outcomes using both a random trial-level split and a separate evaluation that held out entire density levels.

| Metric | Random Split | Unseen-Density Evaluation |
|---|---:|---:|
| Accuracy | 95.6% | 95.8% |
| ROC AUC | 0.994 | 0.994 |
| Brier Score | 0.031 | 0.030 |

The unseen-density evaluation held out 21 density levels, representing 21,000 simulated forests, while training used 80,000 simulated forests. The held-out density levels ranged from 0.00 to 1.00.

Both evaluations produced similar metrics, suggesting that the model performs consistently across the tested density levels in this simulated environment. These results do not establish predictive performance on real-world wildfires.

**Estimated 50% spread threshold:** 0.590 using both empirical interpolation and logistic regression, rounded to three decimal places.