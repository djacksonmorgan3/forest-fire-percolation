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
## Simulation Results

The graph compares observed fire-spread rates from repeated simulated forests with the probabilities estimated by logistic regression. The horizontal baseline represents the training-set spread prevalence, regardless of forest density.

![Forest fire percolation simulation comparing observed spread rates, logistic regression estimates, and the baseline probability](images/forest_fire_logistic_regression.png)

All results come from simulated forests, not real-world wildfire observations.

## Generalization to Unseen Density Levels

In addition to a random trial-level split, the program holds out complete density levels and evaluates logistic regression on those levels. This tests prediction at density values that were not used for training, while keeping the held-out levels within the overall simulated density range. Results still describe this simulated setup, not real-world wildfire behavior.


## Baseline Model Comparison

To evaluate whether logistic regression improves on a simple prediction strategy, I compared it with a baseline model on the same held-out simulated forests.

| Metric | Logistic Regression | Baseline |
|---|---:|---:|
| Accuracy | 0.958 | 0.586 |
| ROC AUC | 0.994 | 0.500 |
| Brier Score | 0.030 | 0.243 |

The logistic regression model performed better than the baseline across all three metrics on the held-out density levels. Its lower Brier score indicates more accurate probability estimates, while its higher ROC AUC indicates stronger discrimination between simulated spread and non-spread outcomes.

The evaluation used 80,000 training rows and 21,000 test rows across 21 held-out density levels. Results describe simulated forest behavior and should not be interpreted as real-world wildfire predictions.