# Multi-Trajectory PINNs for Zero-Terminal HJB Problems

This repository hosts the reference implementation that accompanies the
Multi-Trajectory Physics-Informed Neural Network (MT-PINN) approach for solving
zero-terminal Hamilton–Jacobi–Bellman (HJB) problems. Two benchmark control
problems are provided:

* **High-Dimensional Linear–Quadratic Regulator (LQR):** a 20-dimensional
  continuous-time control task with known Riccati solutions.
* **Optimal Order Execution:** a stochastic control problem based on the
  Gatheral–Schied framework for liquidating an inventory under market impact.

Both case studies include closed-form baselines for quantitative evaluation.
The code is written in Python with [JAX](https://github.com/google/jax) and
[Flax](https://github.com/google/flax) and uses Orbax for checkpoint
management.

## Repository layout

```
.
├── High-Dimensional-LQR/
│   ├── Models/                     # Training notebooks for PINN and MT-PINN variants
│   ├── compute_helper.py           # Closed-form Riccati solutions and PINN rollouts
│   ├── loading_helper.py           # Flax model construction and checkpoint loading
│   ├── plot_helper.py              # Utilities for loss/value function visualisation
│   └── runs/                       # Saved configs, parameters, and histories
│   └── evaluation.ipynb            # Evaluation of model performances 

├── Order-Execution/
│   ├── Models/                     # Training notebooks for 2D/3D curricula
│   ├── compute_helper.py           # GBM path simulation & value/trading trajectories
│   ├── loading_helper.py           # Shared model definition + checkpoint loader
│   ├── plot_helper.py              # Loss curves, value plots, and diagnostics
│   ├── market_eval.py              # Tools for real market data evaluation windows
│   └── runs/                       # Saved experiments (synthetic + market models)
│   └── general_evaluation.ipynb    # General evaluation of model performances for optimal execution application for synthetic and real market backtest experiment
│   └── market_eval_demo.ipynb      # Real market backtest setup and specific results

└── requirements.txt                # Python dependencies
```

All helper modules are duplicated per experiment to keep experiment-specific
defaults isolated. Saved model artefacts under each `runs/` directory follow the
pattern `{tag}_config.json`, `{tag}_params/`, and `{tag}_history.pkl`, which can
be consumed with the `load_model_and_history` helper provided in each folder.

## Getting started

1. **Clone the repository and enter it:**
   ```bash
   git clone [(https://github.com/anthimevalin/Multi-Trajectory-PINNs-Zero-Terminal-HJB.git)]
   cd Multi-Trajectory-PINNs-Zero-Terminal-HJB
   ```
2. **Create an environment** (Python 3.10+ recommended) and install
   dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Open the evaluation notebooks** at the repository root to reproduce the
   reported figures:
   * `High-Dimensional-LQR/general_evaluation.ipynb` – synthetic benchmark for LQR application.
   * `Order-Execution/general_evaluation.ipynb` – synthetic and real-market
     analysis for optimal execution application.
   * `Order-Execution/market_eval_demo.ipynb` – SPY case study with cost,
     variance, and trading diagnostics.

> **Note:** The notebooks expect GPU-enabled JAX for retraining. Evaluation on
> saved checkpoints can be run on CPU-only machines.

## Synthetic benchmark evaluation

The `general_evaluation.ipynb` notebook under `Order-Execution/` reproduces the
synthetic benchmark comparisons for 2D and 3D settings. The pre-trained models
used in the paper are stored under the following directories:

```python
run_dir_PINN = "runs/PINN/run_2"
run_dir_PINN_lam = "runs/PINN_lam/run_2"
run_dir_MTPINN = "runs/MTPINN/run_2"
```

* `run_1`, `run_2`, and `run_3` correspond to λ = 0, 0.05, and 0.1,
  respectively, for the synthetic experiments.
* Tags map to model dimensionality and curriculum stage:
  * `2d_lam0`, `3d_lam0p05`, `3d_lam0p1` — PINN checkpoints for different λ.
  * `alpha_1p0` — curriculum and MT-PINN checkpoints.

Within the notebook, uncomment the following snippet to load the appropriate
models:

```python
# UNCOMMENT for Synthetic Benchmark Experiment
model_vanillaPINN, _, cfg, _ = loading_helper.load_model_and_history(
    run_dir_PINN,
    tag="3d_lam0p05",  # options: 2d_lam0, 3d_lam0p05, 3d_lam0p1
)
model_currPINN, _, _, _ = loading_helper.load_model_and_history(
    run_dir_PINN_lam,
    tag="alpha_1p0",  # options: 2d_lam0, alpha_1p0
)
model_currMTPINN, _, _, _ = loading_helper.load_model_and_history(
    run_dir_MTPINN,
    tag="alpha_1p0",  # options: 2d_lam0, alpha_1p0
)
```

The notebook walks through:

1. **Trajectory generation:** load synthetic price paths and evaluate the value
   functions from the selected models.
2. **Loss and value plots:** reproduce the figures comparing the PINN variants
   by using `plot_helper.plot_flat_history` and helper routines embedded in the
   notebook.
3. **Parameter sweeps:** toggle between λ settings by switching the `run_dir_*`
   paths or tags to visualise their effect on liquidation strategies.

## Real-market evaluation

The `general_evaluation.ipynb` notebook also provides end-to-end routines for
evaluating the SPY market checkpoints:

```python
run_dir_MTPINN_market = "runs/market/market_1"
```

Within the notebook you can:

1. **Select the market window:** load the saved SPY quote data and cost/variance
   targets corresponding to `market_1` (or subsequent numbered runs).
2. **Evaluate trading policies:** call
   `loading_helper.load_model_and_history(run_dir_MTPINN_market, tag="alpha_1p0")`
   to recover the MT-PINN policy and inspect execution quality.
3. **Plot realised trajectories:** use the provided plotting cells to visualise
   realised inventory, transaction costs, and variance profiles against market
   baselines.

For additional diagnostics, open `market_eval_demo.ipynb`, which illustrates how
to configure new SPY evaluation windows, customise inventory targets, and export
plots summarising execution performance.

## High-Dimensional LQR notebooks

The `High-Dimensional-LQR/general_evaluation.ipynb` notebook shows how to:

1. Load analytic Riccati controllers via `compute_helper`.
2. Restore MT-PINN checkpoints with `loading_helper.load_model_and_history`.
3. Compare value functions and control trajectories across curriculum stages
   using the built-in plotting utilities.

## Optimal Order Execution helpers

Inside the `Order-Execution/` directory (or the associated notebooks) you can
directly import the utilities needed for custom experiments:

1. **Simulate geometric Brownian motion paths:**
   ```python
   from compute_helper import simulate_S
   S, dt = simulate_S(S0=50.0, sigma=0.04, T=1.0, Ndt=200, Npaths=64)
   ```
2. **Load trained models (2D state or 3D state with price input):**
   ```python
   from loading_helper import load_model_and_history
   mt_model, _, cfg, _ = load_model_and_history("runs/MTPINN/run_2", "alpha_1p0")
   ```
3. **Generate MT-PINN trading trajectories:**
   ```python
   from compute_helper import compute_pinn_trajectories
   x_pinn, v_pinn, value_pinn = compute_pinn_trajectories(
       value_model=mt_model,
       paths=S,
       T=float(cfg.T),
       X0=float(cfg.x_range[1]),
       dt=dt,
       d3=True,
   )
   ```
4. **Compare with closed-form baselines:** leverage
   `compute_x_star`, `compute_v_star`, and `compute_value_function` from
   `compute_helper` for analytic references, and visualise the results via
   `plot_helper` or the `general_evaluation.ipynb` notebook.

### Market evaluation utilities

For real data analysis, the `market_eval.py` module provides helpers to:

* parse level-1 BBO files (CSV or Zstandard-compressed),
* construct regularised intraday windows, and
* evaluate trading performance in calendar time.

The `market_eval_demo.ipynb` notebook demonstrates how to pair these utilities
with MT-PINN checkpoints trained on historical windows.

## Retraining the models

Full retraining is best performed on GPU-enabled environments such as Google
Colab:

1. **Launch Colab** and clone the repository within a new notebook.
2. **Install dependencies** via `pip install -r requirements.txt`.
3. **Execute the training notebooks** under:
   * `High-Dimensional-LQR/Models/` — contains MT-PINN and baseline PINN training
     workflows for the LQR benchmark.
   * `Order-Execution/Models/` — hosts notebooks for the synthetic (2D/3D) and
     market curricula, including curriculum schedules and hyper-parameter grids.
4. **Save checkpoints** by exporting the `runs/` directory produced during
   training for later evaluation with the general notebooks described above.

Each notebook documents the model architecture (physics-informed value function
networks with time/state inputs), curriculum design, and logging configuration
needed to reproduce the published results.

## Acknowledgements

This code builds upon open-source JAX, Flax, Orbax, Optax, and supporting
scientific Python libraries. Please cite the MT-PINN paper if you use this
repository in academic work.
