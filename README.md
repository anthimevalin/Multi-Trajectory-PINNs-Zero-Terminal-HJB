# Multi-Trajectory PINNs for Zero-Terminal HJB Problems

This repository hosts the reference implementation that accompanies the
Multi-Trajectory Physics-Informed Neural Network (MT-PINN) approach for solving
zero-terminal Hamilton–Jacobi–Bellman (HJB) problems. Two benchmark control
problems are provided:

* **High-Dimensional Linear–Quadratic Regulator (LQR):** a 20-dimensional
  continuous-time control task with known Riccati solutions.
* **Optimal Order Execution:** a stochastic control problem based on the
  Gatheral–Schied framework for liquidating an inventory under market impact.

Both case studies compare classical single-trajectory PINNs against the
multi-trajectory curriculum and include closed-form baselines for quantitative
evaluation. The code is written in Python with
[JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax)
and uses Orbax for checkpoint management.

## Repository layout

```
.
├── High-Dimensional-LQR/
│   ├── Models/                     # Training notebooks for PINN and MT-PINN variants
│   ├── compute_helper.py           # Closed-form Riccati solutions and PINN rollouts
│   ├── loading_helper.py           # Flax model construction and checkpoint loading
│   ├── plot_helper.py              # Utilities for loss/value function visualisation
│   └── runs/                       # Saved configs, parameters, and histories
├── Order-Execution/
│   ├── Models/                     # Training notebooks for 2D/3D curricula
│   ├── compute_helper.py           # GBM path simulation & value/trading trajectories
│   ├── loading_helper.py           # Shared model definition + checkpoint loader
│   ├── plot_helper.py              # Loss curves, value plots, and diagnostics
│   ├── market_eval.py              # Tools for real market data evaluation windows
│   └── runs/                       # Saved experiments (synthetic + market models)
└── requirements.txt                # Python dependencies
```

All helper modules are duplicated per experiment to keep experiment-specific
defaults isolated. Saved model artefacts under each `runs/` directory follow the
pattern `{tag}_config.json`, `{tag}_params/`, and `{tag}_history.pkl`, which can
be consumed with the `load_model_and_history` helper provided in each folder.

## Getting started

1. **Create an environment** (Python 3.10+ recommended) and install
   dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Open the desired notebook** under `High-Dimensional-LQR/Models/` or
   `Order-Execution/Models/` to reproduce the training curricula, or run the
   evaluation notebooks at the directory root for post-training analysis.

> **Note:** The notebooks expect GPU-enabled JAX for the full training runs. You
> can still execute the evaluation scripts on CPU-only machines using the saved
> checkpoints.

## Working with saved checkpoints

Both experiments expose a consistent API for restoring a trained model and its
training history:

```python
from Order-Execution.loading_helper import load_model_and_history

model_fn, params, cfg, history = load_model_and_history(
    run_dir="Order-Execution/runs/MTPINN/run_2",
    tag="alpha_1p0",
)
```

The returned `model_fn` is a JIT-compiled callable that evaluates the learned
value function. The `cfg` object contains the curriculum hyper-parameters (time
horizon, state ranges, penalties, etc.), and `history` stores per-epoch loss
metrics used by the plotting utilities.

## High-Dimensional LQR workflow

1. **Generate system matrices:**
   ```python
   from High-Dimensional-LQR import compute_helper
   A, B, Q, R, S = compute_helper.construct_matrices(dim=20, shift_factor=0.1)
   ```
2. **Recover analytic controllers:**
   ```python
   P_of_t, K_of_t, *_ = compute_helper.closed_form_P_and_K(A, B, Q, R, S, T=1.0)
   ```
3. **Roll out MT-PINN trajectories for comparison:**
   ```python
   import jax.numpy as jnp
   from High-Dimensional-LQR.loading_helper import load_model_and_history
   model_fn, *_ = load_model_and_history("High-Dimensional-LQR/runs/20D/MT-PINN", "MT-PINN_LQR")
   traj = compute_helper.traj_PINN(
       model=model_fn,
       x0=jnp.ones(20),
       T=1.0,
       t_eval=None,
       A=A,
       B=B,
       R=R,
       S=S,
   )
   ```
4. **Visualise and benchmark:** use `High-Dimensional-LQR/plot_helper.py` or the
   `evaluation.ipynb` notebook to compare value functions, control gains, and
   loss curves against the analytic Riccati solution.

## Optimal Order Execution workflow

1. **Simulate geometric Brownian motion paths:**
   ```python
   from Order-Execution import compute_helper
   S, dt = compute_helper.simulate_S(S0=50.0, sigma=0.04, T=1.0, Ndt=200, Npaths=64)
   ```
2. **Load trained models (2D state or 3D state with price input):**
   ```python
   from Order-Execution.loading_helper import load_model_and_history
   mt_model, _, cfg, _ = load_model_and_history("Order-Execution/runs/MTPINN/run_2", "alpha_1p0")
   ```
3. **Generate MT-PINN trading trajectories:**
   ```python
   x_pinn, v_pinn, value_pinn = compute_helper.compute_pinn_trajectories(
       value_model=mt_model,
       paths=S,
       T=float(cfg.T),
       X0=float(cfg.x_range[1]),
       dt=dt,
       d3=True,
   )
   ```
4. **Compare with closed-form baselines:** leverage
   `compute_helper.compute_x_star`, `compute_helper.compute_v_star`, and
   `compute_helper.compute_value_function` for analytic references, and
   visualise the results via `Order-Execution/plot_helper.py` or the
   `general_evaluation.ipynb` notebook.

### Market evaluation utilities

For real SPY market data analysis, the `Order-Execution/market_eval.py` module provides
helpers to:

* parse level-1 BBO files (CSV or Zstandard-compressed),
* construct regularised intraday windows, and
* evaluate trading performance in calendar time.

The `market_eval_demo.ipynb` notebook demonstrates how to pair these utilities
with MT-PINN checkpoints trained on historical windows.

## Reproducing figures

The plotting helpers expose convenient entry points for the figures reported in
the associated work:

* `plot_helper.load_all_histories_for_plot` + `plot_helper.plot_flat_history`
  stitch together curriculum stages to show log-loss trajectories.
* `plot_helper.plot_pinn_vs_exact` overlays PINN predictions against
  closed-form value functions in both original and transformed spaces.

Refer to the provided notebooks for end-to-end scripts that regenerate the
paper plots.
