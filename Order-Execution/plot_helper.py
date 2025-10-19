# ===== plot_helper.py ===== 
import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
from compute_helper import *
from matplotlib.colors import LogNorm
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.lines import Line2D
import seaborn as sns
import jax.numpy as jnp
import matplotlib



################################### Loss Epochs Plot ########################################

# default metric keys to extract
_DEF_KEYS = ["e_pde", "e_bc_zero", "e_term", "e_symm", "e_traj"]

def _norm_alpha_str(s):
    """Convert tag string tokens like 'p05' -> '0.5'"""
    return s.replace('p', '.')

def _to_float(s):
    """Safe float conversion."""
    try:
        return float(s)
    except Exception:
        return None

def _parse_tag(tag):
    """Classify history tag into (kind, value)"""
    if tag.startswith("2d"):
        return "2d", None
    if tag.startswith("3d_alpha_"):
        a = _norm_alpha_str(tag.split("3d_alpha_", 1)[1])
        return "alpha", _to_float(a)
    if tag.startswith("alpha_"):
        a = _norm_alpha_str(tag.split("alpha_", 1)[1])
        return "alpha", _to_float(a)
    if tag == "final":
        return "final", None
    return "other", None

def load_all_histories_for_plot(run_dir, prefer = "3d"):
    """Load and organize training histories from a run directory.

    Picks the preferred history for each alpha value (3d vs plain),
    and returns dict with keys: '2d', 'alpha_*', 'final', plus others.
    """
    run_dir = os.path.abspath(run_dir)
    picks, others = {}, {}
    picked_2d, picked_final = None, None

    for p in glob.glob(os.path.join(run_dir, "*_history.pkl")):
        base = os.path.basename(p)
        if base.startswith("._"):
            continue
        tag = base[:-len("_history.pkl")]
        kind, aval = _parse_tag(tag)

        try:
            with open(p, "rb") as f:
                hist = pickle.load(f)
        except Exception:
            continue

        length = len(hist)
        mtime = os.path.getmtime(p)

        if kind == "alpha" and aval is not None:
            pref_rank = (2 if tag.startswith("3d_alpha_") and prefer == "3d" else
                         2 if tag.startswith("alpha_")    and prefer == "plain" else 1)
            prev = picks.get(aval)
            cand = (tag, hist, pref_rank, length, mtime)
            if (prev is None or
                cand[2] > prev[2] or
                (cand[2] == prev[2] and cand[3] > prev[3]) or
                (cand[2] == prev[2] and cand[3] == prev[3] and cand[4] > prev[4])):
                picks[aval] = cand
        elif kind == "2d":
            picked_2d = hist
        elif kind == "final":
            picked_final = hist
        else:
            others[tag] = hist

    out = {}
    if picked_2d is not None:
        out["2d"] = picked_2d
    for a in sorted(picks):
        out[f"alpha_{a:g}"] = picks[a][1]
    for k, h in sorted(others.items()):
        out[k] = h
    if picked_final is not None:
        out["final"] = picked_final
    return out

def _ordered_stage_keys(H):
    """Order stages for flattening: 2d -> alphas -> others -> final"""
    if not isinstance(H, dict):
        return None
    out = []
    if "2d" in H: out.append("2d")
    alphas = []
    for k in H:
        if k.startswith("alpha_"):
            try:
                alphas.append((float(k.split("_", 1)[1]), k))
            except ValueError:
                pass
    out.extend([k for _, k in sorted(alphas)])
    for k in H:
        if k not in out and k not in ("2d", "final"):
            out.append(k)
    if "final" in H:
        out.append("final")
    return out

def _parse_hist_entry(h):
    """Parse a single history entry into (epoch, total, metrics)"""
    try:
        epoch = int(h[0]); total = float(h[1])
    except Exception:
        return None
    md = h[2] if len(h) > 2 and isinstance(h[2], dict) else {}
    return epoch, total, md

def flatten_history(H):
    """Concatenate staged histories into one flat structure for plotting"""
    keys = _ordered_stage_keys(H)
    if keys is None:  # single-stage list
        parsed = [p for p in (_parse_hist_entry(h) for h in H) if p is not None]
        epochs = [e for (e, _, _) in parsed]
        totals = [L for (_, L, _) in parsed]
        mets   = [m for (_, _, m) in parsed]
        return {"epochs": np.asarray(epochs, float),
                "total":  np.asarray(totals, float),
                "metrics": {n: np.asarray([m.get(n, np.nan) for m in mets]) for n in _DEF_KEYS},
                "boundaries": [epochs[-1]] if epochs else [0],
                "stage_order": ["single"]}
    epochs, totals, boundaries, stage_order = [], [], [0], []
    mets = {n: [] for n in _DEF_KEYS}
    offset = 0
    for k in keys:
        parsed = [p for p in (_parse_hist_entry(h) for h in H[k]) if p is not None]
        if not parsed: continue
        stage_order.append(k)
        e_loc = [e for (e, _, _) in parsed]
        L_loc = [L for (_, L, _) in parsed]
        M_loc = [m for (_, _, m) in parsed]
        epochs.extend([offset + e for e in e_loc])
        totals.extend(L_loc)
        for n in mets: mets[n].extend([m.get(n, np.nan) for m in M_loc])
        offset += e_loc[-1] + 1
        boundaries.append(offset)
    return {"epochs": np.asarray(epochs, float),
            "total":  np.asarray(totals, float),
            "metrics": {k: np.asarray(v, float) for k, v in mets.items()},
            "boundaries": boundaries,
            "stage_order": stage_order}

def _log_safe(y, eps=1e-12):
    """Safe log transform, ignoring NaNs and clipping at eps"""
    y = np.asarray(y, float)
    return np.where(np.isnan(y), np.nan, np.log(np.maximum(y, eps)))

def plot_flat_history(flat):
    """Plot flattened history with stage boundaries"""
    E, T, M = flat["epochs"], flat["total"], flat["metrics"]
    plt.figure(figsize=(12, 8))
    plt.plot(E, _log_safe(T), label="Total", color="black", linewidth=2)
    for name in _DEF_KEYS:
        y = M.get(name, None)
        if y is not None and y.size:
            plt.plot(E, _log_safe(y), label=name, alpha=0.85)
    if len(flat["stage_order"]) > 1:
        ymin, ymax = plt.ylim()
        for b in flat["boundaries"][:-1]:
            plt.axvline(b, ls="--", alpha=0.25, color="tab:blue")
        mids = [(flat["boundaries"][i] + flat["boundaries"][i+1]) / 2
                for i in range(len(flat["boundaries"]) - 1)]
        ytext = ymin + 0.06 * (ymax - ymin)
        for xmid, name in zip(mids, flat["stage_order"]):
            plt.text(xmid, ytext, name, ha="center", va="bottom", fontsize=9, alpha=0.8)
    plt.xlabel("Global epoch (concatenated)")
    plt.ylabel("log loss")
    plt.title("Training loss across entire curriculum")
    plt.legend()
    plt.grid(True)
    plt.show()



################################### Value function plots ########################################

def plot_pinn_vs_exact(model, T=1, S=50, X0=10, x_range=(-10, 10), lam=0.01, kappa=0.05, sigma=0.04, d3=True, label=None):
    """Plot PINN model predictions vs exact value function in original and asinh space

    Args:
        model (jaxlib._jax.PjitFunction): PINN model
        T (int): time horizon
        S (int): stock price
        X0 (int): initial inventory
        x_range (tuple): range of inventories 
        lam (float): risk-aversion
        kappa (float): inventory-hold weighting
        sigma (float): volatility
        d3 (bool): if lambda > 0. Defaults to True.
        label (str): label for the PINN model
    """

    # grid (avoid exact t=0 and t=T for stability)
    X_vals = np.linspace(x_range[0], x_range[1], 50)
    t_vals = np.linspace(0.01, T - 0.05, 50)
    X_grid, t_grid = np.meshgrid(X_vals, t_vals)

    # PINN inputs
    X_flat = X_grid.ravel()
    t_flat = t_grid.ravel()
    S_flat = np.full_like(X_flat, S)
    tau_flat = T - t_flat
    x_lims = [x_range[0], 0.5*x_range[0], 0, 0.5*x_range[1], x_range[1]]
    if d3:
        inputs = jnp.stack([tau_flat, X_flat, S_flat], axis=1)
    else:
        inputs = jnp.stack([tau_flat, X_flat], axis=1)

    # PINN prediction (original space)
    value_pred = np.asarray(model(inputs)).squeeze().reshape(t_grid.shape)

    # exact value (original space) 
    value_exact = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            tau = T - t_grid[i, j]
            value_exact[i, j] = compute_value_function(tau, X_grid[i, j], S, lam, kappa, sigma)

    # mask invalid region for stability
    mask = t_grid <= T - 0.01

    Xg = np.where(mask, X_grid, np.nan)
    tg = np.where(mask, t_grid, np.nan)
    Vp = np.where(mask, value_pred, np.nan)
    Ve = np.where(mask, value_exact, np.nan)

    # plots: PINN, Exact, |Error| 
    fig = plt.figure(figsize=(12,12), constrained_layout=True)

    # 1) PINN
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(Xg, tg, Vp, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
    ax1.set_xlabel('Inventory X'); ax1.set_ylabel('Time t'); ax1.set_zlabel('Value Function Γ')
    ax1.set_title(f'{label} Prediction (Original Space)')
    ax1.view_init(elev=20, azim=45)
    ax1.set_box_aspect(None, zoom=0.85)
    ax1.set_xticks(x_lims)


    # 2) Exact
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(Xg, tg, Ve, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
    ax2.set_xlabel('Inventory X'); ax2.set_ylabel('Time t'); ax2.set_zlabel('Value Function Γ')
    ax2.set_title('Exact Solution (Original Space)')
    ax2.view_init(elev=20, azim=45)
    ax2.set_box_aspect(None, zoom=0.85)
    ax2.set_xticks(x_lims)


    # 3) Absolute error (original space)
    diff = Vp - Ve
    abs_err = np.abs(diff)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(Xg, tg, abs_err, cmap='hot', alpha=0.9, linewidth=0, antialiased=True)
    ax3.set_xlabel('Inventory X'); ax3.set_ylabel('Time t'); ax3.set_zlabel('|Error|')
    ax3.set_title('Absolute Error (Original Space)')
    ax3.view_init(elev=20, azim=45)
    ax3.set_box_aspect(None, zoom=0.85)
    ax3.set_xticks(x_lims)

    plt.show()

    # stats (exclude NaNs) for original space
    valid = np.isfinite(diff) & np.isfinite(Ve)
    rel = np.abs(diff[valid]) / (np.abs(Ve[valid]) + 1e-6)

    L2 = np.sqrt(np.nanmean(diff[valid]**2))              
    Linf = np.nanmax(np.abs(diff[valid]))                 
    
    print("Original Space Error Statistics:")
    print(f"  Mean Absolute Error: {np.nanmean(np.abs(diff[valid])):.6f}")
    print(f"  Max Absolute Error: {np.nanmax(np.abs(diff[valid])):.6f}")
    print(f"  Mean Relative Error: {np.nanmean(rel):.6f}")
    print(f"  Max Relative Error: {np.nanmax(rel):.6f}")
    print(f"  L2 norm (RMSE): {L2:.6f}")

    # ARCSINH SPACE
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)

    Vp_s = np.arcsinh(Vp)
    Ve_s = np.arcsinh(Ve)
    
    # 1) PINN (arcsinh)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(Xg, tg, Vp_s, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
    ax1.set_xlabel('Inventory X'); ax1.set_ylabel('Time t'); ax1.set_zlabel('arcsinh Γ')
    ax1.set_title(f'{label} Prediction (arcsinh space)')
    ax1.view_init(elev=20, azim=45)
    ax1.set_box_aspect(None, zoom=0.85)
    ax1.set_xticks(x_lims)

    # 2) Exact (arcsinh)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(Xg, tg, Ve_s, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
    ax2.set_xlabel('Inventory X'); ax2.set_ylabel('Time t'); ax2.set_zlabel('arcsinh Γ')
    ax2.set_title('Exact Solution (arcsinh space)')
    ax2.view_init(elev=20, azim=45)
    ax2.set_box_aspect(None, zoom=0.85)
    ax2.set_xticks(x_lims)

    # 3) Absolute error (arcsinh)
    diff_s = Vp_s - Ve_s
    abs_err_s = np.abs(diff_s)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(Xg, tg, abs_err_s, cmap='hot', alpha=0.9, linewidth=0, antialiased=True)
    ax3.set_xlabel('Inventory X'); ax3.set_ylabel('Time t'); ax3.set_zlabel('|arcsinh Error|')
    ax3.set_title('Absolute Error (arcsinh space)')
    ax3.view_init(elev=20, azim=45)
    ax3.set_box_aspect(None, zoom=0.85)
    ax3.set_xticks(x_lims)

    plt.show()

    # stats (exclude NaNs) for arcsinh space ---
    valid_s = np.isfinite(diff_s) & np.isfinite(Ve_s)
    rel_s = np.abs(diff_s[valid_s]) / (np.abs(Ve_s[valid_s]) + 1e-6)
    L2_s = np.sqrt(np.nanmean(diff_s[valid_s]**2))        

    print("Arcsinh-Space Error Statistics:")
    print(f"  Mean Arcsinh Absolute Error: {np.nanmean(np.abs(diff_s[valid_s])):.6f}")
    print(f"  Max Arcsinh Absolute Error: {np.nanmax(np.abs(diff_s[valid_s])):.6f}")
    print(f"  Mean Arcsinh Relative Error: {np.nanmean(rel_s):.6f}")
    print(f"  Max Arcsinh Relative Error: {np.nanmax(rel_s):.6f}")
    print(f"  L2 norm (RMSE, arcsinh): {L2_s:.6f}")


def plot_error_heatmaps_multi(models, T=1.0, S=50.0, x_range=(-10, 10), lam=0.01, kappa=0.05, sigma=0.04, d3=True, grid=200, clip_pct=99):
    """ For each model: plot heatmap of asinh of abolusted difference of predicted vs exact value function over (t, X). All panels share a single color scale (cmap=viridis by default).

    Args:
        models (dict): dictionary of all models to compare
        T (int): time horizon
        S (int): stock price
        x_range (tuple): range of inventories 
        lam (float): risk-aversion
        kappa (float): inventory-hold weighting
        sigma (float): volatility
        d3 (bool): if lambda > 0. Defaults to True.
        grid (int): number of grid points per axis. Defaults to 200.
        clip_pct (int): percentile for clipping. Defaults to 99.
    """
    
    # grid (avoid exact t=0 and t=T for stability)
    X_vals = np.linspace(x_range[0], x_range[1], int(grid))
    t_vals = np.linspace(0.01, T - 0.05, int(grid))     
    Xg, tg = np.meshgrid(X_vals, t_vals)
    Xf, tf = Xg.ravel(), tg.ravel()
    tau = T - tf

    Z = jnp.stack([tau, Xf, np.full_like(Xf, S)], axis=1) if d3 else jnp.stack([tau, Xf], axis=1)

    # exact on same grid
    V_exact = np.empty_like(Xg)
    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            V_exact[i, j] = compute_value_function(T - tg[i, j], Xg[i, j], S, lam, kappa, sigma)

    # mask invalid region for stability
    mask = tg <= T - 0.01
    V_exact_m = np.where(mask, V_exact, np.nan)

    # evaluate models and collect asinh(|error|) 
    asinh_errs = {}
    for name, model in models.items():
        V_pred = np.asarray(model(Z)).squeeze().reshape(tg.shape)
        V_pred_m = np.where(mask, V_pred, np.nan)
        err = np.abs(V_pred_m - V_exact_m)
        asinh_errs[name] = np.arcsinh(err)

    # shared vmax across all models
    all_vals = np.concatenate([A[np.isfinite(A)] for A in asinh_errs.values() if np.isfinite(A).any()]) \
               if any(np.isfinite(A).any() for A in asinh_errs.values()) else np.array([1.0])
    vmax_asinh = float(np.nanpercentile(all_vals, clip_pct))
    vmin_asinh = 0.0

    # plotting: 1 row × N columns
    names = list(models.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.6*n + 1.8, 4.6), constrained_layout=True, squeeze=False)
    im_kwargs = dict(
        origin="lower", aspect="auto", interpolation="nearest",
        extent=[X_vals.min(), X_vals.max(), t_vals.min(), t_vals.max()],
    )

    ims = []
    for col, name in enumerate(names):
        ax = axes[0, col]
        im = ax.imshow(asinh_errs[name], vmin=vmin_asinh, vmax=vmax_asinh,
                       cmap="inferno", **im_kwargs)
        ax.set_title(f"{name}")
        ax.set_xlabel("Inventory X")
        if col == 0:
            ax.set_ylabel("Time t")
        ims.append(im)

    # one shared colorbar on the right
    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label(r"$\text{asinh}(|\Gamma_{\text{pred}} - \Gamma_{\text{exact}}|)$")

    plt.show()

    # stats
    print("=== asinh(|error|) stats (shared scaling) ===")
    for name in names:
        A = asinh_errs[name]
        Af = A[np.isfinite(A)]
        print(f"{name:>20}: mean={np.nanmean(Af):.6f}  max={np.nanmax(Af):.6f}")
    
        
        
################################### Trajectory Plots ########################################

def inventory_trajectory_multi(models, paths, dt, T=1.0, X0=10, lam=0.01, kappa=0.05, d3=True):
    """Generate and plot inventory trajectories of multiple models against exact

    Args:
        models (dict): dictionary of all models to compare
        paths (np.ndarray): stock price paths
        dt (float): time step
        T (float): time horizon
        X0 (float): initial inventory
        lam (float): risk-aversion
        kappa (float): inventory-hold weighting
        d3 (bool): if lambda > 0. Defaults to True
    """
    # compute exact trajectories
    x_true = compute_x_star(paths, T=T, X=X0, lam=lam, kappa=kappa, dt=dt)
    n_paths, n_true = x_true.shape
    t_true = np.linspace(0, T, n_true)

    # colors per path (consistent across columns)
    path_colors = sns.color_palette("husl", n_paths)

    # evaluate each model and align to truth grid
    def _interp_to_grid(arr, t_src, t_tgt):
        out = np.full((arr.shape[0], t_tgt.size), np.nan, dtype=float)
        for i in range(arr.shape[0]):
            y = arr[i].astype(float)
            m = np.isfinite(y)
            if m.sum() >= 2:
                out[i] = np.interp(t_tgt, t_src[m], y[m])
        return out

    results = {}
    global_ymin, global_ymax = np.inf, -np.inf
    global_bar_min, global_bar_max = np.inf, -np.inf

    # stash aligned trajectories + finals
    aligned = {}   # name -> (x_pred_aligned, x_final_pred)
    for name, model in models.items():
        x_pred, _, _ = compute_pinn_trajectories(model, paths, T=T, X0=X0, dt=dt, d3=d3)
        t_pred = np.linspace(0, T, x_pred.shape[1])
        x_al = _interp_to_grid(x_pred, t_pred, t_true)
        aligned[name] = (x_al, x_al[:, -1])

        # update global limits (top row)
        global_ymin = min(global_ymin, np.nanmin(x_true), np.nanmin(x_al))
        global_ymax = max(global_ymax, np.nanmax(x_true), np.nanmax(x_al))
        # bottom row (finals)
        global_bar_min = min(global_bar_min, np.nanmin(x_true[:, -1]), np.nanmin(x_al[:, -1]))
        global_bar_max = max(global_bar_max, np.nanmax(x_true[:, -1]), np.nanmax(x_al[:, -1]))

        # metrics
        mask = np.isfinite(x_true) & np.isfinite(x_al)
        rmse = np.sqrt(np.nanmean((x_al[mask] - x_true[mask])**2))
        results[name] = {
            "rmse": float(rmse),
            "final_mean": float(np.nanmean(x_al[:, -1])),
            "final_std":  float(np.nanstd(x_al[:, -1])),
            "final_max_abs": float(np.nanmax(np.abs(x_al[:, -1]))),
        }

    # add a small margin to shared y-lims
    def _pad(a, b, frac=0.05):
        span = max(b - a, 1e-9)
        return a - frac*span, b + frac*span

    ylo, yhi = _pad(global_ymin, global_ymax, 0.06)
    blo, bhi = _pad(global_bar_min, global_bar_max, 0.15)

    # figure: 2 rows × N models columns
    names = list(models.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(6.0*n, 6.8), constrained_layout=True)
    if n == 1:  
        axes = np.array(axes).reshape(2, 1)

    for col, name in enumerate(names):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        x_al, x_final = aligned[name]

        # top: trajectories per path (solid) + truth (dashed)
        for i in range(n_paths):
            ax_top.plot(t_true, x_al[i], color=path_colors[i], lw=2, alpha=0.95)
            ax_top.plot(t_true, x_true[i], color=path_colors[i], lw=2, ls="--", alpha=0.9)
        ax_top.set_ylim(ylo, yhi)
        ax_top.set_xlabel("Time")
        if col == 0:
            ax_top.set_ylabel("Inventory")
        ax_top.set_title(f"{name}: trajectories")
        ax_top.grid(True, ls=":", alpha=0.5)
        
        if col == 0:
            proxy_pred = plt.Line2D([0],[0], color="k", lw=2, label="Model")
            proxy_true = plt.Line2D([0],[0], color="k", lw=2, ls="--", label="True")
            ax_top.legend(handles=[proxy_pred, proxy_true], loc="best")

        # bottom: final inventories, grouped (model vs true) per path
        idx = np.arange(n_paths)
        width = 0.38
        ax_bot.bar(idx - width/2, x_true[:, -1], width, label="True",
                   alpha=0.6, color="lightgray", edgecolor="dimgray")
        ax_bot.bar(idx + width/2, x_final, width, label=name,
                   alpha=0.9, color=[path_colors[i] for i in idx], edgecolor="black", linewidth=0.6)

        # annotate model bars
        for j, val in enumerate(x_final):
            ax_bot.text(j + width/2, val, f"{val:.3f}", ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=8)

        ax_bot.axhline(0, color="red", ls="--", lw=1.2, alpha=0.9)
        ax_bot.set_ylim(blo, bhi)
        ax_bot.set_xlabel("Path")
        if col == 0:
            ax_bot.set_ylabel("Final inventory")
        ax_bot.set_xticks(idx)
        ax_bot.set_xticklabels([f"P{i+1}" for i in idx])
        ax_bot.grid(True, axis="y", ls="--", alpha=0.35)
        ax_bot.set_axisbelow(True)

        # background
        for ax in (ax_top, ax_bot):
            ax.set_facecolor("#f9f9f9")


    fig.suptitle(r"Inventory trajectories $x^*(t)$ and final inventories", y=1.1, fontsize=20)
    fig.patch.set_facecolor("white")
    plt.show()

    # print summary
    print("=== Inventory comparison ===")
    for name in names:
        r = results[name]
        print(f"{name:>18} | RMSE={r['rmse']:.6f}  "
              f"final mean={r['final_mean']:.6f}  std={r['final_std']:.6f}  "
              f"max|final|={r['final_max_abs']:.6f}")





def inventory_trajectory_hist_multi(models, Ndt, Npaths, epsilon, S0=50, sigma=0.04, T=1.0, X0=10, lam=0.01, kappa=0.05, seed=42, d3=True, bins=30, clip_pct=99.5):
    """Plots histograms of absolute final inventory |X_T| for multiple models

    Args:
        models (dict): dictionary of all models to compare
        Ndt (int): number of time steps
        Npaths (int): number of paths
        epsilon (float): target final inventory threshold
        S0 (int): initial stock price
        sigma (float): volatility
        T (float): time horizon
        X0 (int): initial inventory
        lam (float): risk-aversion
        kappa (float): inventory-hold weighting
        seed (int): seed for price simulation
        d3 (bool): if lambda > 0. Defaults to True.
        bins (int): number of histogram bins
        clip_pct (float): percentile for clipping x-axis range
    """
    
    # 1) simulate price paths
    paths, dt = simulate_S(S0, sigma, T, Ndt, Npaths, seed)

    # 2) compute final inventories per model
    abs_final = {}
    for name, model in models.items():
        x_star, _, _ = compute_pinn_trajectories(model, paths, T=T, X0=X0, dt=dt, d3=d3)
        x_final = x_star[:, -1]

        abs_final[name] = np.abs(np.asarray(x_final, float))

    # 3) shared bin edges and range 
    cat = np.concatenate(list(abs_final.values())) if abs_final else np.array([0.0])
    xmax = float(np.nanpercentile(cat, clip_pct))
    xmax = max(xmax, float(epsilon)) * 1.02
    edges = np.linspace(0.0, xmax, int(bins) + 1)

    pal = sns.color_palette("husl", len(abs_final))
    colors = {k: pal[i] for i, k in enumerate(abs_final.keys())}

    # 4) plot
    plt.figure(figsize=(10, 6))
    for name, arr in abs_final.items():
        sns.histplot(
            arr, bins=edges, stat="density", element="step",
            fill=True, alpha=0.45, linewidth=1.2,
            color=colors.get(name, None), label=name
        )

    # epsilon line
    plt.axvline(x=epsilon, color="red", linestyle="--", linewidth=1.6, label=f"ε = {epsilon:g}")

    plt.title(r"Distribution of $|X_T|$ (absolute final inventory) by model")
    plt.xlabel(r"$|X_T|$  (absolute final inventory)")
    plt.ylabel("Density")
    plt.xlim(0.0, xmax)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()

    # 5) inset mean ± std for |X_T|
    ax_inset = plt.gca().inset_axes([0.63, 0.52, 0.33, 0.38])
    labels = list(abs_final.keys())
    means  = [np.nanmean(abs_final[k]) for k in labels]
    stds   = [np.nanstd(abs_final[k])  for k in labels]
    x_pos  = np.arange(len(labels))
    ax_inset.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, color='black')
    for i, k in enumerate(labels):
        ax_inset.plot([x_pos[i]], [means[i]], marker='o', ms=8, color=colors[k])
    ax_inset.axhline(0, color='black', linewidth=0.8)
    ax_inset.set_xticks(x_pos)
    ax_inset.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_inset.set_ylabel("Mean ± Std")
    ax_inset.set_title("Summary", fontsize=9)

    plt.tight_layout()
    plt.show()

    # 6) print stats
    print("=== |X_T| stats by model ===")
    for name, arr in abs_final.items():
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        p95 = float(np.nanpercentile(arr, 95.0))
        max_val = float(np.nanmax(arr)) 
        frac_below_eps = float(np.mean(arr <= epsilon))
        
                
        print(f"{name:>18}: mean={mean:.6f}  std={std:.6f}  "
              f"p95={p95:.6f}  max={max_val:.6f}"
              f"P(|X_T|≤ε)={frac_below_eps:.3f}")


def trading_rate_trajectory_multi(models, paths, T, X0, lam, kappa, dt, d3=True):
    """Trading rate trajectories for the PINN models vs closed-form solution.

    Args:
        models (dict): dictionary of models to compare
        paths (np.ndarray): stock price paths
        T (float): time horizon
        X0 (float): initial inventory
        lam (float): risk-aversion
        kappa (float): inventory-hold weighting
        dt (float): time steps
        d3 (bool): if lambda > 0. Defaults to True.

    """

    # closed-form 
    x_true = compute_x_star(paths, T, X0, lam, kappa, dt)
    v_true = compute_v_star(x_true, paths, T, lam, kappa, dt) 
    t_true = np.linspace(0.0, T, v_true.shape[1])

    # evaluate each model on its own grid
    preds = {}   # name -> dict(t_pred, v_pred) 
    for name, model in models.items():
        _, v_pinn, _ = compute_pinn_trajectories(model, paths, T, X0, dt, d3)
        t_pred = np.linspace(0.0, T, v_pinn.shape[1])
        preds[name] = {"t": t_pred, "v": v_pinn}

    # shared y-limits 
    def _finite_minmax(arr):
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0: return np.inf, -np.inf
        return float(arr.min()), float(arr.max())
    y_lo, y_hi = _finite_minmax(v_true)
    for name in models:
        lo, hi = _finite_minmax(preds[name]["v"])
        y_lo = min(y_lo, lo); y_hi = max(y_hi, hi)
    if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_lo == y_hi:
        y_lim = None
    else:
        pad = 0.05 * (y_hi - y_lo if y_hi > y_lo else 1.0)
        y_lim = (y_lo - pad, y_hi + pad)


    # figure layout 
    n_models = len(models)
    figsize_per_panel=(5.5, 4.5)
    fig_w = figsize_per_panel[0] * n_models
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n_models, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    colors = sns.color_palette("husl", len(paths))
    model_names = list(models.keys())

    for col, name in enumerate(model_names):
        ax = axes[col]
        t_pred = preds[name]["t"]
        v_pred = preds[name]["v"]

        # closed-form (dashed)
        for i in range(len(paths)):
            valid = np.isfinite(v_true[i])
            if valid.any():
                ax.plot(t_true[valid], v_true[i, valid],
                        color=colors[i], linestyle='--', linewidth=2, alpha=0.9)

        # model (solid)
        for i in range(len(paths)):
            vp = np.asarray(v_pred[i], float)
            valid = np.isfinite(vp)
            if valid.any():
                ax.plot(t_pred[valid], vp[valid],
                        color=colors[i], linestyle='-', linewidth=2, alpha=0.95)

        ax.set_title(name)
        ax.set_xlabel("Time t")
        if col == 0:
            ax.set_ylabel(r"Trading rate $v(t)$")
        ax.grid(True, alpha=0.3)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        # legend
        if col == 0:
            handles = [
                Line2D([0], [0], color='k', linestyle='--', lw=2, label='True'),
                Line2D([0], [0], color='k', linestyle='-',  lw=2, label='Model'),
            ]
            ax.legend(handles=handles, loc="upper left")
            
    fig.suptitle("Optimal trading rate v*(t): models vs closed-form", y=1.02, fontsize=20)
    plt.tight_layout()
    plt.show()


################################ Path Value function Plot #####################################
            
def plot_comparison_with_errors_multi(models, paths, T, X0, lam, kappa, sigma, dt, d3=True, figsize_per_row=(14, 4.0)):
    """Plots the value function along multiple paths for multiple models vs closed-form solution, plus error plots.

    Args:
        models (dict): dictionary of all models to compare
        paths (np.ndarray): stock price paths
        T (float): time horizon
        X0 (float): initial inventory
        lam (float): risk-aversion
        kappa (float): inventory-hold weighting
        sigma (float): volatility
        dt (float): time step
        d3 (bool): if lambda > 0. Defaults to True.
        figsize_per_row (tuple): figure size per row
    """
    
    # closed-form
    x_star = compute_x_star(paths, T, X0, lam, kappa, dt)
    v_star = compute_v_star(x_star, paths, T, lam, kappa, dt)

    Npaths, Ndt1 = paths.shape
    t_true = np.linspace(0.0, T, Ndt1)

    # closed-form value function along path
    value_paths = np.zeros((Npaths, Ndt1))
    for p in range(Npaths):
        for i in range(Ndt1):
            tau = T - t_true[i]
            value_paths[p, i] = compute_value_function(tau, x_star[p, i], paths[p, i], lam, kappa, sigma) if tau > 0 else 0.0

    # path colors (consistent across all rows)
    path_colors = cm.viridis(np.linspace(0, 1, Npaths))

    # pre-evaluate models to get interpolated outputs on the true grid
    cache = {}
    for name, model in models.items():
        x_pinn, v_pinn, value_pinn = compute_pinn_trajectories(model, paths, T, X0, dt, d3=d3)
        t_val = np.linspace(0.0, T, value_pinn.shape[1])
        t_x = np.linspace(0.0, T, x_pinn.shape[1])
        t_v = np.linspace(0.0, T, v_pinn.shape[1])

        def _interp_to_grid(arr, t_src, t_tgt):
            out = np.empty((arr.shape[0], t_tgt.size))
            for r in range(arr.shape[0]):
                y = np.asarray(arr[r], float)
                finite = np.isfinite(y)
                if finite.sum() >= 2:
                    out[r, :] = np.interp(t_tgt, t_src[finite], y[finite])
                else:
                    out[r, :] = np.nan
            return out

        value_hat = _interp_to_grid(value_pinn, t_val, t_true)
        x_hat = _interp_to_grid(x_pinn, t_x, t_true)
        v_hat = _interp_to_grid(v_pinn, t_v, t_true)

        cache[name] = {
            "raw_value": value_pinn, 
            "t_val": t_val,
            "x_on_true": x_hat,
            "v_on_true": v_hat,
            "value_on_true": value_hat,
        }

    # figure layout (rows = models, 3 columns)
    n_models = len(models)
    fig_w, fig_h_row = figsize_per_row
    fig = plt.figure(figsize=(fig_w, fig_h_row * n_models))
    gs = fig.add_gridspec(n_models, 3, wspace=0.25, hspace=0.22)

    # plot a 3D path panel with time on x, S on y, x on z, colored by values
    def _plot_3d_panel(ax, time_grid, S_paths, x_paths, value_for_color, norm, title):
        cmap = plt.get_cmap("plasma")
        for p in range(Npaths):
            cols = cmap(norm(value_for_color[p]))
            for i in range(time_grid.size - 1):
                ax.plot([time_grid[i], time_grid[i+1]],
                        [S_paths[p, i],  S_paths[p, i+1]],
                        [x_paths[p, i],  x_paths[p, i+1]],
                        color=cols[i], linewidth=2.6, alpha=0.9)
            # start/end markers
            ax.scatter(time_grid[0],  S_paths[p, 0],  x_paths[p, 0],
                       color='green', s=90, marker='o', edgecolor='black', linewidth=1.2)
            ax.scatter(time_grid[-1], S_paths[p, -1], x_paths[p, -1],
                       color='red',   s=90, marker='s', edgecolor='black', linewidth=1.2)
            # small in-path label
            mid = time_grid.size // 2
            ax.text(time_grid[mid], S_paths[p, mid], x_paths[p, mid] + 0.4*p,
                    f'Path {p+1}', fontsize=9, color='black', weight='bold')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Time t'); ax.set_ylabel('Stock Price S'); ax.set_zlabel('Inventory x')
        ax.view_init(elev=24, azim=45)
        ax.set_box_aspect(None, zoom=0.8)

    # ----- draw rows -----
    for r, (name, _) in enumerate(models.items()):
        axM = fig.add_subplot(gs[r, 0], projection='3d')  # Model (left)
        axA = fig.add_subplot(gs[r, 1], projection='3d')  # Analytical (middle)
        axE = fig.add_subplot(gs[r, 2])                  # Errors (right)

        # values on true grid for shared normalization 
        value_hat = cache[name]["value_on_true"]       
        val_model_flat = value_hat[np.isfinite(value_hat)]
        val_true_flat = value_paths[np.isfinite(value_paths)]
        if val_model_flat.size and val_true_flat.size:
            vmin = float(min(np.min(val_model_flat), np.min(val_true_flat)))
            vmax = float(max(np.max(val_model_flat), np.max(val_true_flat)))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
        norm_row = Normalize(vmin=vmin, vmax=vmax)

        # (left) Model 3D: use model predictions on the true grid for color
        _plot_3d_panel(axM, t_true, paths, cache[name]["x_on_true"],
                       value_hat, norm_row, title=f"{name} — Model")

        # (middle) Analytical 3D: same norm/cmap as model
        _plot_3d_panel(axA, t_true, paths, x_star,
                       value_paths, norm_row, title="Analytical")

        # one shared colorbar for the two 3D panels in this row
        sm = cm.ScalarMappable(cmap=plt.get_cmap("plasma"), norm=norm_row)
        sm.set_array([])
        
        # colorbar attached to both 3D axes
        cbar = fig.colorbar(sm, ax=[axM, axA], fraction=0.046, pad=0.04, shrink=0.9)
        cbar.set_label(r'Value function $\Gamma$', fontsize=12)

        # (right) Errors 2D — per path, three linestyles
        err_val = np.abs(value_hat - value_paths)
        err_x = np.abs(cache[name]["x_on_true"] - x_star)
        err_v = np.abs(cache[name]["v_on_true"] - v_star)

        for p in range(Npaths):
            m = np.isfinite(err_val[p])
        
            axE.plot(t_true[m], err_val[p, m], color=path_colors[p], lw=2.0, alpha=0.95,
                     label=r'$|\Gamma-\hat{\Gamma}|$' if p == 0 else None)
            axE.plot(t_true[m], err_x[p, m],   color=path_colors[p], lw=1.8, ls='--', alpha=0.9,
                     label=r'$|x-\hat{x}|$' if p == 0 else None)
            axE.plot(t_true[m], err_v[p, m],   color=path_colors[p], lw=1.8, ls=':',  alpha=0.9,
                     label=r'$|v-\hat{v}|$' if p == 0 else None)

        axE.set_title("Absolute errors vs time", fontsize=16)
        axE.set_xlabel("Time t"); axE.set_ylabel("Absolute Error")
        axE.grid(True, alpha=0.3)
        axE.legend(ncol=3, fontsize=11, loc='upper right')

    plt.show()
    
    
def safe_mean_std_over_paths(E):
    """finds the mean and std of value function over aggregated paths
    Args:
        E (np.ndarray): array of paths 
    Returns:
        mu (np.ndarray), std (np.ndarray): mean and std over paths
    """
    
    isfin = np.isfinite(E)
    cnt = isfin.sum(axis=0)
    # mean
    sumE = np.nansum(E, axis=0)
    mu = np.divide(sumE, cnt, out=np.full(E.shape[1], np.nan), where=cnt > 0)
    # population variance 
    diff = np.where(isfin, E - mu, 0.0)
    var = np.divide((diff**2).sum(axis=0), cnt, out=np.full(E.shape[1], np.nan), where=cnt > 0)
    sd = np.sqrt(var)
    return mu, sd

def plot_mt3d_plus_allmodels_error_split(models, paths, T, X0, lam, kappa, sigma, dt, mt_key="MTPINN-λ-curr", path_ids=(0, 1, 2), d3=True, cmap_name="viridis", figsize=(12.0, 3.4)):

    # sanity on path ids 
    path_ids = tuple(int(i) for i in path_ids)
    Npaths, Nt = paths.shape

    # grids and closed-form
    t = np.linspace(0.0, T, Nt)
    S = paths
    x_star = compute_x_star(S, T, X0, lam, kappa, dt)
    v_star = compute_v_star(x_star, S, T, lam, kappa, dt)

    val_true = np.zeros((Npaths, Nt))
    for p in range(Npaths):
        for i in range(Nt):
            tau = T - t[i]
            val_true[p, i] = 0.0 if tau <= 0 else compute_value_function(
                tau, x_star[p, i], S[p, i], lam, kappa, sigma
            )

    def _interp_to_grid(arr, t_src, t_tgt):
        out = np.empty((arr.shape[0], t_tgt.size))
        for r in range(arr.shape[0]):
            y = np.asarray(arr[r], float)
            finite = np.isfinite(y)
            out[r, :] = np.interp(t_tgt, t_src[finite], y[finite]) if finite.sum() >= 2 else np.nan
        return out

    # evaluate models
    by_model, mt = {}, None
    for name, model in models.items():
        x_hat, v_hat, val_hat = compute_pinn_trajectories(model, S, T, X0, dt, d3=d3)
        t_x = np.linspace(0.0, T, x_hat.shape[1])
        t_v = np.linspace(0.0, T, v_hat.shape[1])
        t_val = np.linspace(0.0, T, val_hat.shape[1])

        x_on_t = _interp_to_grid(x_hat, t_x, t)
        v_on_t = _interp_to_grid(v_hat, t_v, t)
        val_on_t = _interp_to_grid(val_hat, t_val, t)

        err_val = np.abs(val_on_t - val_true)
        err_x = np.abs(x_on_t - x_star)
        err_v = np.abs(v_on_t - v_star)

        by_model[name] = {"val": val_on_t, "x": x_on_t, "v": v_on_t,
                          "err_val": err_val, "err_x": err_x, "err_v": err_v}
        if name == mt_key:
            mt = {"val": val_on_t, "x": x_on_t}

    if mt is None:
        raise ValueError(f"mt_key '{mt_key}' not found in models.")

    # figure
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        width_ratios=[1.0, 1.0, 1.5],   
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.06, wspace=0.12     
    )

    # Left: 3D axes span all rows
    axM = fig.add_subplot(gs[:, 0], projection="3d")
    axA = fig.add_subplot(gs[:, 1], projection="3d")
    # Right: stacked error panels
    axVal = fig.add_subplot(gs[0, 2])
    axX = fig.add_subplot(gs[1, 2], sharex=axVal)
    axV = fig.add_subplot(gs[2, 2], sharex=axVal)

    # 3D panels (shared value function colormap) 
    vals_for_norm = np.concatenate([mt["val"][list(path_ids)].ravel(),
                                    val_true[list(path_ids)].ravel()])
    vmin = float(np.nanmin(vals_for_norm))
    vmax = float(np.nanmax(vals_for_norm))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    def _plot_3d(ax, Xz, color_vals, title):
        for j, p in enumerate(path_ids):
            cols = cmap(norm(color_vals[p]))
            for i in range(Nt - 1):
                ax.plot([t[i], t[i+1]],
                        [S[p, i], S[p, i+1]],
                        [Xz[p, i], Xz[p, i+1]],
                        color=cols[i], lw=2.4, alpha=0.95)
            ax.scatter(t[0],  S[p, 0],  Xz[p, 0],  color='green', s=80, marker='o', ec='k', lw=0.8)
            ax.scatter(t[-1], S[p, -1], Xz[p, -1], color='red',   s=80, marker='s', ec='k', lw=0.8)
            mid = Nt // 2
            ax.text(t[mid], S[p, mid], Xz[p, mid] + 0.4*j, f'Path {p+1}', fontsize=9, color='k', weight='bold')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"Time $t$"); ax.set_ylabel(r"Stock price $S$"); ax.set_zlabel(r"Inventory $x$")
        ax.view_init(elev=24, azim=45)
        ax.set_box_aspect(None, zoom=0.85)

    _plot_3d(axM, mt["x"], mt["val"], title=f"{mt_key} — Model")
    _plot_3d(axA, x_star,  val_true, title="Analytical")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=[axM, axA], fraction=0.046, pad=0.04, shrink=0.92)
    cbar.set_label(r"Value function $\Gamma$", fontsize=9)

    oi_palette = {
        "Vanilla PINN":   "#E69F00",  # orange
        "PINN-λ-curr":    "#56B4E9",  # sky blue
        "MTPINN-λ-curr":  "#009E73",  # bluish green
    }
    color_map = {name: oi_palette.get(name, f"C{i}") for i, name in enumerate(models.keys())}

    # helper to draw each stacked panel with mean ± std over paths
    def _draw_err(ax, key, ylabel, title, show_legend=False):
        for name, data in by_model.items():
            col = color_map[name]
            sl = list(path_ids)
            mu, sd = safe_mean_std_over_paths(data[key][sl])
            ax.plot(t, mu, lw=2.0, color=col, label=name)
            ax.fill_between(t, mu - sd, mu + sd, color=col, alpha=0.12)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.tick_params(labelleft=True)    
        if show_legend:
            ax.legend(fontsize=7, ncol=max(1, len(models)), frameon=False, loc="upper left")

    # draw stacked error panels
    _draw_err(axVal, "err_val", r"$|\Gamma-\hat{\Gamma}|$", "Value function error", show_legend=True)
    _draw_err(axX, "err_x", r"$|x-\hat x|$", "Inventory error")
    _draw_err(axV, "err_v", r"$|v-\hat v|$", "Trading-rate error")
    axV.set_xlabel("Time $t$")

    # only hide x tick labels on the upper two panels; keep y values everywhere
    axVal.tick_params(labelbottom=False)
    axX.tick_params(labelbottom=False)
