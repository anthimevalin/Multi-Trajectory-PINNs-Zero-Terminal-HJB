import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax import grad
import numpy as np
from numpy.linalg import solve
from scipy.linalg import solve_continuous_are, expm, norm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os, pickle
import matplotlib as mpl
from compute_helper import *


################################### Loss Epochs Plot ########################################
def _parse_hist_entry(h):
    """Parse a single history entry"""    
    try:
        epoch = int(h[0])
        total = float(h[1])
    except Exception:
        return None
    md = {}
    if len(h) >= 3 and isinstance(h[2], dict):
        md = h[2]
    return epoch, total, md

def flatten_single_history(hist, preferred_metric_keys=None):
    """Concatenate history into one flat structure for plotting"""
    parsed = [p for p in (_parse_hist_entry(h) for h in hist) if p is not None]
    if not parsed:
        return {"epochs": np.array([]), "total": np.array([]), "metrics": {}}

    epochs = np.array([e for (e,_,_) in parsed], dtype=float)
    totals = np.array([L for (_,L,_) in parsed], dtype=float)
    mets   = [m for (_,_,m) in parsed]

    all_keys = set().union(*[m.keys() for m in mets]) if mets else set()
    if preferred_metric_keys is None:
        canonical = ["e_pde", "e_term"]
        keys = [k for k in canonical if k in all_keys]
        for k in sorted(all_keys):
            if k not in keys and k.lower() != "total":
                keys.append(k)
    else:
        keys = [k for k in preferred_metric_keys if k in all_keys]

    metrics = {k: np.array([m.get(k, np.nan) for m in mets], dtype=float) for k in keys}
    return {"epochs": epochs, "total": totals, "metrics": metrics}

def _log_safe(y, eps=1e-12):
    """Safe log transform, ignoring NaNs and clipping at eps"""
    y = np.asarray(y, float)
    return np.where(np.isnan(y), np.nan, np.log(np.maximum(y, eps)))

def plot_flat_history(flat, log_scale=True, title="Training loss"):
    """Plot flattened history"""
    E, T, M = flat["epochs"], flat["total"], flat["metrics"]
    plt.figure(figsize=(8,4))
    yT = _log_safe(T) if log_scale else T
    plt.plot(E, yT, label="Total", color="black", linewidth=2)

    for name, vals in M.items():
        lbl = {"e_bc_zero":"e_ic"}.get(name, name)
        y = _log_safe(vals) if log_scale else vals
        plt.plot(E, y, label=lbl, alpha=0.9)

    plt.xlabel("Epoch")
    plt.ylabel("log loss" if log_scale else "loss")
    plt.title(title)
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

def load_history_from_dir(run_dir, tag = "PINN_LQR"):
    path = os.path.join(os.path.abspath(run_dir), f"{tag}_history.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_history_from_dir(run_dir, tag = "PINN_LQR",
                          preferred_metric_keys=None, log_scale=True, title=None):
    hist = load_history_from_dir(run_dir, tag)
    flat = flatten_single_history(hist, preferred_metric_keys=preferred_metric_keys)
    if title is None:
        title = f"{tag} — {os.path.basename(os.path.abspath(run_dir))}"
    plot_flat_history(flat, log_scale=log_scale, title=title)


################################### Evaluation plots ########################################

## Helper functions
def _rollout_value_model(model, x0, T, t_eval, A, B, R, S, Ndt=200):
    x, u, V = traj_PINN(model, np.asarray(x0, float), T, t_eval, A, B, R, S, Ndt=Ndt)
    return np.asarray(x), np.asarray(u), np.asarray(V)

def _l2_err(a, b):  # time-wise L2 across dims
    a = np.asarray(a); b = np.asarray(b)
    return np.linalg.norm(a - b, axis=1)

def _series_stats(err, t, ref=None):
    err = np.asarray(err, float); t = np.asarray(t, float)
    rms = np.sqrt(np.mean(err**2))
    auc = np.trapz(err, t)                     # L1 area under curve
    emax = float(err.max()); tmax = float(t[err.argmax()])
    final = float(err[-1])
    relL2 = None
    if ref is not None:
        ref = np.asarray(ref, float)
        num = np.trapz(err**2, t)
        den = np.trapz(ref**2, t)
        if den > 0: relL2 = float(np.sqrt(num/den))
    return {"max": emax, "t_at_max": tmax, "rms": rms, "auc": auc, "final": final, "relL2": relL2}

def comparsion_plot(models, x0, T, t_eval, A, B, Q, R, S, Ndt=200):
    """Comparison plot of models vs closed-form solution in state trajectories and norm errors of state and control.

    Args:
        models (dict): dictionary of models to compare
        x0 (jnp.array): initial state
        T (flaot): time horizon
        t_eval (np.array): time evaluation points
        A (jnp.array): A matrix
        B (jnp.array): B matrix
        Q (jnp.array): Q matrix
        R (jnp.array): R matrix
        S (jnp.array): S matrix
        Ndt (int): number of time steps. Defaults to 200.
    """
    
    plt.style.use("ggplot")
    mpl.rcParams.update({
        "axes.edgecolor": "0.45",
        "axes.linewidth": 0.8,
        "legend.frameon": True,
        "legend.facecolor": "#f5f5f5",
        "legend.edgecolor": "0.55",
        "legend.framealpha": 1.0,
    })

    # closed-form
    x_true, u_true, _ = traj_closed_form(A, B, Q, R, S, np.asarray(x0, float), t_eval, T)
    x_true = np.asarray(x_true); u_true = np.asarray(u_true)
    x_true_norm = np.linalg.norm(x_true, axis=1)
    u_true_norm = np.linalg.norm(u_true, axis=1)

    has_pinn = ("PINN" in models) and (models["PINN"] is not None)
    has_mtpinn = ("MTPINN" in models) and (models["MTPINN"] is not None)

    if has_pinn:
        x_pinn, u_pinn, _ = _rollout_value_model(models["PINN"], x0, T, t_eval, A, B, R, S, Ndt=Ndt)
    else:
        x_pinn = u_pinn = None

    if has_mtpinn:
        x_mtp, u_mtp, _ = _rollout_value_model(models["MTPINN"], x0, T, t_eval, A, B, R, S, Ndt=Ndt)
    else:
        x_mtp = u_mtp = None

    # errors
    err_x_pinn = _l2_err(x_pinn, x_true) if has_pinn else None
    err_u_pinn = _l2_err(u_pinn, u_true) if has_pinn else None
    err_x_mtp = _l2_err(x_mtp,  x_true) if has_mtpinn else None
    err_u_mtp = _l2_err(u_mtp,  u_true) if has_mtpinn else None

    # stats 
    stats = {}
    if has_pinn:
        stats["PINN"] = {
            "state": _series_stats(err_x_pinn, t_eval, ref=x_true_norm),
            "ctrl":  _series_stats(err_u_pinn, t_eval, ref=u_true_norm),
        }
    if has_mtpinn:
        stats["MT-PINN"] = {
            "state": _series_stats(err_x_mtp, t_eval, ref=x_true_norm),
            "ctrl":  _series_stats(err_u_mtp, t_eval, ref=u_true_norm),
        }

    # figure layout
    fig = plt.figure(figsize=(13, 6.8))
    ax_s_closed = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    ax_s_pinn = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax_s_mtp = plt.subplot2grid((2,6), (0,4), colspan=2)
    ax_err_x = plt.subplot2grid((2,6), (1,1), colspan=2)
    ax_err_u = plt.subplot2grid((2,6), (1,3), colspan=2)

    for ax in [ax_s_closed, ax_s_pinn, ax_s_mtp, ax_err_x, ax_err_u]:
        for side in ("bottom", "left", "top", "right"):
            sp = ax.spines[side]
            sp.set_visible(True)
            sp.set_color("0.45")
            sp.set_linewidth(0.8)
        ax.tick_params(colors="0.25")

    # plot helper
    def plot_all_states(ax, t, X, ttl, ylab=False):
        for i in range(X.shape[1]): ax.plot(t, X[:, i], alpha=0.9)
        ax.set_title(ttl); ax.grid(True, ls=":")
        if ylab: ax.set_ylabel("state x_i(t)")

    # state trajectories on top row
    plot_all_states(ax_s_closed, t_eval, x_true, "Closed-form states", ylab=True)
    if has_pinn:
        plot_all_states(ax_s_pinn, t_eval, x_pinn, "PINN states")
    else:
        ax_s_pinn.set_title("PINN states (N/A)"); ax_s_pinn.axis("off")
    if has_mtpinn:
        plot_all_states(ax_s_mtp, t_eval, x_mtp, "MT-PINN states")
    else:
        ax_s_mtp.set_title("MT-PINN states (N/A)"); ax_s_mtp.axis("off")

    # control and state errors on bottom row
    if has_pinn:
        ax_err_x.plot(t_eval, err_x_pinn, label="PINN")
    if has_mtpinn:
        ax_err_x.plot(t_eval, err_x_mtp, label="MT-PINN")
    ax_err_x.set_title(r"State error  $\|x(t) - x_{\text{closed}}(t)\|_2$")
    ax_err_x.set_ylabel("norm")
    ax_err_x.grid(True, ls=":")
    ax_err_x.legend(loc="best")

    if has_pinn:
        ax_err_u.plot(t_eval, err_u_pinn, label="PINN")
    if has_mtpinn:
        ax_err_u.plot(t_eval, err_u_mtp, label="MT-PINN")
    ax_err_u.set_title(r"Control error  $\|u(t) - u_{\text{closed}}(t)\|_2$")
    ax_err_u.set_xlabel("time t"); ax_err_u.set_ylabel("norm")
    ax_err_u.grid(True, ls=":")
    ax_err_u.legend(loc="best")

    
    fig.suptitle(t="State & Control Trajectory Comparison", y=0.995)
    plt.tight_layout()
    plt.show()

    # print stats
    for name, d in stats.items():
        print(f"[{name}] state  stats:", d["state"])
        print(f"[{name}] control stats:", d["ctrl"])


def inventory_comparison_plot(models, x0, T, t_eval, A, B, Q, R, S, Ndt=200, show_values=True):
    """ Compares the final inventories of the models.

    Args:
        models (dict): dictionary of models to compare
        x0 (jnp.array): initial state
        T (flaot): time horizon
        t_eval (np.array): time evaluation points
        A (jnp.array): A matrix
        B (jnp.array): B matrix
        Q (jnp.array): Q matrix
        R (jnp.array): R matrix
        S (jnp.array): S matrix
        Ndt (int): number of time steps. Defaults to 200.
        show_values (bool): shows the final inventory amount on bar plot. Defaults to True.
    """

    xT_dict = {}

    # models
    for name, model in (models or {}).items():
        if model is None: 
            continue
        x, _, _ = traj_PINN(model, np.asarray(x0, float), T, t_eval, A, B, R, S, Ndt=Ndt)
        xT_dict[str(name)] = np.asarray(x)[-1]

    dims = {v.shape[0] for v in xT_dict.values()}
    d = dims.pop()
    idx = np.arange(d)

    # order
    desired_order = [k for k in ["Closed", "PINN", "MTPINN"] if k in xT_dict] + \
                    [k for k in xT_dict.keys() if k not in {"Closed","PINN","MTPINN"}]
    names = desired_order

    base_colors = {
        "MTPINN": "#1f77b4",   # blue
        "PINN":   "#d62728",   # red
    }
    cmap = plt.cm.get_cmap("tab10", 10)

    def _color_for(k, i):
        return base_colors[k] if k in base_colors else cmap(i % cmap.N)

    # bar layout
    K = len(names)
    group_width = 1.86
    bar_w = min(group_width / K, 0.18)
    offsets = (np.arange(K) - (K - 1) / 2.0) * bar_w

    # figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # dynamic symmetric y-limits across all models
    all_vals = np.concatenate([xT_dict[n] for n in names])
    m = float(np.max(np.abs(all_vals))) if all_vals.size else 1.0
    ylim = 1.12 * m if m > 0 else 0.5
    ax.set_ylim(-ylim, ylim)

    # bars
    bars_by_model = {}
    for k, name in enumerate(names):
        vals = xT_dict[name]
        bars = ax.bar(idx + offsets[k], vals, width=bar_w,
                      color=_color_for(name, k),
                      edgecolor='0.35', linewidth=1.0, alpha=0.95,
                      label=name)
        bars_by_model[name] = bars

    # zero line
    ax.axhline(0.0, color='crimson', linestyle='--', linewidth=1.1, alpha=0.9)
    ax.set_xticks(idx)
    ax.set_xticklabels([f'$x_{{{i+1}}}$' for i in idx])
    ax.set_xlabel('state component', fontsize=12)
    ax.set_ylabel(r'$x_i(T)$', fontsize=12)
    ax.set_title(r"Final state values at $t=T$ (Closed, PINN, MT-PINN)", fontsize=14, pad=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    leg = ax.legend(ncols=min(K, 3), frameon=True)
    if leg is not None:
        leg.get_frame().set_alpha(1.0)

    # show values above bars
    if show_values:
        y_span = 1 * ylim
        min_gap = 0.055 * y_span
        base_off = 0.02 * y_span

        for i in range(d):
            entries = []
            for k, name in enumerate(names):
                val = float(xT_dict[name][i])
                x_pos = float(idx[i] + offsets[k])
                y_base = val + (base_off if val >= 0 else -base_off)
                entries.append([name, x_pos, y_base, val])

            entries.sort(key=lambda r: r[2])
            for j in range(1, len(entries)):
                if entries[j][2] - entries[j-1][2] < min_gap:
                    entries[j][2] = entries[j-1][2] + min_gap

            for name, x_pos, y_lab, val in entries:
                y_lab = max(-ylim * 0.98, min(ylim * 0.98, y_lab))
                ax.text(x_pos, y_lab, f'{val:.3f}',
                        ha='center',
                        va='bottom' if y_lab >= val else 'top',
                        fontsize=8, fontweight='bold',
                        zorder=3)

    plt.tight_layout()
    plt.show()

    # print stats
    for name in names:
        print(f"[{name}] mean|x(T)|={float(np.mean(np.abs(xT_dict[name]))):.4g}, std|x(T)|={float(np.std(np.abs(xT_dict[name]))):.4g}, "
              f"||x(T)||2={float(np.linalg.norm(xT_dict[name])):.4g}, max|x_i(T)|={float(np.max(np.abs(xT_dict[name]))):.4g} at i={int(np.argmax(np.abs(xT_dict[name])))+1}")


# helpers
def _eval_model_value(model, t, x, T, use_tau_input=True):
    """model takes z = [tau, x]; return scalar V(t,x)."""
    tau = (T - float(t)) if use_tau_input else float(t)
    z = np.concatenate([[tau], x.astype(np.float32)])[None, :]
    return float(np.asarray(model(jnp.array(z))))

def _V_true_from_P(P_of_t, t, x):
    P = np.asarray(P_of_t(float(t)))
    return float(x @ P @ x)

def _asinh_err(y, y_true):
    return np.abs(np.arcsinh(y) - np.arcsinh(y_true))

def _stats(y, y_true):
    e = _asinh_err(y, y_true)
    return dict(
        MAE=float(np.mean(e)),
        RMSE=float(np.sqrt(np.mean(e**2))),
        Max=float(np.max(e)),
    )

def value_comparison(models, x0, T, t_eval, A, B, Q, R, S, use_tau_input=True):
    """Comparison plot of the value functions along the optimal trajectory with error plot.

    Args:
        models (dict): dictionary of models to compare
        x0 (jnp.array): initial state
        T (flaot): time horizon
        t_eval (np.array): time evaluation points
        A (jnp.array): A matrix
        B (jnp.array): B matrix
        Q (jnp.array): Q matrix
        R (jnp.array): R matrix
        S (jnp.array): S matrix
        use_tau_input (bool): whether to use t or tau. Defaults to True.
    """
    
    x0 = np.asarray(x0, float).ravel()

    # closed-form
    x_true_traj, _, _ = traj_closed_form(A, B, Q, R, S, x0, t_eval, T)
    x_true_traj = np.asarray(x_true_traj)
    P_of_t, *_ = closed_form_P_and_K(A, B, Q, R, S, T=T)

    V_true = np.array([_V_true_from_P(P_of_t, t, x_true_traj[i])
                       for i, t in enumerate(t_eval)])
    asinh_V_true = np.arcsinh(V_true)

    # model present
    has_pinn   = ("PINN"   in models) and (models["PINN"]   is not None)
    has_mtpinn = ("MTPINN" in models) and (models["MTPINN"] is not None)

    V_pinn = None; V_mtp = None
    if has_pinn:
        V_pinn = np.array([_eval_model_value(models["PINN"], t, x_true_traj[i], T, use_tau_input)
                           for i, t in enumerate(t_eval)])
    if has_mtpinn:
        V_mtp  = np.array([_eval_model_value(models["MTPINN"], t, x_true_traj[i], T, use_tau_input)
                           for i, t in enumerate(t_eval)])

    # figure
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    colors = {
        "closed": "k",
        "PINN":   "#d62728",  # red
        "MTPINN": "#1f77b4",  # blue
    }

    # (Left) asinh(V) along optimal trajectory
    axL.plot(t_eval, asinh_V_true, color=colors["closed"], lw=2, label="closed")
    if has_pinn:
        axL.plot(t_eval, np.arcsinh(V_pinn), color=colors["PINN"],   lw=1.8, label="PINN")
    if has_mtpinn:
        axL.plot(t_eval, np.arcsinh(V_mtp),  color=colors["MTPINN"], lw=1.8, label="MT-PINN")
    axL.set_title(r"$\mathrm{asinh} (V(t, x^*(t)))$")
    axL.set_xlabel("time t"); axL.set_ylabel("asinh(value)")
    axL.grid(True, ls=":"); axL.legend()

    # (Right) asinh-error vs time
    if has_pinn:
        axR.plot(t_eval, _asinh_err(V_pinn, V_true), color=colors["PINN"],   label="PINN")
    if has_mtpinn:
        axR.plot(t_eval, _asinh_err(V_mtp,  V_true), color=colors["MTPINN"], label="MT-PINN")
    axR.set_title(r"asinh-error on $x^*(t)$ :  $|\mathrm{asinh}(V)-\mathrm{asinh}(V^*)|$")
    axR.set_xlabel("time t"); axR.set_ylabel("error")
    axR.grid(True, ls=":"); axR.legend()

    fig.suptitle("Value comparison on the optimal trajectory", y=0.98)
    plt.tight_layout()
    plt.show()

    # print stats
    print("=== asinh-error stats over t on x*(t) ===")
    if has_pinn:   print("PINN   ", _stats(V_pinn, V_true))
    if has_mtpinn: print("MT-PINN", _stats(V_mtp,  V_true))
