
# ===== market_eval.py ===== 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Callable, List, Optional, Tuple
from matplotlib.lines import Line2D
import jax, jax.numpy as jnp
from jax import grad, jit, lax
from loading_helper import ann_gen
from matplotlib.patches import Patch

# Global constants
TRADING_HOURS_PER_DAY = 6.5
SECS_PER_TRADING_DAY  = TRADING_HOURS_PER_DAY * 3600.0
NY_TZ = "America/New_York"




################################### Read BBO-1s and make windows ########################################
def _pick(name_list, candidates):
    for c in candidates:
        if c in name_list:
            return c
    return None

def read_bbo_file(path):
    """Read BBO-1s file and returns DataFrame with columns: ['timestamp','mid'] where timestamp NY timezone

    Args:
        path (str): path of BBO-1s file (csv or zstd-compressed csv)

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    try:
        df = pd.read_csv(path, compression="zstd")
    except Exception:
        df = pd.read_csv(path)

    cols = df.columns
    ts_col  = _pick(cols, ["ts_recv", "ts", "timestamp", "time"])
    bid_col = _pick(cols, ["bid_px_00", "bid_px", "bid_price", "bid"])
    ask_col = _pick(cols, ["ask_px_00", "ask_px", "ask_price", "ask"])
    if ts_col is None or bid_col is None or ask_col is None:
        raise ValueError(f"Can't find timestamp/bid/ask columns in {path}. Got {list(cols)}")

    out = df[[ts_col, bid_col, ask_col]].rename(
        columns={ts_col: "timestamp_utc", bid_col: "bid", ask_col: "ask"}
    )
    # timestamp -> UTC -> NY
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp_utc"])
    out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
    out = out.dropna(subset=["bid", "ask"])

    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["timestamp"] = out["timestamp_utc"].dt.tz_convert(NY_TZ)
    out = out[["timestamp", "mid"]].sort_values("timestamp").reset_index(drop=True)
    return out

def regularize_to_grid(df_local, start_local, end_local, freq = "5s"):
    """Put window on an exact grid [start, end) with step 'freq'

    Args:
        df_local (pd.DataFrame): DataFrame of ['timestamp','mid'] in NY time.
        start_local (pd.Timestamp): start time in NY time
        end_local (pd.Timestamp): end time in NY time
        freq (str): frequency of data/grid

    Returns:
        pd.DataFrame: DataFrame with ['timestamp','mid'] on exact grid
    """
    # make sure start_local, end_local are tz-aware NY
    idx = pd.date_range(start=start_local, end=end_local - pd.Timedelta(seconds=1), freq=freq, tz=NY_TZ)
    
    # reindex + ffill + bfill
    s = (df_local.set_index("timestamp")["mid"].reindex(idx).ffill().bfill())
    
    return pd.DataFrame({"timestamp": idx, "mid": s.values})

def extract_windows_for_day(df_local, date_local, start_times=("09:45", "11:45", "13:45"), duration="2h", grid_freq = "5s", min_coverage = 0.80):
    """Extracts intraday windows from one day's data.

    Args:
        df_local (pd.DataFrame): DataFrame of ['timestamp','mid'] in NY time.
        date_local (str): date in NY time
        start_times (tuple): tuple of start times in NY time. Defaults to ("09:45", "11:45", "13:45").
        duration (str): duration of each window. Defaults to "2h".
        grid_freq (str): frequency of data/grid. Defaults to "5s".
        min_coverage (float): minimum coverage ratio (0-1) to accept a window. Defaults to 0.80.

    Returns:
        out (dict): {'date','start','end','window','df'}
    """
    
    out = []
    for i, st in enumerate(start_times, start=1):
        start = pd.Timestamp(f"{date_local} {st}", tz=NY_TZ)
        end   = start + pd.Timedelta(duration)
        # slice
        m = (df_local["timestamp"] >= start) & (df_local["timestamp"] < end)
        df_w = df_local.loc[m, ["timestamp", "mid"]].copy()

        # coverage filter (raw observations vs expected steps at 1-second cadence)
        exp_seconds = int((end - start).total_seconds())
        got = len(df_w)
        if got < min_coverage * exp_seconds:
            continue

        df_w = regularize_to_grid(df_w, start, end, freq=grid_freq)

        out.append({
            "date": pd.Timestamp(date_local).date(),
            "start": start,
            "end": end,
            "window": i,
            "df": df_w
        })
    return out

def make_intraday_windows(paths, start_times=("09:45","11:45","13:45"), duration="2h", grid_freq = "5s", min_coverage = 0.80):
    """Extracts intraday windows from multiple BBO-1s files.

    Args:
        paths (list): list of paths of BBO-1s files (csv or zstd-compressed csv).
        start_times (tuple): tuple of start times in NY time. Defaults to ("09:45", "11:45", "13:45").
        duration (str): duration of each window. Defaults to "2h".
        grid_freq (str): frequency of data/grid. Defaults to "5s".
        min_coverage (float): minimum coverage ratio (0-1) to accept a window. Defaults to 0.80.

    Returns:
        combined (pd.DataFrame), windows (list): combined DataFrame with all windows + list of raw windows (dicts).
    """
    
    # reads each file, splits by calendar date (NY time)
    windows: List[dict] = []
    for p in paths:
        df = read_bbo_file(p)
        for date_val, df_day in df.groupby(df["timestamp"].dt.date, sort=True):
            day_start = pd.Timestamp(f"{date_val} 09:30", tz=NY_TZ)
            day_end   = pd.Timestamp(f"{date_val} 16:00", tz=NY_TZ)
            mkt = df_day[(df_day["timestamp"] >= day_start) & (df_day["timestamp"] < day_end)].copy()
            if mkt.empty:
                continue
            # for each date, extracts windows
            ws = extract_windows_for_day(mkt, date_val,
                                         start_times=start_times,
                                         duration=duration,
                                         grid_freq=grid_freq,
                                         min_coverage=min_coverage)
            windows.extend(ws)

    # combines all windows into one DataFrame 
    frames = []
    for w in windows:
        df_w = w["df"].copy()
        df_w["date"] = pd.to_datetime(w["date"]).date()
        df_w["window"] = int(w["window"])
        df_w["window_id"] = f"{w['date']}_w{w['window']}"
        frames.append(df_w)

    if frames:
        combined = pd.concat(frames, axis=0, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["timestamp","mid","date","window","window_id"])

    return combined, windows

# Window container 
@dataclass
class Window:
    date: pd.Timestamp           # trading date
    window_id: str               # e.g. '2025-02-10_w1'
    t: pd.DatetimeIndex          # NY timestamps
    mid: np.ndarray              # mid prices
    S0: float                    # first mid
    dt_sec: float                # sampling step (seconds)
    dt_days: float               # dt in trading days
    T_days: float                # horizon in trading days
    S_norm: np.ndarray           # mid / S0
    t_days: np.ndarray           # running time in days


def make_window(df, date, window_id):
    """
    Constructs a Window object from raw DataFrame of ['timestamp','mid']
    Args:
        df (pd.DataFrame): DataFrame of ['timestamp','mid']
        date (pd.Timestamp): date of the window
        window_id (str): unique window id

    Returns:
        Window: window object
    """
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    t = pd.DatetimeIndex(df["timestamp"])
    if t.tz is None:
        t = t.tz_localize(NY_TZ)
    else:
        t = t.tz_convert(NY_TZ)

    m = df["mid"].astype(float).to_numpy()

    # median spacing in seconds
    dsec = (t[1:] - t[:-1]).total_seconds().astype(float)
    dt_sec  = float(np.median(dsec))
    dt_days = dt_sec / SECS_PER_TRADING_DAY
    T_days  = dt_days * (len(m) - 1)

    S0     = float(m[0])
    S_norm = m / S0
    t_days = np.arange(len(m), dtype=float) * dt_days

    return Window(date=pd.Timestamp(date), window_id=window_id,
                  t=t, mid=m, S0=S0, dt_sec=dt_sec, dt_days=dt_days,
                  T_days=T_days, S_norm=S_norm, t_days=t_days)

def windows_from_list(raw_windows):
    """Converts a list of raw window dicts to a list of Window objects.
    Args:
        raw_windows (List[dict]): list of dicts of windows

    Returns:
        List[Window]: list of window objects
    """
    
    out = []
    for w in raw_windows:
        date_ts = pd.Timestamp(w["date"])
        win_id = w.get("window_id")
        if win_id is None:
            idx = int(w.get("window", 1))
            win_id = f"{date_ts.date()}_w{idx}"
        out.append(make_window(w["df"], date_ts, win_id))
    return out

################################### Metrics (cost and exposure) ########################################

def implementation_shortfall_bps(mid, q, S0, X0):
    """ Calculates implementation shortfall in bps where positive is worse and enforces exact finish

    Args:
        mid (np.ndarray): mid prices
        q (np.ndarray): trade sizes (sell is +, buy is -)
        S0 (float): initial mid price
        X0 (float): initial shares

    Returns:
        float: implementation shortfall in bps
    """
    
    # force sum(q) == |X0|
    total = float(np.sum(q))
    if abs(total - abs(X0)) > 1e-8:
        q = q * (abs(X0) / total)

    pv_exec  = float(np.dot(mid[:len(q)], q))
    pv_bench = float(S0 * abs(X0))

    cost = pv_bench - pv_exec   
    

    return 1e4 * cost / pv_bench


def exposure_proxy(x, X0):
    """Computes exposure proxy E[x_t^2]/X0^2"""
    
    return float(np.mean((x[:-1] / X0) ** 2))


################################### Policies ########################################
class TWAPPolicy:
    def __init__(self, name="TWAP"):
        self.name = name 

    def simulate(self, W, X0):
        # equal-sized slices over the N steps 
        N = len(W.mid) - 1
        q = np.full(N, X0 / N, dtype=float)

        # inventory path x_t with x_0 = X0
        x = np.empty(N + 1, dtype=float); x[0] = X0
        for k in range(N):
            x[k+1] = x[k] - q[k]

        # ensure finish exactly |X0|
        assert abs(np.sum(q) - abs(X0)) < 1e-8  # exact finish

        # execution rate in shares/day
        v = q / W.dt_days

        # cost/risk metrics
        cost = implementation_shortfall_bps(W.mid, q, W.S0, X0)
        exp  = exposure_proxy(x, X0)
        return {"x": x, "q": q, "v": v, "cost_bps": cost, "exposure": exp}


class PINNPolicy:
    """JAX-compiled simulator (lax.scan)."""
    def __init__(self, cfg, params, use_S=True, name="PINN"):
        self.name = name
        self.use_S = use_S     

        # build NN 
        ann = ann_gen(cfg)
        params_static = params

        @jit
        def U(z):
            return ann.apply(params_static, z[None, :])[0, 0]

        dU = jit(grad(U))

        @jit
        def _simulate_core(t_days, s_feat, dt_days, T_days, X0):
            # one scan step over time
            def step(x_norm, inputs):
                ti, si = inputs
                tau = jnp.clip(T_days - ti, 0.0, cfg.T)
                z = jnp.array([tau, x_norm, si]) if self.use_S else jnp.array([tau, x_norm])

                # control from value gradient
                u = U(z)
                ux = dU(z)[1]
                
                vnorm = 0.5 * ux 

                # sell-only
                vnorm = jnp.maximum(0.0, vnorm)

                # forward Euler 
                x_next = x_norm - vnorm * dt_days
                x_next = jnp.maximum(0.0, x_next)
                return x_next, (x_next, vnorm)

            # start fully long in normalized units
            x0 = jnp.array(1.0, dtype=jnp.float32)
            inputs = (t_days, s_feat if self.use_S else jnp.zeros_like(t_days))

            # scan over time steps
            _, (x_next_all, vnorm) = lax.scan(step, x0, inputs)

            # recover full path 
            x_norm_path = jnp.concatenate([x0[None], x_next_all])

            # positive trades only
            q_norm = jnp.maximum(0.0, x_norm_path[:-1] - x_norm_path[1:])
            q = X0 * q_norm
            leftover = x_norm_path[-1] * X0
            q = q.at[-1].add(leftover)

            # rebuild x from q so the path matches the enforced finish
            x = jnp.empty((x_norm_path.shape[0],), dtype=jnp.float32)
            x = x.at[0].set(X0)
            x = x.at[1:].set(X0 - jnp.cumsum(q))
            
            # Shares/day
            v = q / dt_days
            return x, q, v

        self._simulate_core = _simulate_core

    def simulate(self, W, X0):
        x, q, v = self._simulate_core(
            jnp.asarray(W.t_days[:-1], dtype=jnp.float32),
            jnp.asarray(W.mid[:-1], dtype=jnp.float32),
            jnp.asarray(W.dt_days, dtype=jnp.float32),
            jnp.asarray(W.T_days, dtype=jnp.float32),
            jnp.asarray(X0, dtype=jnp.float32),
        )
        x = np.array(x)
        q = np.array(q)
        v = np.array(v)

        # cost/risk metrics
        cost = implementation_shortfall_bps(W.mid, q, W.S0, X0)
        exp  = exposure_proxy(x, X0)
        return {"x": x, "q": q, "v": v, "cost_bps": cost, "exposure": exp}


################################### Evaluation and plotting ########################################
def evaluate_models_over_windows(models, windows, X0_shares):
    """Evaluate multiple models over multiple windows

    Args:
        models (Policy): dict of model to evaluate
        windows (list): list of windows
        X0_shares (int): initial shares (not normalized)

    Returns:
        pd.DataFrame: DataFrame with columns: ['date','window_id','model','cost_bps','exposure','X0','S0'] 
    """
    rows = []
    trajs = {name: {} for name in models}
    for W in windows:
        for name, policy in models.items():
            sim = policy.simulate(W, X0_shares)
            rows.append({
                "date": W.date.date(),
                "window_id": W.window_id,
                "model": name,
                "cost_bps": sim["cost_bps"],
                "exposure": sim["exposure"],
                "X0": X0_shares,
                "S0": W.S0,
            })
            trajs[name][W.window_id] = {"window": W, **sim}
    return pd.DataFrame(rows),{"trajectories": trajs}


def _as_ny(idx_like):
    dtidx = pd.DatetimeIndex(idx_like)
    if dtidx.tz is None:
        return dtidx.tz_localize(NY_TZ)
    return dtidx.tz_convert(NY_TZ)


def plot_window_overlay(traj_by_model_or_multi, ncols = 2):
    """overlay prices with inventory trajectories

    Args:
        traj_by_model_or_multi (dict): allows for multiple windows for each model
        ncols (int): number of subplot columns
    """

    # detect single vs multi
    is_single = False
    if isinstance(traj_by_model_or_multi, dict):
        sample_val = next(iter(traj_by_model_or_multi.values()))
        is_single = isinstance(sample_val, dict) and ("window" in sample_val)

    if is_single:
        any_traj = next(iter(traj_by_model_or_multi.values()))
        wid = any_traj["window"].window_id
        multi = {wid: traj_by_model_or_multi}
    else:
        multi = traj_by_model_or_multi

    # model order 
    first_wid = next(iter(multi.keys()))
    model_names = list(multi[first_wid].keys())

    custom_colors = {
        "TWAP": "#1f77b4",
        "MT-PINN λ=0.00": "#ff7f0e",  # blue
        "MT-PINN λ=0.05": "green",  # orange
        "MT-PINN λ=0.10": "#d62728",  # red
    }
    inv_colors = {name: custom_colors.get(name, "gray") for name in model_names}
    mid_color = "black"

    # figure
    ids = list(multi.keys())
    n = len(ids)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    plt.style.use("seaborn-v0_8-dark")
    plt.rcParams.update({"font.size": 12,
                         "axes.titlesize": 13,
                         "axes.labelsize": 12})

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10 * ncols, 4.5 * nrows), squeeze=False
    )

    # plotting loop 
    for i, wid in enumerate(ids):
        ax1 = axes[i // ncols][i % ncols]
        any_traj = next(iter(multi[wid].values()))
        W = any_traj["window"]

        # mid price
        ax1.plot(_as_ny(W.t), W.mid, color=mid_color, alpha=0.9, lw=0.9, label="mid")
        ax1.set_ylabel("Price")
        ax1.set_xlabel("Time (ET)")
        ax1.grid(True, which="both", axis="both")

        # inventories
        ax2 = ax1.twinx()
        for name, d in multi[wid].items():
            ls = "--" if name == "TWAP" else "-"
            lw = 2.0 if "0.10" in name else 1.5
            ax2.plot(
                _as_ny(W.t),
                d["x"],
                lw=lw,
                ls=ls,
                alpha=0.95,
                color=inv_colors.get(name, "gray"),
                label=f"{name} inventory",
            )
        ax2.set_ylabel("Inventory (shares)")
        ax2.grid(False) 
        ax2.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.6)

        # panel title
        cost_bits = " | ".join(
            f"{k}: {multi[wid][k]['cost_bps']:.1f} bps" for k in model_names if k in multi[wid]
        )
        ax1.set_title(f"{W.date.date()}  |  {cost_bits}")

    # hide unused subplots
    for j in range(n, nrows * ncols):
        ax = axes[j // ncols][j % ncols]
        ax.axis("off")

    # shared legend 
    handles = [Line2D([0], [0], color=mid_color, lw=1.2, label="mid")]
    for m in model_names:
        ls = "--" if m == "TWAP" else "-"
        lw = 2.0 if "0.10" in m else 1.5
        handles.append(Line2D([0], [0], color=inv_colors[m], lw=lw, ls=ls, label=f"{m} inventory"))

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.show()


def plot_spy_prices(combined_df):
    """Plots SPY prices with intraday windows highlighted

    Args:
        combined_df (pd.DataFrame): DataFrame with columns: ['timestamp','mid','date','window','window_id']
    """
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # sort data by timestamp
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    # colors for each window
    window_colors = {1: '#A23B72', 2: '#F18F01', 3: '#C73E1D'}
    
    # create sequential x-axis
    combined_df['x_idx'] = range(len(combined_df))
    
    # plot each window separately 
    for window_id in combined_df['window_id'].unique():
        window_data = combined_df[combined_df['window_id'] == window_id]
        window_num = window_data['window'].iloc[0]
        
        ax.plot(window_data['x_idx'], window_data['mid'], 
                color='#2E86AB', linewidth=1, alpha=0.9)
    
    # vertical lines and labels for windows
    prev_date = None
    for window_id in combined_df['window_id'].unique():
        window_data = combined_df[combined_df['window_id'] == window_id]
        window_num = window_data['window'].iloc[0]
        date = window_data['date'].iloc[0]
        
        start_idx = window_data['x_idx'].iloc[0]
        end_idx = window_data['x_idx'].iloc[-1]
        
        # vertical line at start of window
        ax.axvline(x=start_idx, color=window_colors[window_num], 
                  linestyle='-', alpha=0.3, linewidth=1)
        
        # vertical line at end of window
        ax.axvline(x=end_idx, color=window_colors[window_num], 
                  linestyle='--', alpha=0.3, linewidth=1)
        
        # window label
        ax.text((start_idx + end_idx) / 2, 
               ax.get_ylim()[1] * 0.99,
               f'W{window_num}', 
               ha='center', va='top',
               fontsize=9, color=window_colors[window_num],
               fontweight='bold')
        
        # date separator if its new day
        if prev_date is not None and date != prev_date:
            ax.axvline(x=start_idx, color='black', 
                      linestyle='-', alpha=0.5, linewidth=2)
            # date label
            ax.text(start_idx, ax.get_ylim()[0], 
                   date.strftime('%b %d'),
                   rotation=90, va='bottom', ha='right',
                   fontsize=9, color='black', alpha=0.7)
        
        prev_date = date
    
    # x-axis labels
    num_labels = 15
    label_indices = np.linspace(0, len(combined_df)-1, num_labels, dtype=int)
    
    ax.set_xticks(label_indices)
    ax.set_xticklabels([combined_df.iloc[i]['timestamp'].strftime('%m/%d %H:%M') 
                        for i in label_indices], rotation=45, ha='right')
    
    ax.set_xlabel('Date and Time (ET)', fontsize=11)
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.set_title('SPY Price Evolution with Intraday Windows', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    
    # legend
    legend_elements = [
        Patch(facecolor=window_colors[1], alpha=0.5, label='Window 1 (09:45-11:45)'),
        Patch(facecolor=window_colors[2], alpha=0.5, label='Window 2 (11:45-13:45)'),
        Patch(facecolor=window_colors[3], alpha=0.5, label='Window 3 (13:45-15:45)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # adjust layout
    plt.tight_layout()
    plt.show()
    
    # summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"Price range: ${combined_df['mid'].min():.2f} - ${combined_df['mid'].max():.2f}")
    print(f"Total windows: {combined_df['window_id'].nunique()}")
    print(f"Data points: {len(combined_df):,}")
    print(f"Windows per day:")
    for date in sorted(combined_df['date'].unique()):
        windows_on_date = combined_df[combined_df['date'] == date]['window'].unique()
        print(f"  {date}: Windows {sorted(windows_on_date)}")
        
        
        
def _sigma_day_from_window(W):
    """Daily return vol from a single window, using log-returns and scaling to 1 trading day"""
    s = np.asarray(W.mid, float)
    if len(s) < 2:
        return np.nan
    # log returns
    r = np.diff(np.log(s)) 
    window_seconds = W.dt_sec * (len(s) - 1)
    if window_seconds <= 0:
        return np.nan
    # realized variance over window
    rv = float(np.sum(r * r))  
    var_day = rv * (SECS_PER_TRADING_DAY / window_seconds)
    return float(np.sqrt(max(var_day, 0.0)))

def simple_T_sigma_srange(Ws, q_low = 0.01, q_high = 0.99):
    """return dictionary with T_days, sigma_day, s_range (in dollars) from a list of windows

    Args:
        Ws (list): list of windows
        q_low (float): quantile for s_range low
        q_high (float): quantile for s_range high

    Returns:
        dict: {"T_days": T_days, "sigma_day": sigma_day, "s_range": (s_lo, s_hi)}
    """
    
    
    
    # trading days
    T_days = float(np.median([W.T_days for W in Ws])) if Ws else float("nan")

    # sigma_day 
    sigmas = np.array([_sigma_day_from_window(W) for W in Ws], float)
    sigmas = sigmas[np.isfinite(sigmas)]
    sigma_day = float(np.median(sigmas)) if sigmas.size else float("nan")

    # s-range in dollars
    all_S = np.concatenate([np.asarray(W.mid, float) for W in Ws]) if Ws else np.array([])
    all_S = all_S[np.isfinite(all_S)]
    if all_S.size:
        s_lo = float(np.quantile(all_S, q_low))
        s_hi = float(np.quantile(all_S, q_high))
    else:
        s_lo = s_hi = float("nan")

    return {"T_days": T_days, "sigma_day": sigma_day, "s_range": (s_lo, s_hi)}


def compute_cost_variance_points(results, models):
    """ Points for cost-variance plot and returns: window_id, model, exposure, cost_bps for the chosen models"""
    cols = ["window_id", "model", "exposure", "cost_bps"]
    R = results.loc[:, cols].copy()
    if models is not None:
        R = R[R["model"].isin(models)]
    return R


def plot_cost_variance_scatter(points, order = None, connect_order = None):
    """cost-variance scatter plot across windows and models

    Args:
        points (pd.DataFrame): points for plotting
        order (list): list of model names to order
        connect_order (list): list of model names to connect by their means
    """
    
    if order is None:
        order = sorted(points["model"].unique(), key=str)

    colors = plt.cm.tab10.colors
    cmap = {name: colors[i % len(colors)] for i, name in enumerate(order)}

    fig, ax = plt.subplots(figsize=(7, 4))

    # per-window scatter
    for name in order:
        P = points[points["model"] == name]
        ax.scatter(P["exposure"], P["cost_bps"], s=26, alpha=0.85,
                   label=name, color=cmap[name])

    # mean markers and text
    means = []
    for name in order:
        P = points[points["model"] == name]
        mx, my = P["exposure"].mean(), P["cost_bps"].mean()
        means.append((name, mx, my))
        ax.scatter([mx], [my], s=60, marker="D", edgecolor="k",
                   color=cmap[name], zorder=3)

    # connect selected models 
    if connect_order:
        xs, ys = [], []
        for name in connect_order:
            row = next(((n, x, y) for (n, x, y) in means if n == name), None)
            if row is not None:
                xs.append(row[1]); ys.append(row[2])
        if len(xs) >= 2:
            ax.plot(xs, ys, lw=1.6, alpha=0.9, color="C7")

    ax.axvline(1/3, ls="--", lw=1, alpha=0.5, color="gray")
    ax.set_xlabel("Risk (exposure proxy)")
    ax.set_ylabel(r"Realized Cost (bps of $|X_0|S_0$)")
    ax.set_title(r"Cost–Variance Trade-off: MT-PINN (Varying $\lambda$) vs. TWAP Benchmark")
    ax.grid(True, alpha=0.8)
    ax.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.show()


def summarize_cost_variance(points):
    """summary stats for cost-variance means and std"""
    return (
        points.groupby("model", as_index=False)
              .agg(exposure_mean=("exposure", "mean"),
                   exposure_std=("exposure", "std"),
                   cost_bps_mean=("cost_bps", "mean"),
                   cost_bps_std=("cost_bps", "std"),
                   n=("cost_bps", "size"))
              .sort_values("model")
    )


################################################################################
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- One-shot Bloomberg-like style (dark panel, subtle grid, light text) ---
def _set_bbg_style():
    mpl.rcParams.update({
        "figure.facecolor": "#0b0c0e",
        "axes.facecolor":   "#0b0c0e",
        "axes.edgecolor":   "#3a3d40",
        "axes.labelcolor":  "#d6d6d6",
        "xtick.color":      "#d6d6d6",
        "ytick.color":      "#d6d6d6",
        "text.color":       "#d6d6d6",
        "grid.color":       "#4a4d50",
        "grid.linestyle":   "--",
        "grid.linewidth":   0.6,
        "axes.grid":        True,
        "font.size":        11,
        "savefig.facecolor":"#0b0c0e",
        "savefig.edgecolor":"#0b0c0e",
        "axes.titleweight": "bold",
        "axes.titlelocation":"left",
        "figure.autolayout": True,
        "legend.frameon":   False,
    })

def plot_spy_prices_bbg(combined_df):
    """Bloomberg-like plot of SPY prices with intraday windows highlighted.

    Args:
        combined_df (pd.DataFrame): columns ['timestamp','mid','date','window','window_id']
    """

    _set_bbg_style()

    fig, ax = plt.subplots(figsize=(16, 8), dpi=120)

    # --- data prep ---
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    combined_df['x_idx'] = np.arange(len(combined_df))

    # --- palette (muted window markers + amber primary) ---
    price_color   = "#f5b800"  # amber highlight (Bloomberg-like)
    window_colors = {1: "#7aa2ff", 2: "#8be9a8", 3: "#ff79aa"}  # muted/cool contrasts on dark

    # --- plot main line (single color for consistency) ---
    ax.plot(combined_df['x_idx'], combined_df['mid'],
            color=price_color, linewidth=1.6, alpha=0.95, label="SPY Mid")

    # --- window vertical markers + labels ---
    prev_date = None
    y_top = combined_df['mid'].max()
    y_bot = combined_df['mid'].min()
    y_pad = 0.02 * (y_top - y_bot)
    for window_id in combined_df['window_id'].unique():
        w = combined_df[combined_df['window_id'] == window_id]
        window_num = int(w['window'].iloc[0])
        date = w['date'].iloc[0]

        start_idx = int(w['x_idx'].iloc[0])
        end_idx   = int(w['x_idx'].iloc[-1])

        # start/end markers
        ax.axvline(x=start_idx, color=window_colors[window_num], linestyle='-',  alpha=0.30, linewidth=1.0)
        ax.axvline(x=end_idx,   color=window_colors[window_num], linestyle='--', alpha=0.30, linewidth=1.0)

        # window label (centered, just under top)
        ax.text((start_idx + end_idx)/2,
                y_top - y_pad,
                f"W{window_num}",
                ha='center', va='top',
                fontsize=9, color=window_colors[window_num], fontweight='bold')

        # date separator when new day starts
        if prev_date is not None and date != prev_date:
            ax.axvline(x=start_idx, color="#9aa0a6", linestyle='-', alpha=0.55, linewidth=1.2)
            ax.text(start_idx, y_bot + y_pad,
                    pd.to_datetime(date).strftime('%b %d'),
                    rotation=90, va='bottom', ha='right',
                    fontsize=9, color="#b8bcbc", alpha=0.85)
        prev_date = date

    # --- axes: ticks, grid, spines ---
    num_labels = 15
    label_idx = np.linspace(0, len(combined_df)-1, num_labels, dtype=int)
    ax.set_xticks(label_idx)
    ax.set_xticklabels(
        [pd.to_datetime(combined_df.iloc[i]['timestamp']).strftime('%m/%d %H:%M') for i in label_idx],
        rotation=45, ha='right'
    )
    ax.set_xlabel('Date & Time (ET)')
    ax.set_ylabel('Price ($)')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#3a3d40")

    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.5)

    # --- title & subtitle (left-aligned) ---
    ax.set_title("SPY — Intraday Price with Windows", loc="center")


    # --- legend (minimal, Bloomberg-like) ---
    legend_items = [
        Line2D([0], [0], color=price_color, lw=2, label="SPY Mid"),
        Patch(facecolor=window_colors[1], alpha=0.5, label='W1 (09:45–11:45)'),
        Patch(facecolor=window_colors[2], alpha=0.5, label='W2 (11:45–13:45)'),
        Patch(facecolor=window_colors[3], alpha=0.5, label='W3 (13:45–15:45)'),
    ]
    leg = ax.legend(handles=legend_items, loc='best', fontsize=9, ncol=1)
    for txt in leg.get_texts():
        txt.set_color("#d6d6d6")

    plt.tight_layout()
    plt.show()

    # --- summary (unchanged) ---
    print("\n=== Summary Statistics ===")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"Price range: ${combined_df['mid'].min():.2f} - ${combined_df['mid'].max():.2f}")
    print(f"Total windows: {combined_df['window_id'].nunique()}")
    print(f"Data points: {len(combined_df):,}")
    print(f"Windows per day:")
    for date in sorted(combined_df['date'].unique()):
        windows_on_date = combined_df[combined_df['date'] == date]['window'].unique()
        print(f"  {date}: Windows {sorted(windows_on_date)}")
