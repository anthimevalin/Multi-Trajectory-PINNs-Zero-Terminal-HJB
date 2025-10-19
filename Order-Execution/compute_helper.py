import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import jax.numpy as jnp
from jax import grad


def simulate_S(S0, sigma, T, Ndt, Npaths, seed=42):
    """Simulate stock price paths using Euler-Maruyama for driftless GBM

    Args:
        S0 (float): initial stock price
        sigma (float): volatility
        T (float): time horizon
        Ndt (int): distretization steps
        Npaths (int): number of simulated paths

    Returns:
        S (np.ndarray), dt (float): stock price paths, time step
    """
    np.random.seed(seed)
    dt = T / Ndt
    S = np.zeros((Npaths, Ndt + 1))
    S[:, 0] = S0
    for i in range(1, Ndt + 1):
        # Brownian motion
        dW = np.random.normal(0, np.sqrt(dt), Npaths)
        # Euler-Maruyama discretization of driftless GBM
        S[:, i] = S[:, i - 1] * (1 + sigma * dW)
    return S, dt

########################################################################################################################

def compute_value_function(T, X, S, lam, kappa, sigma):
    """Closed-form value function from Gatheral-Schied

    Args:
        T (float): time horizon
        X (float): inventory
        S (float): stock price
        lam (float): risk-aversion
        kappa (float): inventory-hold weighting
        sigma (float): volatility

    Returns:
        value_function (float): value function at (T, X, S)
    """
    coth_term = np.cosh(kappa*T)/(np.sinh(kappa*T))
    tanh_term = np.tanh(kappa*T/2)

    # integral term
    def integrand(t):
        return (np.tanh(kappa * t / 2))**2 * np.exp(-sigma**2 * t)
    
    integral, _ = quad(integrand, 0, T)

    # final expression
    term1 = kappa * X**2 * coth_term
    term2 = (lam * X * S / kappa) * tanh_term
    term3 = (lam**2 * S**2 * np.exp(sigma**2 * T) / (4 * kappa**2)) * integral
    value_function = term1 + term2 - term3
    return value_function


def compute_x_star(S, T, X, lam, kappa, dt):
    """Closed-form optimal inventory trajectory from Gatheral-Schied

    Args:
        S (np.ndarray): stock price paths 
        T (float): time horizon
        X (float): initial inventory
        lam (float): risk-aversion
        kappa (float): inventory-hold weighting
        dt (float): time step

    Returns:
        x_star (np.ndarray): optimal inventory trajectory
    """
    Npaths, Ndt = S.shape
    
    x_star = np.zeros((Npaths, Ndt))
    sinh_Tk = np.sinh(kappa * T)
    t_grid = np.linspace(0, T, Ndt)

    for i in range(Ndt):
        t = t_grid[i]        
        sinh_term = np.sinh((T-t)*kappa)
        
        # integration weights for s in [0, t]
        s_vals = t_grid[:i+1]    
        weights = 1 / (1 + np.cosh((T-s_vals) * kappa))
        
        # trapezoidal integration
        integrand = (S[:, :i+1] - X) * weights[None, :]
        integral = np.trapezoid(integrand, dx=dt, axis=1)
        
        bracket_term = X / sinh_Tk - (lam / (2*kappa)) * integral
        x_star[:, i] = sinh_term * bracket_term
        
    return x_star

def compute_v_star(x_star, S, T, lam, kappa, dt):
    """Closed-form optimal trading rate trajectory from Gatheral-Schied

    Args:
        x_star (np.ndarray): closed-form optimal inventory trajectory
        S (np.ndarray): stock price paths
        T (float): time horizon
        lam (float): risk-aversion
        kappa (float): inventory-hold weighting
        dt (float): time step

    Returns:
        v_start (np.ndarray): optimal trading rate trajectory
    """
    Npaths, Ndt1 = S.shape
    v_star = np.zeros((Npaths, Ndt1))
    t_grid = np.linspace(0, T, Ndt1)
   
    eps = 1e-6  # for numerical stability to avoid zero division

    for i in range(Ndt1):
        t = t_grid[i]
        if T - t < eps:
            v_star[:, i] = np.nan
            continue

        x_t = x_star[:, i]
        S_t = S[:, i]
        
        tau = T - t
        coth_term = 1.0 / np.tanh(kappa * tau)
        tanh_term = np.tanh((kappa * tau) / 2.0)
        
        first_term = x_t * kappa * coth_term
        second_term = (lam * S_t) / (2 * kappa) * tanh_term

        v_star[:, i] = first_term + second_term

    return v_star

###########################################################################################

"""PINN Trajectory Computation"""
def compute_pinn_trajectories(value_model, paths, T, X0, dt, d3=True):
    """Compute trajectories using value function output of PINN model

    Args:
        value_model (jaxlib._jax.PjitFunction): PINN model
        paths (np.ndarray): stock price paths
        T (float): time horizon
        X0 (float): initial inventory
        dt (float): time step
        d3 (bool): if lambda > 0. Defaults to True.

    Returns:
        x_pinn (np.ndarray), v_pinn (np.ndarray), value_pinn (np.ndarray): inventory, trading rate, value function trajectories from PINN model
    """
    Npaths, Ndt1 = paths.shape
    x_pinn = np.zeros((Npaths, Ndt1))
    v_pinn = np.zeros((Npaths, Ndt1))
    value_pinn = np.zeros((Npaths, Ndt1))
    x_pinn[:, 0] = X0

    # wrapper for PINN value function evaluation
    def f(z):
        out = value_model(z[None, :])   
        return jnp.squeeze(out)        

    f_dx = grad(lambda z: f(z), argnums=0) # derivative w.r.t state

    for p in range(Npaths):
        for i in range(Ndt1):
            tau = T - i * dt
            if d3:
                z = jnp.array([tau, x_pinn[p, i], paths[p, i]]) # (tau, X, S)
            else:
                z = jnp.array([tau, x_pinn[p, i]]) # (tau, X)

            gamma = float(f(z))
            value_pinn[p, i] = gamma

            if i < Ndt1 - 1:
                dGamma_dz = f_dx(z) # derivative Gamma w.r.t z
                dGamma_dx = float(dGamma_dz[1]) # derivative w.r.t x

                v = 0.5 * dGamma_dx # optimal trading rate under HJB
                v_pinn[p, i] = v

                # forward Euler inventory update
                x_pinn[p, i + 1] = x_pinn[p, i] - v * dt
            else:
                v_pinn[p, i] = np.nan

    return x_pinn, v_pinn, value_pinn
