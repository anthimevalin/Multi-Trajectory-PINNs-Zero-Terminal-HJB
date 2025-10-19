
import jax.numpy as jnp
from jax import grad
import numpy as np
from numpy.linalg import solve
from scipy.linalg import solve_continuous_are, expm, norm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def construct_matrices(dim = 20, shift_factor = 0.1):
    """Build block-diagonal (A, B, Q, R, S) for a dim-state LQR by tiling a 4×4 prototype.

    Args:
        dim (int): wanted dimensions
        shift_factor (float): Per-block negative diagonal shift added to A to vary dynamics

    Returns:
        A (jnp.array), B (jnp.array), Q (jnp.array), R (jnp.array), S (jnp.array): the desired matrices
    """
    
    dtype=jnp.float32

    # Base 4×4 prototype (Ntogramatzidis, 2003)
    A_base = jnp.array([
        [-8.95,  -6.45,   0.00,   0.00],
        [ 2.15,  -0.35,   0.00,   0.00],
        [-10.89, -40.94, -16.10, -7.95],
        [ 8.17,  28.87,   7.07,  -0.20]
    ], dtype=dtype)

    B_base = jnp.array([
        [1., 0.],
        [0., 0.],
        [0., 1.],
        [1., 1.]
    ], dtype=dtype)

    Q_base = jnp.array([
        [ 5.,  4., 13., 16.],
        [ 4.,  5., 11., 14.],
        [13., 11., 34., 42.],
        [16., 14., 42., 52.]
    ], dtype=dtype)

    R_base = jnp.eye(2, dtype=dtype)

    S_base = jnp.array([
        [1., 2.],
        [2., 1.],
        [3., 5.],
        [4., 6.]
    ], dtype=dtype)

    # Small utility: block-diagonal assembly 
    def blkdiag(*blocks):
        n = len(blocks)
        return jnp.block([
            [blocks[i] if i == j else jnp.zeros((blocks[i].shape[0], blocks[j].shape[1]), dtype=dtype)
             for j in range(n)]
            for i in range(n)
        ])

    full = dim // 4         # number of full 4×4 blocks
    rem  = dim % 4          # size of the final truncated block 
    n_blocks = full + (1 if rem > 0 else 0)

    A_blocks, B_blocks, Q_blocks, R_blocks, S_blocks = [], [], [], [], []

    for j in range(n_blocks):
        # block size r: 4 for full blocks; 'rem' for the last (if rem > 0)
        r = 4 if (j < full) else (rem if rem > 0 else 4)

        # slice top-left sub-blocks of proper size
        A_j = A_base[:r, :r]
        B_j = B_base[:r, :]          # r×2
        Q_j = Q_base[:r, :r]
        S_j = S_base[:r, :]          # r×2
        R_j = R_base                 # 2×2

        # vary dynamics slightly per block with a diagonal shift
        shift = shift_factor * (j + 1)
        A_j = A_j - shift * jnp.eye(r, dtype=dtype)

        A_blocks.append(A_j)
        B_blocks.append(B_j)
        Q_blocks.append(Q_j)
        R_blocks.append(R_j)
        S_blocks.append(S_j)

    # assemble big block-diagonal matrices
    A = blkdiag(*A_blocks)                      # (dim, dim)
    B = blkdiag(*B_blocks)                      # (dim, 2*n_blocks)
    Q = blkdiag(*Q_blocks)                      # (dim, dim)
    R = blkdiag(*R_blocks)                      # (2*n_blocks, 2*n_blocks)
    S = blkdiag(*S_blocks)                      # (dim, 2*n_blocks)

    #  shape checks 
    assert A.shape == (dim, dim), f"A shape {A.shape} != {(dim, dim)}"
    assert Q.shape == (dim, dim), f"Q shape {Q.shape} != {(dim, dim)}"
    assert B.shape[0] == dim and S.shape[0] == dim, "B/S row dims must match 'dim'"
    assert R.shape[0] == R.shape[1] == 2 * n_blocks, "R must be 2×2 per block"


    return A, B, Q, R, S


def closed_form_P_and_K(A, B, Q, R, S, T=1.0, eps=1e-9):
    P1 = solve_continuous_are(A, B, Q, R, None, S)
    if norm(P1) < 1e-10:
        P1 = np.zeros_like(P1)
    K1 = np.linalg.solve(R, B.T @ P1 + S.T)
    P2 = solve_continuous_are(-A, -B, Q, R, None, S)
    K2 = np.linalg.solve(R, S.T - B.T @ P2)
    F1 = A - B @ K1
    F2 = A - B @ K2
    E_F1_T = expm(F1 * T)

    def X_of_t(t):
        E_F1_t = expm(F1 * t)
        E_F2_tT = expm(F2 * (t - T))
        return E_F1_t - E_F2_tT @ E_F1_T

    def Lambda_of_t(t):
        E_F1_tT = expm(F1 * (t - T))
        E_F2_tT = expm(F2 * (t - T))
        return (P1 @ E_F1_tT + P2 @ E_F2_tT) @ expm(F1 * T)

    def P_of_t(t):
        if not (0.0 <= t < T - eps):
            raise ValueError("t must be in [0, T - eps)")
        P = Lambda_of_t(t) @ np.linalg.inv(X_of_t(t))
        return 0.5 * (P + P.T) 

    def K_of_t(t):
        P = P_of_t(t)
        return np.linalg.solve(R, B.T @ P + S.T)

    return P_of_t, K_of_t, X_of_t, Lambda_of_t  

def closed_value_function(t, x, P_of_t):
    P = P_of_t(t)
    return np.dot(x.T, P @ x)


def traj_PINN(model, x0, T, t_eval, A, B, R, S, Ndt=200):
    A_np, B_np, S_np, R_np = map(np.asarray, (A, B, S, R))
    invR_np = np.linalg.inv(R_np)

    dt = T / (Ndt - 1)

    x_pinn = np.zeros((Ndt, x0.size))
    u_pinn = np.zeros((Ndt, B_np.shape[1]))
    value_pinn = np.zeros(Ndt)

    x_pinn[0] = x0

    def V_of_z(z):               
        return jnp.squeeze(model(z[None, :]))

    def grad_V(t, x_np):          
        x_jax = jnp.asarray(x_np)
        def V_of_x(x_inner):
            z = jnp.concatenate([jnp.array([t]), x_inner])
            return V_of_z(z)
        return np.asarray(grad(V_of_x)(x_jax))

    for i, t in enumerate(t_eval):
        x = x_pinn[i]       

        z_jax = jnp.concatenate([jnp.array([t]), jnp.asarray(x)])
        V = V_of_z(z_jax)           
        value_pinn[i] = V

        if i < Ndt - 1:

            term = 0.5 * (B_np.T @ grad_V(t, x)) + S_np.T @ x   
            u = -invR_np @ term                       

            u_pinn[i] = u
            x_pinn[i+1] = x + dt * (A_np @ x + B_np @ u)
        else:
            u_pinn[i] = np.nan         

    return x_pinn, u_pinn, value_pinn

def traj_closed_form(A, B, Q, R, S, x0, t_eval, T=1.0, eps=1e-9):
    P_of_t, K_of_t, X_of_t, Lambda_of_t = closed_form_P_and_K(A, B, Q, R, S, T)

    invX0 = np.linalg.inv(X_of_t(0))
    x_closed = []
    u_closed = []
    for t in t_eval:
        Xt = X_of_t(t)
        xt = Xt @ invX0 @ x0
        lambda_t = Lambda_of_t(t) @ invX0 @ x0
        ut = -np.linalg.solve(R, B.T @ lambda_t + S.T @ xt)
        x_closed.append(xt)
        u_closed.append(ut)
    x_closed = np.array(x_closed)
    u_closed = np.array(u_closed)
    value_closed = [closed_value_function(t, x0, P_of_t) for t in t_eval]

    return x_closed, u_closed, value_closed
