#This script performs Langevin dynamics simulations for different diffusion coefficients using a KL divergence threshold of 0.03.  
#It computes and plots KL divergence vs. lag time and extracts the effective memory time τ_M when the divergence falls below the threshold.

#SIMULACIONES CON THRESHOLD DE 0.03
import os
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from KDEpy import FFTKDE
from numba import njit

# Parámetros fijos
dt = 1/250
T = 10
N = int(T/dt)
gamma = 1.0
n_runs = 50
a, b, h = -3.0, 1.0, 1.0

#potencial y la derivada
@njit
def U(x):
    return 0.5 * a * x**2 + 0.25 * b * x**4 - h * x

@njit
def dU_dx(x):
    return a * x + b * x**3 - h

# Euler-Maruyama
@njit
def simulate_langevin(N, dt, D, gamma, x0):
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        eta = np.sqrt(2 * D * dt) * np.random.normal()
        x[i] = x[i-1] - (1.0/gamma) * dU_dx(x[i-1]) * dt + eta
    return x

# Ajuste de rango para KDE con numba
@njit
def adjust_range_numba(arr, eps=1e-3, pad=0.1):
    mn = np.min(arr)
    mx = np.max(arr)
    if mn == mx:
        return (mn - eps, mx + eps)
    padding = (mx - mn) * pad
    return (mn - padding, mx + padding)

# Test de Markov con divergencia KL 
def parzen_markov_test_fft(x, lag, M=50):
    # Construye series rezagadas
    X1 = x[:-2*lag:lag]
    X2 = x[lag:-lag:lag]
    X3 = x[2*lag::lag]

    #quita tamaños demasiado pequeños
    if len(X1) < 2 or len(X2) < 2 or len(X3) < 2:
        return np.nan

    # Bandwidth de Silverman
    data12 = np.column_stack([X1, X2])
    n1 = len(X1)
    sigma = np.mean(np.std(data12, axis=0, ddof=1))
    if sigma <= 0:
        return np.nan
    bw = (4/(2+2))**(1/6) * sigma * n1**(-1/6)
    bw12 = bw23 = bw2 = bw123 = bw




    # Mallas
    r1 = adjust_range_numba(X1)
    r2 = adjust_range_numba(X2)
    r3 = adjust_range_numba(X3)
    g1 = np.linspace(r1[0], r1[1], M)
    g2 = np.linspace(r2[0], r2[1], M)
    g3 = np.linspace(r3[0], r3[1], M)
    dx1, dx2, dx3 = g1[1]-g1[0], g2[1]-g2[0], g3[1]-g3[0]

    try:
        # KDE de pares (1,2)
        kde12 = FFTKDE(bw=bw12).fit(data12)
        pts12 = np.column_stack([m.ravel() for m in np.meshgrid(g1, g2, indexing='ij')])
        p12 = kde12.evaluate(pts12).reshape(M, M)
        p12 = np.clip(p12, 1e-10, None)
        p12 /= np.sum(p12) * dx1 * dx2

        # KDE de pares (2,3)
        data23 = np.column_stack([X2, X3])
        kde23 = FFTKDE(bw=bw23).fit(data23)
        pts23 = np.column_stack([m.ravel() for m in np.meshgrid(g2, g3, indexing='ij')])
        p23 = kde23.evaluate(pts23).reshape(M, M)
        p23 = np.clip(p23, 1e-10, None)
        p23 /= np.sum(p23) * dx2 * dx3

        # KDE marginal de X2
        kde2 = FFTKDE(bw=bw2).fit(X2.reshape(-1,1))
        p2 = kde2.evaluate(g2.reshape(-1,1))
        p2 = np.clip(p2, 1e-10, None)
        p2 /= np.sum(p2) * dx2

        # KDE tripleta (1,2,3)
        data123 = np.column_stack([X1, X2, X3])
        kde123 = FFTKDE(bw=bw123).fit(data123)
        pts123 = np.column_stack([m.ravel() for m in np.meshgrid(g1, g2, g3, indexing='ij')])
        p123 = kde123.evaluate(pts123).reshape(M, M, M)
        p123 = np.clip(p123, 1e-10, None)
        p123 /= np.sum(p123) * dx1 * dx2 * dx3

        # distribución markoviana ideal
        pcond = p23 / p2[:, np.newaxis]
        p_markov = np.einsum('ij,jk->ijk', p12, pcond)
        p_markov /= np.sum(p_markov) * dx1 * dx2 * dx3

        # Divergencia KL
        kl = np.sum(p123 * np.log(p123 / p_markov)) * dx1 * dx2 * dx3
        return kl
    except Exception:
        return np.nan

lag_steps = np.unique(np.round(np.logspace(0, 3, 50)).astype(int))
lag_times = lag_steps * dt

# Función principal para un valor de D con cache
def run_simulation_for_D(D_val):
    params = {"dt": dt, "T": T, "N": N, "D": D_val,
              "gamma": gamma, "n_runs": n_runs, "a": a, "b": b, "h": h}
    h_val = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    cache_file = f"cache_kl_{h_val}.npz"
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return lag_times, data["avg_kl"]

    kl_runs = []
    for _ in tqdm(range(n_runs), desc=f"D={D_val}"):
        x0 = np.random.normal(-2 if np.random.rand()<0.5 else 2, 0.1)
        x = simulate_langevin(N, dt, D_val, gamma, x0)
        kl_vals = [parzen_markov_test_fft(x, l) for l in lag_steps]
        kl_runs.append(kl_vals)

    avg_kl = np.nanmean(kl_runs, axis=0)
    np.savez(cache_file, avg_kl=avg_kl)
    return lag_times, avg_kl

# Ejecución
if __name__ == "__main__":
    D_list = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0]

    results = {}
    for D_val in D_list:
        tau, avg_kl = run_simulation_for_D(D_val)
        results[D_val] = avg_kl

    #curvas de KL vs tau
    plt.figure(figsize=(8, 6))
    for D_val, avg_kl in results.items():
        plt.plot(lag_times, avg_kl, label=f"D = {D_val}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\tau$ (s)")
    plt.ylabel("KL Divergence")
    plt.title("KL vs tau for several D")
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig("KL_vs_tau_vs_D_100runs.svg", dpi=300)
    plt.show()

    # tau_M(D) con threshold observado
    epsilon = 0.03
    tau_M_list = []
    for D_val in D_list:
        avg_kl = results[D_val]
        idx = np.where(avg_kl < epsilon)[0]
        if idx.size > 0:
            tau_M_list.append(lag_times[idx[0]])
        else:
            tau_M_list.append(np.nan)

    #tau_M vs D 
    plt.figure(figsize=(6,4))
    plt.plot(D_list, tau_M_list, 'o-', markersize=6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Diffusion coefficient $D$")
    plt.ylabel(r"Effective Markov time $\tau_M$ (s)")
    plt.title(r"Characteristic memory time $\tau_M(D)$ (threshold $\mathcal{D}_{KL}<%.2f$)"%epsilon)
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig("tauM_vs_D.svg", dpi=300)
    plt.show()
