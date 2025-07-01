#!/usr/bin/env python3
"""
kl_vs_D_physical_tau_fixed.py

Simulación Langevin + divergencia KL vs τ para distintos valores de D.
Incluye dos métodos para estimar el τ_M físico:
  1) Suavizado + primer mínimo local en la curva suavizada de KL.
  2) Umbral relativo pequeño sobre el mínimo global de KL.
Mejoras: cálculo de anchos de banda adaptados a cada dimensión (1D, 2D, 3D).
"""
import os
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from KDEpy import FFTKDE
from numba import njit
from scipy.signal import savgol_filter
import csv

# Parámetros fijos
dt = 1/250
T = 10
N = int(T / dt)
gamma = 1.0
n_runs = 30   # réplicas para suavizar
a, b, h = -3.0, 1.0, 1.0

# Potencial y derivada
@njit
def U(x):
    return 0.5 * a * x**2 + 0.25 * b * x**4 - h * x

@njit
def dU_dx(x):
    return a * x + b * x**3 - h

# Integrador Euler-Maruyama
def simulate_langevin(N, dt, D, gamma, x0):
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        eta = np.sqrt(2 * D * dt) * np.random.normal()
        x[i] = x[i-1] - (1.0/gamma) * dU_dx(x[i-1]) * dt + eta
    return x

# Rango para KDE
def adjust_range(arr, eps=1e-3, pad=0.1):
    mn, mx = np.min(arr), np.max(arr)
    if mn == mx:
        return mn - eps, mx + eps
    padding = (mx - mn) * pad
    return mn - padding, mx + padding

# Prueba de Markov con divergencia KL usando KDEpy y anchos de banda por dimensión
def parzen_markov_test_fft(x, lag, M=50):
    X1 = x[:-2*lag:lag]
    X2 = x[lag:-lag:lag]
    X3 = x[2*lag::lag]

    if len(X1) < 2 or len(X2) < 2 or len(X3) < 2:
        return np.nan

    # Datos conjuntos
    data12 = np.column_stack([X1, X2])
    data23 = np.column_stack([X2, X3])
    data2  = X2.reshape(-1,1)
    data123 = np.column_stack([X1, X2, X3])

    #sigma por eje
    sigma12 = np.mean(np.std(data12, axis=0, ddof=1))
    sigma23 = np.mean(np.std(data23, axis=0, ddof=1))
    sigma2  = np.std(X2, ddof=1)
    sigma123= np.mean(np.std(data123, axis=0, ddof=1))

    # Número de muestras
    n12 = len(X1)
    n23 = len(X2)
    n2  = len(X2)
    n123= len(X1)

    # Anchos de banda según dimensión d
    # d=2 para p12 y p23, d=1 para p2, d=3 para p123
    bw12  = (4/(2+2))**(1/6) * sigma12  * n12**(-1/6)
    bw23  = (4/(2+2))**(1/6) * sigma23  * n23**(-1/6)
    bw2   = (4/(1+2))**(1/5) * sigma2   * n2 **(-1/5)
    bw123 = (4/(3+2))**(1/7) * sigma123* n123**(-1/7)

    # Mallas y volúmenes de celda
    r1 = adjust_range(X1)
    r2 = adjust_range(X2)
    r3 = adjust_range(X3)
    g1 = np.linspace(r1[0], r1[1], M)
    g2 = np.linspace(r2[0], r2[1], M)
    g3 = np.linspace(r3[0], r3[1], M)
    dx1, dx2, dx3 = g1[1]-g1[0], g2[1]-g2[0], g3[1]-g3[0]

    # KDE 2D p12
    kde12 = FFTKDE(bw=bw12).fit(data12)
    pts12 = np.column_stack([m.ravel() for m in np.meshgrid(g1,g2,indexing='ij')])
    p12 = kde12.evaluate(pts12).reshape(M,M)
    p12 = np.clip(p12,1e-10,None)
    p12 /= np.sum(p12)*dx1*dx2

    # KDE 2D p23
    kde23 = FFTKDE(bw=bw23).fit(data23)
    pts23 = np.column_stack([m.ravel() for m in np.meshgrid(g2,g3,indexing='ij')])
    p23 = kde23.evaluate(pts23).reshape(M,M)
    p23 = np.clip(p23,1e-10,None)
    p23 /= np.sum(p23)*dx2*dx3

    # KDE 1D p2
    kde2 = FFTKDE(bw=bw2).fit(data2)
    p2 = kde2.evaluate(g2.reshape(-1,1))
    p2 = np.clip(p2,1e-10,None)
    p2 /= np.sum(p2)*dx2

    # KDE 3D p123
    kde123 = FFTKDE(bw=bw123).fit(data123)
    pts123 = np.column_stack([m.ravel() for m in np.meshgrid(g1,g2,g3,indexing='ij')])
    p123 = kde123.evaluate(pts123).reshape(M,M,M)
    p123 = np.clip(p123,1e-10,None)
    p123 /= np.sum(p123)*dx1*dx2*dx3

    # Factorización markoviana y divergencia KL
    pcond = p23 / p2[:,np.newaxis]
    p_markov = np.einsum('ij,jk->ijk', p12, pcond)
    p_markov /= np.sum(p_markov)*dx1*dx2*dx3

    kl = np.sum(p123 * np.log(p123 / p_markov)) * dx1 * dx2 * dx3
    return kl

# Precomputa retardos
lag_steps = np.unique(np.round(np.logspace(0,3.5,60)).astype(int))
lag_times = lag_steps * dt


# MÉTODO 1: Suavizado + primer mínimo local

def smooth_kl_curve(kl_vals, window_length=11, polyorder=3):
    if len(kl_vals) < window_length:
        return kl_vals.copy()
    return savgol_filter(kl_vals, window_length, polyorder)

def first_local_minimum(lag_times, kl_smooth):
    """Busca el primer i que kl_smooth[i] < kl_smooth[i-1] y kl_smooth[i] < kl_smooth[i+1]. Devuelve lag_times[i] o np.nan si no hay tal punto."""
    for i in range(1, len(kl_smooth) - 1):
        if kl_smooth[i-1] > kl_smooth[i] < kl_smooth[i+1]:
            return lag_times[i]
    return np.nan


# MÉTODO 2: Umbral relativo pequeño sobre el mínimo global
def tau_by_small_relative(lag_times, kl_vals, delta=0.01):
    """Encuentra el primer tau tal que KL(tau) <= (1 + delta)*min(KL)."""
    kl_arr = np.array(kl_vals)
    if np.all(np.isnan(kl_arr)):
        return np.nan
    kl_min = np.nanmin(kl_arr)
    threshold = (1 + delta) * kl_min
    idx = np.where(kl_arr <= threshold)[0]
    return lag_times[idx[0]] if idx.size > 0 else np.nan

# Función principal: simula y devuelve KL_promedio para cada D
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
        x0 = np.random.normal(-2 if np.random.rand() < 0.5 else 2, 0.1)
        x = simulate_langevin(N, dt, D_val, gamma, x0)
        kl_vals = [parzen_markov_test_fft(x, l) for l in lag_steps]
        kl_runs.append(kl_vals)

    kl_runs = [run for run in kl_runs if not np.all(np.isnan(run))]
    if len(kl_runs) == 0:
        avg_kl = np.full_like(lag_times, np.nan)
    else:
        avg_kl = np.nanmean(kl_runs, axis=0)

    np.savez(cache_file, avg_kl=avg_kl)
    return lag_times, avg_kl

# BUCLE PRINCIPAL
if __name__ == "__main__":
    D_list = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0, 8.0, 10.0] # Lista de coeficientes de difusión

    results = {}
    for D_val in D_list:
        tau_arr, avg_kl = run_simulation_for_D(D_val)
        results[D_val] = avg_kl

    # Calcular tau_M por cada método
    tau_M_smooth = []
    tau_M_relative = []
    for D_val in D_list:
        avg_kl = results[D_val]

        # Método 1: Suavizado + primer mínimo local
        kl_smooth = smooth_kl_curve(avg_kl, window_length=11, polyorder=3)
        tau1 = first_local_minimum(lag_times, kl_smooth)
        tau_M_smooth.append(tau1)

        # Método 2: Umbral relativo pequeño (delta = 0.01)
        tau2 = tau_by_small_relative(lag_times, avg_kl, delta=0.01)
        tau_M_relative.append(tau2)

    # resultados
    print("=== tau_M (smooth + local min) ===")
    for D, tauM in zip(D_list, tau_M_smooth):
        print(f"D = {D:.3f}, τ_M = {tauM}")

    print("\n=== tau_M (small relative threshold) ===")
    for D, tauM in zip(D_list, tau_M_relative):
        print(f"D = {D:.3f}, τ_M = {tauM}")

    # Graficar KL vs τ con líneas verticales para ambos criterios
    plt.figure(figsize=(10, 6))
    for i, D_val in enumerate(D_list):
        avg_kl = results[D_val]
        color = plt.cm.tab10(i % 10)

        min_len = min(len(lag_times), len(avg_kl))
        plt.plot(lag_times[:min_len], avg_kl[:min_len], label=f"D={D_val}", color=color)

        t1 = tau_M_smooth[i]
        if np.isfinite(t1) and t1 > 0:
            plt.axvline(x=t1, color=color, linestyle="--", alpha=0.7)

        t2 = tau_M_relative[i]
        if np.isfinite(t2) and t2 > 0:
            plt.axvline(x=t2, color=color, linestyle="-.", alpha=0.7)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\tau$ (s)")
    plt.ylabel("KL Divergence")
    plt.title("KL vs τ (solid=avg, dashed=τ_M smooth, dashdot=τ_M relative)")
    plt.legend(fontsize="small", ncol=2, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig("KL_vs_tau_with_two_tauM_methods.svg", dpi=300)
    plt.show()

    # Graficar τ_M vs D para cada criterio
    plt.figure(figsize=(8, 5))
    plt.plot(D_list, tau_M_smooth, "o-", label="tau_M (smooth + local min)")
    plt.plot(D_list, tau_M_relative, "s--", label="tau_M (small relative)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Diffusion coefficient $D$")
    plt.ylabel(r"Effective memory time $\tau_M$ (s)")
    plt.title(r"$\tau_M(D)$ Comparison")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig("tauM_vs_D_comparison.svg", dpi=300)
    plt.show()

    # Exportar resultados a CSV
    with open("tauM_vs_D_physical.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["D", "tauM_smooth", "tauM_relative"])
        writer.writerows(zip(D_list, tau_M_smooth, tau_M_relative))
