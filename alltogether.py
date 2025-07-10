import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from KDEpy import FFTKDE
from scipy.signal import savgol_filter
from tqdm import tqdm
import hashlib
import json
import os

a, b, h = -3.0, 1.0, 1.0  # Landau potential parameters
gamma = 1.0            # friction
dt = 1/250             # time step
T = 10.0               # total simulation time (s)
N = int(T / dt)        # number of steps

@njit
def U(x):
    return 0.5 * a * x**2 + 0.25 * b * x**4 - h * x

@njit
def dU_dx(x):
    return a * x + b * x**3 - h

@njit
def simulate_langevin(N, dt, D, gamma, x0):
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        eta = np.sqrt(2 * D * dt) * np.random.normal()
        x[i] = x[i-1] - (1.0/gamma) * dU_dx(x[i-1]) * dt + eta
    return x

# Barrier Crossing 
def find_barrier(data, threshold, direction='forward'):
    for i in range(1, len(data)):
        if direction == 'forward' and data[i-1] < threshold <= data[i]:
            return i
        if direction == 'backward' and data[i-1] > threshold >= data[i]:
            return i
    return None

#KL Divergence / Markov Test
lag_steps = np.unique(np.round(np.logspace(0, 3.5, 60)).astype(int))
lag_times = lag_steps * dt
M = 50  # KDE grid size

def adjust_range(arr, pad=0.1, eps=1e-3):
    mn, mx = np.min(arr), np.max(arr)
    if mn == mx:
        return mn - eps, mx + eps
    pad_val = (mx - mn) * pad
    return mn - pad_val, mx + pad_val


def parzen_markov_test_fft(x, lag):
    X1 = x[:-2*lag:lag]
    X2 = x[lag:-lag:lag]
    X3 = x[2*lag::lag]
    if len(X1) < 2 or len(X2) < 2 or len(X3) < 2:
        return np.nan
    sigma_x = np.std(X1, ddof=1)
    sigma_y = np.std(X2, ddof=1)
    sigma = np.sqrt(sigma_x * sigma_y)
    n1 = len(X1)
    d = 2
    bw = sigma * (4.0/(d+2))**(1.0/(d+4)) * n1**(-1.0/(d+4))
    if bw <= 0:
        return np.nan
    r1 = adjust_range(X1); r2 = adjust_range(X2); r3 = adjust_range(X3)
    g1 = np.linspace(r1[0], r1[1], M)
    g2 = np.linspace(r2[0], r2[1], M)
    g3 = np.linspace(r3[0], r3[1], M)
    dx1, dx2, dx3 = g1[1]-g1[0], g2[1]-g2[0], g3[1]-g3[0]
    data12 = np.column_stack([X1, X2])
    kde12 = FFTKDE(bw=bw).fit(data12)
    G1, G2 = np.meshgrid(g1, g2, indexing='ij')
    p12 = kde12.evaluate(np.column_stack([G1.ravel(), G2.ravel()])).reshape(M, M)
    data23 = np.column_stack([X2, X3])
    kde23 = FFTKDE(bw=bw).fit(data23)
    G2b, G3 = np.meshgrid(g2, g3, indexing='ij')
    p23 = kde23.evaluate(np.column_stack([G2b.ravel(), G3.ravel()])).reshape(M, M)
    kde2 = FFTKDE(bw=bw).fit(X2.reshape(-1,1))
    p2 = kde2.evaluate(g2.reshape(-1,1))
    kde123 = FFTKDE(bw=bw).fit(np.column_stack([X1, X2, X3]))
    G1b, G2c, G3b = np.meshgrid(g1, g2, g3, indexing='ij')
    p123 = kde123.evaluate(np.column_stack([G1b.ravel(), G2c.ravel(), G3b.ravel()])).reshape(M,M,M)
    p12 = np.clip(p12,1e-10,None)/np.sum(p12)/(dx1*dx2)
    p23 = np.clip(p23,1e-10,None)/np.sum(p23)/(dx2*dx3)
    p2  = np.clip(p2,1e-10,None)/np.sum(p2)/dx2
    p123= np.clip(p123,1e-10,None)/np.sum(p123)/(dx1*dx2*dx3)
    p_markov = (p12[:,:,None]*p23[None,:,:])/p2[None,:,None]
    p_markov /= np.sum(p_markov)*dx1*dx2*dx3
    return np.sum(p123*np.log(p123/p_markov))*dx1*dx2*dx3

# Smoothing + Tau detection
def smooth_and_find_tau(lags, kl_vals):
    kl_smooth = savgol_filter(kl_vals,11,3)
    for i in range(1,len(kl_smooth)-1):
        if kl_smooth[i]<kl_smooth[i-1] and kl_smooth[i]<kl_smooth[i+1]:
            return lag_times[i]
    return np.nan

# Caching
cache_dir='cache_kl'
if not os.path.exists(cache_dir): os.makedirs(cache_dir)
def run_kl_for_D(D):
    key=hashlib.md5(json.dumps({'D':D}).encode()).hexdigest()
    cf=os.path.join(cache_dir,f'kl_{key}.npz')
    if os.path.exists(cf): return np.load(cf)['avg_kl']
    kl_runs=[]
    for _ in tqdm(range(30),desc=f'KL D={D}'):
        x0=np.random.choice([-1.5,1.5]); x=simulate_langevin(N,dt,D,gamma,x0)
        kl_runs.append([parzen_markov_test_fft(x,lag) for lag in lag_steps])
    avg_kl=np.nanmean(kl_runs,axis=0); np.savez(cf,avg_kl=avg_kl)
    return avg_kl

# main
def main():
    D_values=[0.5,1.0,2.0,4.0]
    colors=plt.cm.tab10(range(len(D_values)))
    # prepare MFPT & hist data
    x_vals = np.linspace(-4,4,1000)
    # find exact minima and barrier by solving dU_dx=0
    coeffs = [b, 0.0, a, -h]  # b x^3 + a x - h = 0
    roots = np.roots(coeffs)
    real_roots = np.sort(np.real(roots[np.isreal(roots)]))
    if len(real_roots) >= 3:
        minima = [real_roots[0], real_roots[2]]
        barrier = real_roots[1]
    else:
        minima = [real_roots[0], real_roots[-1]]
        barrier = np.mean(real_roots)

    mfpt_fwd, mfpt_bwd, std_fwd, std_bwd = [], [], [], []
    hist_data = {'forward': {}, 'backward': {}}
    for D in D_values:
        f_times, b_times = [], []
        for _ in range(500):
            x0 = np.random.choice(minima)
            traj = simulate_langevin(N, dt, D, gamma, x0)
            if x0 < barrier:
                idx = find_barrier(traj, barrier, 'forward')
                if idx is not None:
                    f_times.append(idx * dt)
            else:
                idx = find_barrier(traj, barrier, 'backward')
                if idx is not None:
                    b_times.append(idx * dt)
        # ensure non-empty lists
        if len(f_times) > 0:
            mfpt_fwd.append(np.mean(f_times))
            std_fwd.append(np.std(f_times))
        else:
            mfpt_fwd.append(0)
            std_fwd.append(0)
        if len(b_times) > 0:
            mfpt_bwd.append(np.mean(b_times))
            std_bwd.append(np.std(b_times))
        else:
            mfpt_bwd.append(0)
            std_bwd.append(0)
        hist_data['forward'][D] = f_times
        hist_data['backward'][D] = b_times

    # 2x2 layout
    fig, axes = plt.subplots(2,2,figsize=(12,12), constrained_layout=True)

    # Top-left: Potential with minima and barrier
    axes[0,0].plot(x_vals, U(x_vals), color='black')
    axes[0,0].axvline(minima[0], color='blue', linestyle='--', label='Minima')
    axes[0,0].axvline(minima[1], color='blue', linestyle='--')
    axes[0,0].axvline(barrier, color='red', linestyle='-', label='Barrier')
    axes[0,0].scatter(minima, U(np.array(minima)), color='blue')
    axes[0,0].scatter([barrier], [U(barrier)], color='red')
    axes[0,0].set_title('Asymmetric Double-Well Potential')
    axes[0,0].set_xlabel('x'); axes[0,0].set_ylabel('U(x)'); axes[0,0].grid(True)
    axes[0,0].legend()

    # Top-right: MFPT vs D
    axes[0,1].errorbar(D_values, mfpt_fwd, yerr=std_fwd, fmt='o-', color='blue', label='Forward MFPT')
    axes[0,1].errorbar(D_values, mfpt_bwd, yerr=std_bwd, fmt='s-', color='red', label='Backward MFPT')
    axes[0,1].set_xscale('log'); axes[0,1].set_yscale('log')
    axes[0,1].set_title('MFPT vs D'); axes[0,1].set_xlabel('D'); axes[0,1].set_ylabel('MFPT (s)')
    axes[0,1].legend(); axes[0,1].grid(True)

    # Bottom-left: MFPT Distributions
    for D,col in zip(D_values,colors):
        axes[1,0].hist(hist_data['forward'][D], bins=20, density=True, alpha=0.4, edgecolor=col, facecolor=col, label=f'Fwd D={D}')
        axes[1,0].hist(hist_data['backward'][D], bins=20, density=True, alpha=0.2, edgecolor=col, facecolor=col, hatch='//', label=f'Bwd D={D}')
    axes[1,0].set_title('MFPT Distributions'); axes[1,0].set_xlabel('Time (s)'); axes[1,0].set_ylabel('Density')
    axes[1,0].legend(); axes[1,0].grid(True)

    # Bottom-right: KL vs τ
    for D,col in zip(D_values,colors):
        avg_kl = run_kl_for_D(D)
        tau = smooth_and_find_tau(lag_times, avg_kl)
        axes[1,1].loglog(lag_times, avg_kl, color=col, label=f'D={D}')
        if not np.isnan(tau): axes[1,1].axvline(x=tau, color=col, linestyle='--')
    axes[1,1].set_title('KL vs τ with τₘ'); axes[1,1].set_xlabel('τ (s)'); axes[1,1].set_ylabel('KL Divergence')
    axes[1,1].legend(); axes[1,1].grid(True)

    plt.show()

if __name__=='__main__':
    main()
