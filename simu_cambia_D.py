import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numba import njit
from tqdm import tqdm
#ESTE CODIGO ME GENERA LOS HISTOGRAMAS DE CRUCE HACIA ADELANTE Y HACIA ATRAS, ASI COMO LOS TIEMPOS MEDIOS DE CRUCE (MFPT) PARA DIFERENTES COEFICIENTES DE DIFUSION D
# param generales
dt = 1/250           # Paso de tiempo (s)
T = 10               # Tiempo total (s)
N = int(T / dt)      # pasos
gamma = 1            # friccion

# Parametros del potencial de Landau
a, b, h = -3.0, 1.0, 1.0

# FUNCIONES DEL POTENCIAL Y DINAMICA 
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

# DETECTAR CRUCES DE BARRERA
def find_barrier_crossings(x_traj, threshold, direction='forward'):
    crossings = []
    if direction == 'forward':
        for i in range(1, len(x_traj)):
            if x_traj[i-1] < threshold and x_traj[i] >= threshold:
                crossings.append(i)
    elif direction == 'backward':
        for i in range(1, len(x_traj)):
            if x_traj[i-1] > threshold and x_traj[i] <= threshold:
                crossings.append(i)
    return crossings

# BUCLE PRINCIPAL
def main():
    sns.set(style="whitegrid")

    D_values = [0.5, 1.0, 2.0, 4.0]  # Coeficientes de difusion
    palette = sns.color_palette("muted", len(D_values))
    n_simulations = 1000

    # minimos y barrera
    x_vals = np.linspace(-4, 4, 1000)
    dpot = dU_dx(x_vals)
    mask = np.abs(dpot) < 0.05
    critical_pts = x_vals[mask]
    if len(critical_pts) >= 3:
        minima = np.sort([critical_pts[0], critical_pts[-1]])
        barrier = critical_pts[len(critical_pts)//2]
    else:
        minima = np.array([-1.5, 1.5])
        barrier = 0.0

    print(f"Minima at: {minima}, Barrier at: {barrier:.3f}")

    mfpt_forward_all = []
    mfpt_backward_all = []
    std_forward_all = []
    std_backward_all = []
    forward_times_dict = {}
    backward_times_dict = {}

    for D in D_values:
        print(f"\nSimulating for D = {D}...")
        forward_times = []
        backward_times = []

        for _ in tqdm(range(n_simulations)):
            x0 = np.random.choice([minima[0], minima[1]])
            traj = simulate_langevin(N, dt, D, gamma, x0)
            if x0 < barrier:
                crossings = find_barrier_crossings(traj, barrier, direction='forward')
                if crossings:
                    forward_times.append(crossings[0] * dt)
            else:
                crossings = find_barrier_crossings(traj, barrier, direction='backward')
                if crossings:
                    backward_times.append(crossings[0] * dt)

        forward_times = np.array(forward_times)
        backward_times = np.array(backward_times)

        print(f"  Forward crossings for D={D}: {len(forward_times)} events")
        print(f"  Backward crossings for D={D}: {len(backward_times)} events")

        mfpt_forward_all.append(np.mean(forward_times))
        mfpt_backward_all.append(np.mean(backward_times))
        std_forward_all.append(np.std(forward_times))
        std_backward_all.append(np.std(backward_times))

        forward_times_dict[D] = forward_times
        backward_times_dict[D] = backward_times

    #histo 
    plt.figure(figsize=(10, 6))
    for D, color in zip(D_values, palette):
        sns.histplot(forward_times_dict[D], kde=False, bins=20, stat="density", label=f'Forward D={D}', color=color, element="step", fill=True, alpha=0.4)
    plt.title("Forward Barrier Crossing Time Distributions", fontsize=18)
    plt.xlabel("Crossing Time (s)", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("forward_crossing_times.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))
    for D, color in zip(D_values, palette):
        sns.histplot(backward_times_dict[D], kde=False, bins=20, stat="density", label=f'Backward D={D}', color=color, element="step", fill=True, alpha=0.4)
    plt.title("Backward Barrier Crossing Time Distributions", fontsize=18)
    plt.xlabel("Crossing Time (s)", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("backward_crossing_times.png", dpi=300)
    plt.show()

    #MFPT VS D
    plt.figure(figsize=(8, 6))
    plt.errorbar(D_values, mfpt_forward_all, yerr=std_forward_all, fmt='o-', color='blue', label='Forward MFPT')
    plt.errorbar(D_values, mfpt_backward_all, yerr=std_backward_all, fmt='s-', color='red', label='Backward MFPT')
    plt.xlabel("Diffusion Coefficient D", fontsize=14)
    plt.ylabel("MFPT (s)", fontsize=14)
    plt.title("MFPT vs Diffusion Coefficient", fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mfpt_vs_D.png", dpi=300)
    plt.show()

    # TABLA
    summary_df = pd.DataFrame({
        "D": D_values,
        "Forward MFPT (s)": mfpt_forward_all,
        "Backward MFPT (s)": mfpt_backward_all,
        "Std Forward (s)": std_forward_all,
        "Std Backward (s)": std_backward_all
    })

    print("\nSUMMARY OF BARRIER CROSSING EVENTS:")
    print(summary_df)
    summary_df.to_csv("mfpt_summary.csv", index=False)
    print("\nSummary table saved as 'mfpt_summary.csv'")

if __name__ == "__main__":
    main()
