# diode_rectifier_dc_link_euler.py
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
f = 50.0
w = 2*np.pi*f
Vll_rms = 3700.0
Vph_peak = Vll_rms/np.sqrt(3) * np.sqrt(2)  # phase peak

Rg, Lg = 0.05, 1e-3              # grid R, L
Cdc = 2e-3                       # DC link capacitance
Rload = 50.0                     # DC load resistance

h = 2.5e-4                       # step size [s]
Tsim = 1                       # simulate 100 ms
N = int(Tsim/h)

# -------------------------
# Helper
# -------------------------
def grid_voltages(t):
    va = Vph_peak * np.cos(w*t)
    vb = Vph_peak * np.cos(w*t - 2*np.pi/3)
    vc = Vph_peak * np.cos(w*t + 2*np.pi/3)
    return np.array([va, vb, vc])

# -------------------------
# States
# -------------------------
ia, ib, ic, vdc = 0.0, 0.0, 0.0, 0.0

time = np.zeros(N)
log_vdc = np.zeros(N)

# -------------------------
# Forward Euler simulation
# -------------------------
for k in range(N):
    t = k*h
    time[k] = t

    # Grid voltages
    vg = grid_voltages(t)
    iabc = np.array([ia, ib, ic])

    # Decide conduction: max/min of grid voltages
    idx_max = np.argmax(vg)
    idx_min = np.argmin(vg)

    vr = np.zeros(3)
    vr[idx_max] = +0.5*vdc
    vr[idx_min] = -0.5*vdc

    # Grid current dynamics: di/dt = (vg - Rg*i - vr)/Lg
    di = (vg - Rg*iabc - vr)/Lg
    iabc = iabc + h*di

    # DC current: i_k - i_m
    idc = iabc[idx_max] - iabc[idx_min]

    # DC link dynamics: dv/dt = (-vdc/Rload + idc)/Cdc
    dv = (-vdc/Rload + idc)/Cdc
    vdc = vdc + h*dv

    # store
    ia, ib, ic = iabc
    log_vdc[k] = vdc

# -------------------------
# Plot results
# -------------------------
plt.figure(figsize=(8,4))
plt.plot(time, log_vdc, label="Vdc (Euler h=2.5e-4)")
plt.xlabel("Time [s]")
plt.ylabel("DC Link Voltage [V]")
plt.title("Diode Rectifier DC Link Voltage (Forward Euler)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
