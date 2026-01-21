#!/usr/bin/env python3
"""
frt_switching_euler.py

FRT switching simulation using Forward Euler integration.

Fault (switch closes to ground): t = 0.5 s
Fault cleared (switch opens):   t = 0.9 s
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
V_m = 3000.0         # Source amplitude (V) phase-to-ground
f = 50.0             # Hz
omega = 2*np.pi*f

L_s = 1.0e-3         # H
R_s = 0.2            # Ohm
R_L = 50.0           # Ohm
C_p = 200e-9         # F

R_sw_closed = 0.01   # Ohm
R_sw_open   = 1e9    # Ohm

t_fault = 0.5        # s
t_clear = 0.9        # s

t_start = 0.0
t_stop  = 1.5
dt = 1e-6            # step size (1 µs)
N = int((t_stop - t_start) / dt)

# -------------------------
# Helper functions
# -------------------------
def v_source(t):
    return V_m * np.sin(omega * t)

def R_sw_time(t):
    if t_fault <= t < t_clear:
        return R_sw_closed
    else:
        return R_sw_open

# -------------------------
# Initialization
# -------------------------
t = np.linspace(t_start, t_stop, N)
i = np.zeros(N)      # inductor current
v_n = np.zeros(N)    # node voltage

# -------------------------
# Forward Euler loop
# -------------------------
for k in range(N-1):
    Rsw = R_sw_time(t[k])
    v_s = v_source(t[k])

    # di/dt
    di_dt = (v_s - R_s * i[k] - v_n[k]) / L_s
    # dvn/dt
    dvn_dt = (i[k] - v_n[k]*(1/R_L + 1/Rsw)) / C_p

    # Euler updates
    i[k+1] = i[k] + dt * di_dt
    v_n[k+1] = v_n[k] + dt * dvn_dt

# -------------------------
# Derived signals
# -------------------------
v_s_all = v_source(t)
R_sw_profile = np.array([R_sw_time(tt) for tt in t])
i_sw = v_n / R_sw_profile
i_load = v_n / R_L

# -------------------------
# Plot results
# -------------------------
plt.figure(figsize=(12,8))

ax1 = plt.subplot(3,1,1)
plt.plot(t, v_s_all, '--', label='v_s (source)')
plt.plot(t, v_n, label='v_n (node)')
plt.axvspan(t_fault, t_clear, color='red', alpha=0.1, label='fault interval')
plt.ylabel('Voltage (V)')
plt.legend(); plt.grid(True)

ax2 = plt.subplot(3,1,2, sharex=ax1)
plt.plot(t, i, label='i (line current)')
plt.plot(t, i_load, label='i_load')
plt.plot(t, i_sw, label='i_sw')
plt.axvline(t_fault, color='k', linestyle=':')
plt.axvline(t_clear, color='k', linestyle=':')
plt.ylabel('Current (A)')
plt.legend(); plt.grid(True)

ax3 = plt.subplot(3,1,3, sharex=ax1)
plt.semilogy(t, R_sw_profile, label='R_sw(t)')
plt.ylabel('Switch resistance (Ω)')
plt.xlabel('Time (s)')
plt.grid(True, which='both')
plt.legend()

plt.suptitle('FRT Switching Simulation (Forward Euler)')
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()
