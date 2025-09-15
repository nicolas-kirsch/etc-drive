# vn_simulation_0_5s_decay.py
import numpy as np
import matplotlib.pyplot as plt
import time as timo

# --------------------------
# Simulation parameters
# --------------------------
fs = 10000              # Sampling frequency (Hz)
T = 1.0                 # Total simulation time (s)
dt = 1/fs               # Time step
time = np.arange(0, T, dt)

# --------------------------
# Source voltage (AC)
# --------------------------
V_m = 1.0               # Source amplitude (Volts)
f = 50                  # Frequency (Hz)
v_s = V_m * np.sin(2 * np.pi * f * time)

# --------------------------
# Line and switch parameters
# --------------------------
L_s = 0.001        # H
R_s = 0.1             # Ohm
R_sw = 0.0           # Ohm
R_L = 10.0           # Load resistance (Ohm)

# --------------------------
# Switch closing time (fault)
# --------------------------
t_fault = 0.247           # Switch closes at 0.5 s
k_fault = int(t_fault / dt)
k_end_fault = k_fault + int(0.3 / dt)  # Fault lasts for 0.05 s
# --------------------------
# Initialization
# --------------------------
v_n = np.zeros_like(time)    # Node voltage
i_line = np.zeros_like(time) # Line current
V_L = np.zeros_like(time)    # Voltage at point P

# Initial conditions
v_n[0] = 0.0
i_line[0] = 0.0

# --------------------------
# Simulation loop (forward Euler)
# --------------------------
for k in range(len(time)-1):
    # Switch state: open before fault, closed after
    if k < k_fault:
        v_n[k+1] = v_s[k+1]  # effectively open
    elif k > k_end_fault:     
        dvn_dt = ( - (R_s + R_L)*v_n[k] + R_L * v_s[k] ) / L_s
        v_n[k+1] = v_n[k] + dt * dvn_dt
 
    else:
        # Line current derivative
        di_line_dt = (v_s[k] - v_n[k] - R_s * i_line[k]) / L_s
        # Update line current
        i_line[k+1] = i_line[k] + dt * di_line_dt
        # Node voltage from switch Ohm's law
        v_n[k+1] = v_n[k]*(1-dt*(R_s+R_sw)/L_s) + dt * R_sw/L_s * v_s[k] 
        # Voltage at point P (ideal line)

    
    V_L[k+1] = v_n[k+1]

# --------------------------
# Plot results
# --------------------------
plt.figure(figsize=(10,5))
plt.plot(time, v_s, label='Source v_s(t)', linestyle='--')
plt.plot(time, V_L, label='Voltage at P / Node v_n(t)')
plt.axvline(t_fault, color='r', linestyle=':', label='Switch closes')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Node Voltage and Voltage at Point P with Slow Decay')
plt.grid(True)
plt.legend()
plt.show()
