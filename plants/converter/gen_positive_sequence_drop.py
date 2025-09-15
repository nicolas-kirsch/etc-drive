import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
f = 50
w = 2*np.pi*f
vg = 1.0
dt = 0.001
t_end = 1.5
t_drop_start = 0.4
t_drop_duration = 0.5
V1_target = 0.7  # target positive-sequence fraction

# --- Time vector ---
t = np.arange(0, t_end, dt)

# --- Pre-fault instantaneous voltages ---
theta = w * t
va = vg * np.cos(theta)
vb = vg * np.cos(theta - 2*np.pi/3)
vc = vg * np.cos(theta + 2*np.pi/3)

Va_phasor = vg * np.exp(1j*theta)
Vb_phasor = vg * np.exp(1j*(theta - 2*np.pi/3))
Vc_phasor = vg * np.exp(1j*(theta + 2*np.pi/3))

# --- Compute positive sequence ---
a = np.exp(1j * 2*np.pi/3)        # 120-degree operator

k = 2 * V1_target - 1  # derived analytically for two-phase drop, real < 1
print(k)
# --- Apply drop during event ---
start_idx = int(t_drop_start / dt)
end_idx = int((t_drop_start + t_drop_duration) / dt)
va[start_idx:end_idx] *= k
vb[start_idx:end_idx] *= k

Va_phasor[start_idx:end_idx] *= k
Vb_phasor[start_idx:end_idx] *= k
# vc remains unchanged

# Positive sequence
V1 = (Va_phasor + a*Vb_phasor + a**2*Vc_phasor)/3
V1_norm = np.abs(V1)

# --- Plot ---
plt.figure(figsize=(12,5))

# Time-domain waveforms
plt.subplot(2,1,1)
plt.plot(t, V1.real, label='Re{V1(t)}')
plt.plot(t, V1.imag, label='Im{V1(t)}')
plt.title('Positive Sequence Voltage V1(t)')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [p.u.]')
plt.grid(True)
plt.legend()

# Norm
plt.subplot(2,1,2)
plt.plot(t, V1_norm, label='|V1(t)|', color='r')
plt.title('Norm of Positive Sequence Voltage |V1(t)|')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [p.u.]')
plt.grid(True)
plt.legend()

# Norm
plt.figure(figsize=(12,5))
plt.plot(t, va, label='va')
plt.plot(t, vb, label='vb')
plt.plot(t, vc, label='vc')
plt.title('Phase Voltages va, vb, vc')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [p.u.]')

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
