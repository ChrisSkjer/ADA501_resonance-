import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Systemparametere
m = 1.0      # Masse [kg]
c = 2.0      # Demping [Ns/m]
k = 1000.0   # Fjærstivhet [N/m]
Fy = 100.0   # Flytegrense [N]
F0 = 110.0   # Amplitude ytre kraft
omega = 2.0  # Frekvens ytre kraft

# Fjærkraft: elastisk-plastisk
def spring_force(x):
    Fx = k * x
    if abs(Fx) <= Fy:
        return Fx
    else:
        return Fy * np.sign(x)

# Ytre kraft: kort puls
def external_force(t):
    return F0 if 0 <= t <= 0.2 else 0.0

# ODE-system
def system(t, y):
    x, v = y
    dxdt = v
    Fs = spring_force(x)
    dvdt = (external_force(t) - c*v - Fs) / m
    return [dxdt, dvdt]

# Løsning
y0 = [0.0, 0.0]
t_span = (0, 10)
t_eval = np.linspace(*t_span, 2000)
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Beregn fjærkraft for hele x(t)
x_vals = sol.y[0]
Fs_vals = np.array([spring_force(xi) for xi in x_vals])

# Plot forskyvning
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(sol.t, x_vals, label="x(t) – Forskyvning")
plt.axhline(Fy / k, color='gray', linestyle='--', label="Flytegrense (x_y)")
plt.axhline(-Fy / k, color='gray', linestyle='--')
plt.ylabel("Forskyvning [m]")
plt.legend()
plt.grid()

# Plot fjærkraft
plt.subplot(2, 1, 2)
plt.plot(sol.t, Fs_vals, label="Fjærkraft F_spring(x)", color='orange')
plt.axhline(Fy, color='red', linestyle='--', label="±Fy")
plt.axhline(-Fy, color='red', linestyle='--')
plt.xlabel("Tid [s]")
plt.ylabel("Fjærkraft [N]")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
