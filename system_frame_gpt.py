import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Materialparametere
E = 210e9  # N/m²
Ic = 6.6e-6   # m⁴
Wb = 1106e-6  # m³ (fra mm³ til m³)

# Geometri
h = 4  # m

# Fjærstivhet
k = 6 * E * Ic / h**3  # N/m

# Flytegrense og forskyvning
sigma_y = 355e6          # Pa
Fy = sigma_y * Wb        # N
u_y = Fy / k             # Kritisk elastisk forskyvning før flyt

# Systemparametere
m = 500      # kg
c = 50.0     # Ns/m
F0 = 80 *10**3     # N
omega = 2.0  # rad/s

# Ytre kraft: kort puls
def external_force(t):
    return F0 if 0 <= t <= 0.2 else 0.0

# ODE-system med ideell plastisk flyt og lagret plastisk deformasjon
def system(t, y):
    x, v, x_p = y
    delta_x = x - x_p  # Elastisk del av forskyvningen

    # Fjærkraft
    if abs(delta_x) <= u_y:
        Fs = k * delta_x
        dx_pdt = 0.0  # Ingen plastisk flyt
    else:
        Fs = Fy * np.sign(delta_x)
        # Tillat plastisk flyt bare i samme retning som bevegelse
        dx_pdt = v if np.sign(v) == np.sign(delta_x) else 0.0

    dxdt = v
    dvdt = (external_force(t) - c * v - Fs) / m
    return [dxdt, dvdt, dx_pdt]

# Initialbetingelser og løsning
y0 = [0.0, 0.0, 0.0]  # [x, v, x_p]
t_span = (0, 10)
t_eval = np.linspace(*t_span, 5000)
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Hent løsninger
x_vals = sol.y[0]
v_vals = sol.y[1]
xp_vals = sol.y[2]
delta_x = x_vals - xp_vals

# Fjærkraft
Fs_vals = np.where(
    np.abs(delta_x) <= u_y,
    k * delta_x,
    Fy * np.sign(delta_x)
)

# Plot forskyvning
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(sol.t, x_vals, label="x(t) – Forskyvning")
plt.plot(sol.t, xp_vals, '--', label="x_p(t) – Plastisk forskyvning")
plt.axhline(u_y, color='gray', linestyle='--', label="Flytegrense ±u_y")
plt.axhline(-u_y, color='gray', linestyle='--')
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

# Kraft–forskyvning (hysterese) plot
plt.figure(figsize=(6, 5))
plt.plot(x_vals, Fs_vals, color='darkgreen', label="F_s vs x")
plt.xlabel("Forskyvning x(t) [m]")
plt.ylabel("Fjærkraft F_s [N]")
plt.title("Hysterese: kraft–forskyvning med ideell plastisk flyt")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
