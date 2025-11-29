import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Systemparametere
m = 1.0      # Masse [kg]
c = 6.0      # Demping [Ns/m]
k = 1000.0   # Fjærstivhet [N/m]
Fy = 132.0   # Flytegrense [N]
F0 = 150     # Amplitude ytre kraft
omega = 2.0  # Frekvens ytre kraft

# Ytre kraft: kort puls
def external_force(t):
    return F0 if 0 <= t <= 0.2 else 0.0
    #return F0*np.sin(2*np.pi*t)
# ODE-system med plastisk deformasjon
def system(t, y):
    x, v, x_p = y
    u_y = Fy / k               # Flyteforskyvning
    delta_x = x - x_p          # Elastisk strekk i fjæren

    # Elastisk eller plastisk flyt
    if abs(delta_x) > u_y:
        Fs = Fy * np.sign(delta_x)  # Fjærkraft holdes konstant
        # Flyt bare i samme retning som bevegelse
        if np.sign(v) == np.sign(delta_x):
            dx_pdt = v              # Fjæren "glir" plastisk
        else:
            dx_pdt = 0.0            # Ingen plastisk oppdatering ved retur
    else:
        Fs = k * delta_x            # Vanlig fjærkraft innen elastisk sone
        dx_pdt = 0.0                # Ingen plastisk oppdatering

    dxdt = v
    dvdt = (external_force(t) - c * v - Fs) / m
    return [dxdt, dvdt, dx_pdt]


# Initialverdier og tidsoppløsning
y0 = [0.0, 0.0, 0.0]  # x, v, x_p
t_span = (0, 10)
t_eval = np.linspace(*t_span, 2000)
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Hent løsninger
x_vals = sol.y[0]
v_vals = sol.y[1]
xp_vals = sol.y[2]
Fs_vals = k * (x_vals - xp_vals)

# Plotting
plt.figure(figsize=(10, 5))

# Forskyvning
plt.subplot(2, 1, 1)
plt.plot(sol.t, x_vals, label="x(t) – Forskyvning")
plt.axhline(Fy / k, color='gray', linestyle='--', label="Flytegrense (x_y)")
plt.axhline(-Fy / k, color='gray', linestyle='--')
plt.ylabel("Forskyvning [m]")
plt.legend()
plt.grid()

# Fjærkraft
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
plt.title("Hysterese: kraft–forskyvning")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
