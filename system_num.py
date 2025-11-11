import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#løser masse demper fjær  systemet med ekstern kraft numerisk 

# Parametre
m, c, k = 1.0, 0.5, 20.0
omega_n = np.sqrt(k/m) #naturlig udempet frekvens
F = lambda t: 0.5 * np.sin(4.4* t)  # ytre kraft
#F = lambda t: 0  # ytre kraft
# ODE-funksjon: y = [x, v]
def system(t, y):
    x, v = y
    dxdt = v
    dvdt = (F(t) - c * v - k * x) / m
    return [dxdt, dvdt]

# Tidsintervall og initialbetingelser
t_span = (0,30)
y0 = [0, 0.0]
t_eval = np.linspace(*t_span, 1000)

# Løs systemet
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Plot
plt.plot(sol.t, sol.y[0], label='x(t)')
plt.plot(sol.t, sol.y[1], label='v(t)', linestyle='--')
plt.title("Løsning med solve_ivp")
plt.xlabel("Tid [s]")
plt.legend()
plt.grid()
plt.show()
