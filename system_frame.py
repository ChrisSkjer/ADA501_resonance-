import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#matrial parametrenHEA100
E = 210000 #N/mm^2
Iy = 3.49*10**6 #mm^4
Wpl = 2*41.5*10**3 #mm^3 
fy = 355 #N/mm^2

# Parametre (sett disse til dine faktiske verdier)
m_kg = 1000.0
m = m_kg / 1000.0   # omregning til N·s²/mm       
c   = 0.1          # demping
Mp  = fy*Wpl          # Nmm plastisk momentkapasitet for tverrsnittet
h   = 2000           # søylehøyde
k_el = 6*E*Iy/h**3        

# Flytekraft og flyteforskyvning
Fy   = Mp / h               # flytekraft (horisontal)
x_y  = Fy / k_el            # flyteforskyvning (elastisk del)

print("Flytekraft Fy =", Fy, "N")
print("Flyteforskyvning x_y =", x_y, "mm")

# Last som funksjon av tid
def F_ext(t):
    # F.eks. lastpuls som driver systemet inn i plastisk
    if 0.5 < t < 2.0:
        return 10000
    else:
        return 0.0

# Høyresiden til ODE-systemet: y = [x, v, x_p]
def rhs(t, y):
    x, v, x_p = y

    # Elastisk del av deformasjonen
    x_e = x - x_p
    f   = k_el * x_e   # fjærkraft

    # Sjekk flyt basert på ELASTISK forskyvning
    if abs(x_e) > x_y and f * v > 0:
        # plastisk flyt: mer av deformasjonen går inn i x_p
        x_p_dot = v
        # kraftnivået begrenses til flytenivå
        f = Fy * np.sign(x_e)
    else:
        # elastisk: ingen plastisk utvikling
        x_p_dot = 0.0

    # Bevegelseslikningen
    a = (F_ext(t) - c * v - f) / m

    return [v, a, x_p_dot]

# Startbetingelser
x0  = 0.0    # startforskyvning
v0  = 0.0    # starthastighet
xp0 = 0.0    # plastisk del i start
y0  = [x0, v0, xp0]

# Tidsoppsett
t_span = (0.0, 10.0)
t_eval = np.linspace(t_span[0], t_span[1], 2001)

sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval)

t  = sol.t
x  = sol.y[0]    # total forskyvning
v  = sol.y[1]    # hastighet
xp = sol.y[2]    # plastisk forskyvning (varig del)

# Plot total og plastisk forskyvning
plt.figure()
plt.plot(t, x,  label="x(t) total")
plt.plot(t, xp, "--", label="x_p(t) plastisk")
plt.xlabel("Tid [s]")
plt.ylabel("Forskyvning [m]")
plt.title("Dynamisk respons med varig plastisk deformasjon\n(flytegrense basert på forskyvning)")
plt.grid(True)
plt.legend()
plt.show()
