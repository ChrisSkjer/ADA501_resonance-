import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Materialparametre HEA100 (mm–N–s-system)
E   = 210000          # N/mm^2
Iy  = 3.49e6          # mm^4
Wpl = 83e3        # mm^3 
Wel = 72.8e3
fy  = 355             # N/mm^2

# Systemparametre
m_kg = 11000.0
m    = m_kg / 1000.0   # N*s^2/mm  (masse i mm-systemet)
c    = 2.3            # N*s/mm    (demping)
Mp   = fy * Wpl        # Nmm       (plastisk momentkapasitet)
Mel = fy*Wel
h    = 2000.0          # mm        (søylehøyde)
k_el = 6*E*Iy/h**3     # N/mm      (elastisk stivhet i topp)

# Flytenivå (start)
Fy  = 2*Mel / h          # N
x_y = Fy / k_el       # mm (elastisk flyteforskyvning)

# Bilinjær hardening: liten stivhet etter flyt
alpha  = 0.01          # 1 % av elastisk stivhet
k_pl   = alpha * k_el  # N/mm

print("Fy  =", Fy, "N")
print("x_y =", x_y, "mm")
print("k_el =", k_el, "N/mm")
print("k_pl =", k_pl, "N/mm")
print("beregning c for dempingsforhold", 0.05*2*np.sqrt(m*k_el))
print("naturlig egenvikelfrekves er", np.sqrt(k_el/m))

# Ytre last - sinus som krysser flyt flere ganger
def F_ext(t):
    P0 = 1.2 * Fy          # topp litt over flyt
    omega = 2*np.pi/2.0    # periode 2 s
    return P0 * np.sin(omega * t)
    #return Fy*0.9

# RHS: y = [x, v, x_p]
def rhs(t, y):
    x, v, x_p = y

    x_e = x - x_p              # elastisk del
    f   = k_el * x_e           # kraft fra fjæren

    # Standard elastisk til vi når flyt
    x_p_dot = 0.0

    # Sjekk plastisk flyt (bilinjær med hardening)
    if abs(f) >= Fy and f * v > 0:
        # Plastisk flyt: juster x_p_dot slik at tangentstivheten blir k_pl
        x_p_dot = v * (1.0 - k_pl / k_el)
        # Merk: vi "klipper" ikke til Fy – hardening gjør at kraften kan bli > Fy

    # Bevegelseslikning
    a = (F_ext(t) - c * v - f) / m
    return [v, a, x_p_dot]

# Integrasjon
t_span = (0.0, 60.0)
t_eval = np.linspace(*t_span, 2001)
y0 = [0.0, 0.0, 0.0]   # x, v, x_p

sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval)

t  = sol.t
x  = sol.y[0]      # total forskyvning
v  = sol.y[1]      # hastighet
xp = sol.y[2]      # plastisk del

# Beregn fjærkraft og ytre last til plott
f_hist     = k_el * (x - xp)
F_ext_hist = np.array([F_ext(ti) for ti in t])

# --- Plott 1: forskyvning (total + plastisk) ---
plt.figure()
plt.plot(t, x,  label="x(t) total")
plt.plot(t, xp, "--", label="x_p(t) plastisk")
plt.xlabel("Tid [s]")
plt.ylabel("Forskyvning [mm]")
plt.title("Forskyvning med bilinjær hardening")
plt.grid(True)
plt.legend()

# --- Plott 2: kraft over tid ---
plt.figure()
plt.plot(t, f_hist, label="fjærkraft f(t)")
plt.plot(t, F_ext_hist, "--", label="ytre last F_ext(t)")
plt.xlabel("Tid [s]")
plt.ylabel("Kraft [N]")
plt.title("Kraftrespons over tid (bilinjær med varig plastisk deformasjon)")
plt.grid(True)
plt.legend()

# --- Plott 3: kraft–forskyvning (hysterese) ---
plt.figure()
plt.plot(x, f_hist)
plt.xlabel("Forskyvning x [mm]")
plt.ylabel("Fjærkraft f [N]")
plt.title("Kraft–forskyvningskurve (bilinjær + varig plastisk)")
plt.grid(True)

plt.show()
