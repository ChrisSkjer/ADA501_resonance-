import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#TODO: Legg inn Mel

# materialparametre HEA100
E = 210000 # N/mm^2
Iy = 3.49*10**6 # mm^4
Wpl = 83e3      # mm^3 
Wel = 72.8e3    # mm^3
fy = 355        # N/mm^2

# Parametre (sett disse til dine faktiske verdier)
m_kg = 11000.0
m = m_kg / 1000.0   # N·s^2/mm       
c   = 7.0           # N·s/mm (viskøs demping)

Mp  = fy * Wpl      # Nmm plastisk momentkapasitet for tverrsnittet
Mel = fy * Wel
h   = 2000.0        # mm søylehøyde
k_el = 6*E*Iy/h**3  # N/mm
omega = np.sqrt(k_el/m)

# Flytekraft og flyteforskyvning (her fortsatt basert på Mp)
Fy   = 2 * Mel / h        # N (horisontal flytekraft, 2Mp fordi to søyler)
x_y  = Fy / k_el         # mm (flyteforskyvning elastisk del)

# Plastisk dempningskoeffisient (tunes)
eta = 0.1 * c            # N·s/mm, f.eks. 10% av lineær demping

print("Flytekraft Fy =", Fy, "N")
print("Flyteforskyvning x_y =", x_y, "mm")
print("beregning c for dempingsforhold", 0.05*2*np.sqrt(m*k_el))
print("naturlig egenvikelfrekvens er", omega)

# Last som funksjon av tid
def F_ext(t):
    # F.eks. lastpuls som driver systemet inn i plastisk
    if 0.0 < t < 1.0:
        #return Fy*0.5*np.sin(3*omega*t)
        #return Fy*1.2*np.sin(np.pi*t)
        return Fy * 0.8
    else:
        #return 6000*np.sin(omega*t)*np.exp(-3*t)
        return 0.0

def rhs(t, y):
    x, v, x_p = y

    x_e = x - x_p
    f_trial = k_el * x_e          # ren elastisk kraft
    f = f_trial
    x_p_dot = 0.0
    F_plast = 0.0                 # plastisk energitap-kraft

    if abs(f_trial) > Fy:
        # Kraften får aldri være større enn Fy (perfekt plastisk kappe)
        f = Fy * np.sign(f_trial)

        # Plastisk flyt bare hvis vi faktisk laster i samme retning
        if f_trial * v > 0:
            x_p_dot = v           # slik at x_e holder seg omtrent rundt x_y
            # plastisk energitap: bare aktiv når x_p_dot != 0
            F_plast = eta * x_p_dot

    # Bevegelseslikning, nå med ekstra plastisk dempingsledd
    a = (F_ext(t) - c * v - f - F_plast) / m
    return [v, a, x_p_dot]

# Startbetingelser
x0  = 0.0    # mm
v0  = 0.0    # mm/s
xp0 = 0.0    # mm
y0  = [x0, v0, xp0]

# Tidsoppsett
t_span = (0.0, 30.0)
t_eval = np.linspace(t_span[0], t_span[1], 2001)

sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval)

t  = sol.t
x  = sol.y[0]    # total forskyvning (mm)
v  = sol.y[1]    # hastighet (mm/s)
xp = sol.y[2]    # plastisk forskyvning (mm)

# Beregn fjærkraft-historikk for plott
f_hist = []
for xi, vi, xpi in zip(x, v, xp):
    x_e = xi - xpi
    f_trial = k_el * x_e
    f = f_trial
    if abs(f_trial) > Fy:
        f = Fy * np.sign(f_trial)
    f_hist.append(f)
f_hist = np.array(f_hist)

# Plot 1: forskyvning
plt.figure()
plt.plot(t, x, label="x(t) total")
plt.plot(t, xp, "--", label="x_p(t) plastisk")
plt.xlabel("Tid [s]")
plt.ylabel("Forskyvning [mm]")
plt.title("Forskyvning over tid")
plt.grid(True)
plt.legend()

# Plot 2: fjærkraft
plt.figure()
plt.plot(t, f_hist, label="fjærkraft f(t)")
plt.xlabel("Tid [s]")
plt.ylabel("Kraft [N]")
plt.title("Fjærkraft over tid")
plt.grid(True)
plt.legend()

# Plot 3: kraft–forskyvning (hysterese)
plt.figure()
plt.plot(x, f_hist)
plt.xlabel("Forskyvning x [mm]")
plt.ylabel("Fjærkraft f [N]")
plt.title("Kraft–forskyvningskurve (hysterese)")
plt.grid(True)

plt.show()
