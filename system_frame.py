import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#matrial parametrenHEA100
E = 210000 #N/mm^2
Iy = 3.49*10**6 #mm^4
Wpl = 2*41.5*10**3 #mm^3 
fy = 355 #N/mm^2

# Parametre (sett disse til dine faktiske verdier)
m_kg = 11000.0
m = m_kg / 1000.0   # omregning til N·s²/mm       
c   = 2.3          # demping
Mp  = fy*Wpl          # Nmm plastisk momentkapasitet for tverrsnittet
h   = 2000           # søylehøyde
k_el = 6*E*Iy/h**3        
omega = np.sqrt(k_el/m)
# Flytekraft og flyteforskyvning
Fy   = 2*Mp / h               # flytekraft (horisontal) 2Mp fordi vi har to søyler
x_y  = Fy / k_el            # flyteforskyvning (elastisk del)

print("Flytekraft Fy =", Fy, "N")
print("Flyteforskyvning x_y =", x_y, "mm")
print("beregning c for dempingsforhold", 0.05*2*np.sqrt(m*k_el))
print("naturlig egenvikelfrekves er", omega)

# Last som funksjon av tid
def F_ext(t):
    # F.eks. lastpuls som driver systemet inn i plastisk
    if 0.0 < t < 5:
        return Fy*0.5*np.sin(1.2*omega*t)
        #return Fy*1.2*np.sin(np.pi*t)
        #return Fy*1.2
    else:
        #return 6000*np.sin(omega*t)*np.exp(-3*t)
        return 0.0
    
def rhs(t, y):
    x, v, x_p = y

    x_e = x - x_p
    f_trial = k_el * x_e          # ren elastisk kraft
    f = f_trial
    x_p_dot = 0.0

    if abs(f_trial) > Fy:
        # Kraften får aldri være større enn Fy
        f = Fy * np.sign(f_trial)

        # Plastisk flyt bare hvis vi faktisk laster i samme retning
        if f_trial * v > 0:
            x_p_dot = v        # slik at x_e holder seg rundt x_y

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

plt.figure()
plt.plot(x, f_hist)
plt.xlabel("Forskyvning x [mm]")
plt.ylabel("Fjærkraft f [N]")
plt.title("Kraft–forskyvningskurve (hysterese)")
plt.grid(True)
plt.show()

plt.show()