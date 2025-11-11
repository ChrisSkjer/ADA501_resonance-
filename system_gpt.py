import numpy as np
import matplotlib.pyplot as plt

# Initialbetingelser
x0 = 1     # forskyvning ved t=0
v0 = 0      # hastighet ved t=0

# Systemparametre
m = 1
c = 0.5
k = 20
t = np.linspace(0, 10, 1000)

# Beregninger
omega_n = np.sqrt(k/m)
zeta = c / (2*np.sqrt(m*k))
omega_d = omega_n * np.sqrt(1 - zeta**2)

if zeta < 1:  # underdempet
    print("Svinger (underdempet)")
    

    # Beregn D og E fra initialbetingelser
    D = x0
    E = (v0 + zeta * omega_n * x0) / omega_d
    print(f"D: {D}, E:{E}")
    x_full = np.exp(-zeta * omega_n * t) * (D * np.cos(omega_d * t) + E * np.sin(omega_d * t))

elif zeta > 1:  # overdempet
    print("Overdempet")
    r1 = -omega_n * (zeta - np.sqrt(zeta**2 - 1))
    r2 = -omega_n * (zeta + np.sqrt(zeta**2 - 1))

    A = (v0 - r2 * x0) / (r1 - r2)
    B = x0 - A

    x_full = A * np.exp(r1 * t) + B * np.exp(r2 * t)

else:  # kritisk dempet
    print("Kritisk dempet")
    r = -omega_n
    A = x0
    B = v0 - r * x0

    x_full = (A + B * t) * np.exp(r * t)

# Plott
plt.plot(t, x_full, label='x(t)')
plt.title('Respons for masse-fjær-demper-system')
plt.xlabel('Tid [s]')
plt.ylabel('Forskyvning x(t)')
plt.grid()
plt.legend()
plt.show()
