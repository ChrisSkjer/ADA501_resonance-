import numpy as np
import matplotlib.pyplot as plt

x_left = np.linspace(0, 0.99, 300)
x_right = np.linspace(1.01, 4, 300)

y_left = 1 / (1 - x_left)
y_right = 1 / (1 - x_right)

plt.figure()
plt.plot(x_left, y_left)
plt.plot(x_right, y_right)
plt.axvline(1, linestyle='--')   # markerer (ω/ω_n)^2 = 1
plt.xlabel(r'$(\omega / \omega_n)^2$')
plt.ylabel(r'$1 / (1 - (\omega / \omega_n)^2)$')
plt.ylim(-10, 10)                # begrens y-akse så du ser formen
plt.grid(True)
plt.show()


