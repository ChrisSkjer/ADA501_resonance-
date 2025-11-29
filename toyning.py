import numpy as np
import matplotlib.pyplot as plt

# Eksempel: Lineær elastisk + flyt
E = 210e9              # Elastisitetsmodul [Pa]
sigma_y = 355e6        # Flytegrense [Pa]
strain_y = sigma_y / E # Tøyning ved flyt

# Ingeniør-tøyning
strain_eng = np.linspace(0, 0.3, 500)
sigma_eng = np.where(strain_eng <= strain_y,
                     E * strain_eng,
                     sigma_y)  # Idealplastisk flyt

# Sann spenning og tøyning
strain_true = np.log(1 + strain_eng)
sigma_true = sigma_eng * (1 + strain_eng)

# Plot
plt.figure(figsize=(8,5))
plt.plot(strain_eng, sigma_eng / 1e6, label="Ingeniørspenning–tøyning", linestyle='--')
plt.plot(strain_true, sigma_true / 1e6, label="Sann spenning–tøyning", linewidth=2)
plt.xlabel("Tøyning")
plt.ylabel("Spenning [MPa]")
plt.title("Ingeniør vs. sann spenning–tøyning")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
