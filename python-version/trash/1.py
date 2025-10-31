import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt

# Exemples de points
x = np.array([7, 10, 15, 20, 25])
y = np.array([3, 3.05, 3.18, 3.24, 3.38])

# Interpolation cubique (splines)
cubic = CubicSpline(x, y)

# Valeurs pour tracer la courbe
x_new = np.linspace(min(x), max(x), 100)
plt.plot(x, y, 'o', label='points')
plt.plot(x_new, cubic(x_new), label='cubique')
plt.legend()
plt.show()

M = 2000
E = 200000

x_new = np.linspace(min(x), max(x), 100)
M_simu = [E * cubic(x) / 100 / 12 / ( 1 - (1 + cubic(x) / 100 / 12)**(-x * 12)) for x in x_new]

plt.plot(x_new, M_simu, label='mensualité')
plt.xlabel("Durée (années)")
plt.ylabel("Mensualité (€)")
# Plot horizontal line for M
plt.axhline(y=M, color='r', linestyle='--', label='M = 2000€')
plt.legend()
plt.show()
# Find loan duration such that M_simu = M
f = interp1d(M_simu, x_new, fill_value="extrapolate")
D = f(M)
print(f"Durée du prêt pour une mensualité de {M}€: {D:.2f} années")