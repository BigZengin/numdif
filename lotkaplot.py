import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from proj1 import *

a = 3
b = 9
c = 15
d = 15
y0 = np.array([1,1])
ivp = LotkaVolterra(t0=0, T=10, y0=y0, a=a, b=b, c=c, d=d)
solver_LV = RK34(ivp, N=1, tol=1e-8, adaptive = True)
solver_LV.integrate()

# Extract results
t = solver_LV.grid
x = solver_LV.us[0, :]
y = solver_LV.us[1, :]

# Time evolution
plt.figure()
plt.plot(t, x, label="Prey (x)")
plt.plot(t, y, label="Predator (y)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Lotka–Volterra Predator–Prey Model")
plt.legend()
plt.grid(True)

# Phase plot (x vs y)
plt.figure()
plt.plot(x, y)
plt.xlabel("Prey (x)")
plt.ylabel("Predator (y)")
plt.title("Phase Plane of Lotka–Volterra System")
plt.grid(True)

plt.show()

# =====================================================
# 4. Check conservation of H(x, y)
# =====================================================
# Long simulation to test drift in invariant
def H(a, b, c, d, x, y):
    return c*x + b*y - d*np.log(x) - a*np.log(y)

# Compute invariant H
H_vals = H(ivp.a, ivp.b, ivp.c, ivp.d, x, y)
H0 = H_vals[0]
rel_error = np.abs(H_vals / H0 - 1)

# Plot drift over time (semilog)
plt.figure()
plt.loglog(t, rel_error)
plt.xlabel("Time")
plt.ylabel("|H(x,y)/H(x0,y0) - 1|")
plt.title("Relative Drift of Invariant H(x, y)")
plt.grid(True)
plt.show()
