import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from proj1 import *

#RK4 vector plot
A = np.array([[-1, 1], [1, -3]])
y0 = np.array([1, 2])
ivp = LTE(t0=0, T=1, y0=y0, A=A)
solver = RK4(ivp, N=100)
solver.integrate()

# RK4 vector plot (numerical solution)
plt.figure()
plt.plot(solver.grid, solver.us[0, :], label="u1(t)")
plt.plot(solver.grid, solver.us[1, :], label="u2(t)")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("RK4 Approximation of System")
plt.legend()

# Exact solution plot
ts = np.linspace(0, 1)
ys = [ivp.exact(t) for t in ts]
plt.figure()
plt.plot(ts, [y[0] for y in ys], label="y1(t) exact")
plt.plot(ts, [y[1] for y in ys], label="y2(t) exact")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Exact Solution of System")
plt.legend()

# Error vs time plot
evaluator = Evaluator(ivp, RK4)
[errs, grid] = evaluator.errvstime(100, True)
plt.figure()
plt.semilogy(grid, errs)
plt.xlabel("t")
plt.ylabel("Absolute Error")
plt.title("Error vs Time (RK4)")
plt.grid(True)

# Error vs step size plot (global error)
Ns = 2**np.arange(3, 11)
[errs, hs] = evaluator.errvsh(Ns, True)
plt.figure()
plt.loglog(hs, errs, 'o-', label="RK4 Error at T")

# O(h⁴) comparison line
plt.loglog(hs, errs[-1] * (hs / hs[-1])**4, '--', label="O(h⁴) Line")
plt.xlabel("Step size (h)")
plt.ylabel("Error at T")
plt.title("Global Error vs Step Size (RK4)")
plt.legend()
plt.grid(True)

plt.show()



# =====================================================
# 2. Adaptive RK34 test
# =====================================================
solver_adapt = RK34(ivp, N=1, tol=1e-3, adaptive = True)

solver_adapt.integrate()

# plot adaptive steps
plt.figure()
plt.plot(solver_adapt.grid, solver_adapt.us[0, :], 'o-', label="u1(t)")
plt.plot(solver_adapt.grid, solver_adapt.us[1, :], 'o-', label="u2(t)")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("RK34 Adaptive Approximation")
plt.legend()

# compare adaptive vs exact
ys_exact = np.array([ivp.exact(t) for t in solver_adapt.grid])
diff = np.linalg.norm(solver_adapt.us - ys_exact.T, axis=0)
plt.figure()
plt.semilogy(solver_adapt.grid, diff, label="|u - y_exact|")
plt.xlabel("t")
plt.ylabel("Error")
plt.title("Adaptive RK34: Error vs Time")
plt.legend()
plt.grid(True)

# show adaptive step sizes
steps = np.diff(solver_adapt.grid)
plt.figure()
plt.plot(solver_adapt.grid[:-1], steps, 'o-')
plt.xlabel("t")
plt.ylabel("Step size h")
plt.title("Adaptive RK34 Step Size Evolution")
plt.grid(True)


plt.show()




