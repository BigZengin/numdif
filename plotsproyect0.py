import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from main import *

#Explicit Euler Scalar plot
ivp = LTE(t0 = 0, T = 1 ,y0=np.array([2]) , A=np.array
([[ -1]]))
solver = ExplicitEuler(ivp , N = 100)
solver.integrate()
plt.plot(solver.grid ,solver.us.flatten() )

#Exact scalar plot
ts = np.linspace(0, 1)
ys = [ivp.exact(t) for t in ts]
plt.plot(ts, ys)
plt.show()

#difference scalar plot
ys = [ivp.exact(t).item() for t in solver.grid]
dif =  solver.us.flatten() - ys

plt.plot(solver.grid ,dif)
plt.show()

#Explicit euler vector plot
A = np.array([[-1, 1], [1, -3]])
y0 = np.array([1, 2])
ivp = LTE(t0=0, T=1, y0=y0, A=A)
solver = ExplicitEuler(ivp, N=100)
solver.integrate()

# Plot the first component of the approximations {u_n} vs. the temporal grid points {t_n}:
plt.plot(solver.grid, solver.us[0, :], label="u1(t)")
plt.plot(solver.grid, solver.us[1, :], label="u2(t)")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.legend()


#Exact Vector plot
ts = np.linspace(0, 1)
ys = [ivp.exact(t) for t in ts]
plt.plot(ts, ys, label="exact")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.show()


ys = np.array([ivp.exact(t) for t in solver.grid]).T
dif =  solver.us - ys
difnorm= np.array([scipy.linalg.norm(dif[:,i])/scipy.linalg.norm(ys[:, i]) for i in range(solver.N+1)])
plt.plot(solver.grid, difnorm, label="norm")
plt.plot(solver.grid ,dif[0,:])
plt.plot(solver.grid ,dif[1,:])
plt.legend()
plt.show()

evaluator = Evaluator(ivp, ExplicitEuler)
[errs, grid] = evaluator.errvstime(100, True)
# Plot errors with log-scale on the y-axis
plt.figure()
plt.semilogy(grid, errs)

# Compute the error at t = T for different h corresponding to N = 2^3, 2^4, ..., 2^10
Ns = 2**np.arange(3, 11)
[errs, hs] = evaluator.errvsh(Ns,True)

# Plot these errors in a log-log scale
plt.figure()
plt.loglog(hs, errs)

# Plot a comparison line which is O(h)
plt.loglog(hs, errs[-1] * hs / hs[-1], '--', label="O(h) Line")
plt.legend()
plt.show()




Ns = 2**np.arange(3, 13)
T_val = 1.0  
t0_val = 0
hs_ref = (T_val - t0_val) / Ns

# Try for different lambda values

lambdas = [-2.0, -0.1, 1.0] # Different values for λ 
colors = ['b', 'r', 'g']
y0_val = np.array([1.0])

plt.figure(figsize=(10, 6))
plt.title(f'Final Global Error vs. h (Explicit Euler Method)')
plt.xlabel('Time Step h (log scale)')
plt.ylabel('Final Error ||u(T)-y(T)|| (log scale)')
plt.grid(True, which="both", ls="--")

# 1. Plot error curves for different λ
for i, lam in enumerate(lambdas):
    A_scalar = np.array([[lam]])
    ivp_scalar = LTE(t0=t0_val, T=T_val, y0=y0_val, A=A_scalar)
    evaluator_scalar = Evaluator(ivp_scalar, ExplicitEuler)
    [errs, hs] = evaluator_scalar.errvsh(Ns, True)
    plt.loglog(hs, errs, marker='o', linestyle='-', color=colors[i], 
               label=f'Error $\\lambda={lam}$')

# 2. Plot the O(h) reference line.
O_h_line = errs[-1] * (hs_ref / hs_ref[-1])
plt.loglog(hs_ref, O_h_line, 'k--', label='Reference $O(h)$ (Slope -1)')

plt.legend()
plt.show()


# Case 1: λ > 0 (Stable System)
lam_stab = -1.0
ivp_stab = LTE(t0=0, T=5, y0=np.array([2.0]), A=np.array([[lam_stab]]))
evaluator_stab = Evaluator(ivp_stab, ExplicitEuler)
N_time = 200 # Number of steps to resolve time evolution

# Plot 1: Absolute Error vs. Time (semilogy)
[errs_abs, grid] = evaluator_stab.errvstime(N_time, True)
plt.figure(figsize=(10, 6))
plt.semilogy(grid, errs_abs, label='Absolute Error')
plt.title(f'Absolute Error vs. Time ($\\lambda={lam_stab}$, Stable)')
plt.xlabel('Time t')
plt.ylabel('Absolute Error ||u(t)-y(t)|| (log scale)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

# Plot 2: Relative Error vs. Time (semilogy)
[errs_rel, grid] = evaluator_stab.errvstime(N_time, False)
plt.figure(figsize=(10, 6))
plt.semilogy(grid, errs_rel, label='Relative Error')
plt.title(f'Relative Error vs. Time ($\\lambda={lam_stab}$, Stable)')
plt.xlabel('Time t')
plt.ylabel('Relative Error (log scale)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()


# Test Matrix: Stable System (Eigenvalues -1 and -3)
A_test = np.array([[-1, 10], [0, -3]])
y0_test = np.array([1, 1])
t0_test = 0
T_test = 10 

ivp_test = LTE(t0=t0_test, T=T_test, y0=y0_test, A=A_test)
evaluator_test = Evaluator(ivp_test, ExplicitEuler)

# Error vs hs
hs_ref_test = (T_test - t0_test) / Ns 

[errs_h, hs_h] = evaluator_test.errvsh(Ns, True) # Absolute Error

plt.figure(figsize=(10, 6))
plt.loglog(hs_h, errs_h, marker='o', label='Vector Absolute Error')

# Plot the O(h) reference line
O_h_line_test = errs_h[-1] * (hs_ref_test / hs_ref_test[-1])
plt.loglog(hs_ref_test, O_h_line_test, 'k--', label='Reference $O(h)$ (Slope -1)')

plt.title(f'Final Global Error vs. h for Matrix A (T={T_test})')
plt.xlabel('Time Step h (log scale)')
plt.ylabel('Final Error ||u(T)-y(T)|| (log scale)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

#(Error vs. Time)

N_time_test = 500 # More steps due to larger T

# Plot 1: Absolute Error 
[errs_time_abs, grid_time] = evaluator_test.errvstime(N_time_test, True)
plt.figure(figsize=(10, 6))
plt.semilogy(grid_time, errs_time_abs, label='Absolute Error')
plt.title(f'Absolute Error vs. Time (Matrix A, T={T_test})')
plt.xlabel('Time t')
plt.ylabel('Absolute Error ||u(t)-y(t)|| (log scale)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

# Plot 2: Relative Error 
[errs_time_rel, grid_time] = evaluator_test.errvstime(N_time_test, False)
plt.figure(figsize=(10, 6))
plt.semilogy(grid_time, errs_time_rel, label='Relative Error')
plt.title(f'Relative Error vs. Time (Matrix A, T={T_test})')
plt.xlabel('Time t')
plt.ylabel('Relative Error (log scale)')
plt.grid(True, which="both", ls="--")
plt.legend()

plt.show()
