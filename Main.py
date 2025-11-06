import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
class IVP:
    def __init__(self, t0, T, y0):
        self.t0 = t0
        self.T = T
        self.y0 = y0


    def f(self,t,y):
        return NotImplemented("The function f has not been given")

class LTE(IVP):
    def __init__(self, t0, T, y0, A):
        super().__init__(t0, T, y0)
        self.A = A


    def f(self, t, y):
        return self.A @ y


    def exact(self, t):
        exponential = scipy.linalg.expm(t*self.A)
        return exponential @ self.y0

class Solver:
    def __init__(self, ivp, N: int):
        self.ivp = ivp
        self.N = N
        self.u = ivp.y0
        self.grid = np.linspace(ivp.t0, ivp.T, N + 1)
        self.h = (ivp.T - ivp.t0) / N
        self.us = None 

    def step(self, tn, un):
        raise NotImplementedError("step() not implemented in subclass")

    def integrate(self):
        m = len(self.u) 
        us = np.zeros((m, self.N + 1))
        us[:, 0] = self.u

        for i in range(self.N):
            tn = self.grid[i]
            un = us[:, i]
            us[:, i + 1] = self.step(tn, un)

        self.us = us
        return self.grid, self.us

class ExplicitEuler(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)

    def step(self, tn, un):
        return un + self.h * self.ivp.f(tn, un)
    


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


