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
    
class Evaluator():
    def __init__(self, ivp, solverType):
        self.ivp = ivp
        self.solverType  = solverType
    
    def errvstime(self, N, errType): #errtype is boolean, true absolute.
        
        solver = self.solverType(self.ivp, N)
        solver.integrate()
        us = solver.us
        ys = np.array([ivp.exact(t) for t in solver.grid]).T
        dif =  solver.us - ys
        if errType:
            errs = np.array([scipy.linalg.norm(dif[:,i]) for i in range(N+1)])
        else:
            errs = np.array([scipy.linalg.norm(dif[:,i])/scipy.linalg.norm(ys[i]) for i in range(N+1)])

        return errs , solver.grid
    
    def errvsh(self, Ns, errType):
        errs = np.zeros(len(Ns))
        hs = np.zeros(len(Ns))
        for i in range (len(Ns)):
            solver = self.solverType(self.ivp, Ns[i])
            solver.integrate()
            [error, grid] = self.errvstime(Ns[i], errType)
            err = error[-1]
            errs[i] = err
            hs[i] = Ns[i]/self.ivp.T
        return errs, hs
    






        


        


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
[grid, errs] = evaluator.errvstime(100, True)

# Plot errors with log-scale on the y-axis
plt.figure()
plt.semilogy(grid, errs)

# Compute the error at t = T for different h corresponding to N = 2^3, 2^4, ..., 2^10
Ns = 2**np.arange(3, 11)
[hs, errs] = evaluator.errvsh(Ns,True)

# Plot these errors in a log-log scale
plt.figure()
plt.loglog(hs, errs)

# Plot a comparison line which is O(h)
plt.loglog(hs, hs * 2 * errs[-1] / hs[-1])
plt.show()
