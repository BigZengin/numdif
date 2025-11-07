import numpy as np
import scipy as sp
import scipy.linalg
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
        ys = np.array([self.ivp.exact(t) for t in solver.grid]).T
        dif =  solver.us - ys
        if errType:
            errs = np.array([scipy.linalg.norm(dif[:,i]) for i in range(N+1)])
        else:
            errs = np.array([scipy.linalg.norm(dif[:,i])/scipy.linalg.norm(ys[:,i]) for i in range(N+1)])

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
            hs[i] = (self.ivp.T - self.ivp.t0) / Ns[i]  
        return errs, hs
