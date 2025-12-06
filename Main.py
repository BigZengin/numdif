import numpy as np
import scipy as sp
import scipy.linalg
class IVP:
    def __init__(self, t0, T, y0):
        self.t0 = t0
        self.T = T
        self.y0 = y0


    def f(self,t,y):
        raise NotImplementedError("The function f has not been given")

class LTE(IVP):
    def __init__(self, t0, T, y0, A):
        super().__init__(t0, T, y0)
        self.A = A


    def f(self, t, y):
        return self.A @ y


    def exact(self, t):
        exponential = scipy.linalg.expm(t*self.A)
        return exponential @ self.y0

class VanderPol(IVP):
    def __init__(self, t0, T, y0, mu):
        super().__init__(t0, T, y0)
        self.mu = mu
    def f(self, t, y):
        return np.array([y[1],self.mu*(1-y[0]**2) * y[1]-y[0]])

class LotkaVolterra(IVP):
    def __init__(self, t0, T, y0, a, b, c, d):
        super().__init__(t0, T, y0)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def f(self, t, u):
        x = u[0]
        y = u[1]
        dxdt = self.a * x - self.b * x * y
        dydt = self.c * x * y - self.d * y
        return np.array([dxdt, dydt])

class Solver:
    def __init__(self, ivp, N: int, tol=1e-10, adaptive = False):
        self.ivp = ivp
        self.N = N
        self.u = ivp.y0
        self.grid = np.linspace(ivp.t0, ivp.T, N + 1)
        self.h = (ivp.T - ivp.t0) / N
        self.us = None
        self.tol = tol
        self.adaptive = adaptive

    def step(self, tn, un):
        raise NotImplementedError("step() not implemented in subclass")

    def integrate(self):
        if not self.adaptive:
            # fixed-step branch
            m = len(self.u)
            us = np.zeros((m, self.N + 1))
            us[:, 0] = self.u
            for i in range(self.N):
                tn = self.grid[i]
                un = us[:, i]
                us[:, i + 1] = self.step(tn, un)
            self.us = us
            return self.grid, self.us

        # adaptive branch
        T = self.ivp.T
        tn = self.ivp.t0
        un = self.u.copy()
        grid = [tn]
        us_list = [un.copy()]

        k = self.k
        f0 = self.ivp.f(tn, un)
        self.h = abs(T - tn) * (self.tol ** (1.0 / k)) / (100 * (1.0 + np.linalg.norm(f0)))

        while tn < T:
            un = self.step(tn, un)
            tn += self.h
            grid.append(tn)
            us_list.append(un)
            self.h = self.newstep()
            if tn + self.h > T:
                self.h = T - tn

        self.grid = np.array(grid)
        self.us = np.column_stack(us_list)
        return self.grid, self.us

    def newstep(self):
        raise NotImplementedError("newstep() not implemented in subclass")
    

class ExplicitEuler(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)

    def step(self, tn, un):
        return un + self.h * self.ivp.f(tn, un)


class TrapezoidalRule(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)

    def step(self, tn, un):
        A = self.ivp.A
        m = len(un)
        I = np.eye(m)

        # Calculate f(t_n, u_n)
        fn = self.ivp.f(tn, un)

        # Solve the system
        lhs = I - (self.h / 2) * A
        rhs = un + (self.h / 2) * fn
        un1 = scipy.linalg.solve(lhs, rhs)
        return un1


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
    
