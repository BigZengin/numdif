import numpy as np
import scipy.linalg
from Main import *

class RK4(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)
    
    def step(self, tn, un):
        Y1 = self.ivp.f(tn, un)
        Y2 = self.ivp.f(tn + self.h/2, un + self.h * Y1/2)
        Y3 = self.ivp.f(tn + self.h/2, un + self.h * Y2/2)
        Y4 = self.ivp.f(tn + self.h, un + self.h * Y3)
        return un + self.h/6 * (Y1 + 2*Y2 + 2*Y3 + Y4)

class RK34(Solver):
    def __init__(self, ivp, N, tol=1e-10, adaptive = False):
        super().__init__(ivp, N, tol=1e-10, adaptive=adaptive)
        self.errest = None
        self.errestold = None
        self.tol = tol
        self.rn = self.tol
        self.rnold = self.tol
        self.k = 4

    def step(self, tn, un):
        Y1 = self.ivp.f(tn, un)
        Y2 = self.ivp.f(tn + self.h/2, un + self.h * Y1/2)
        Y3 = self.ivp.f(tn + self.h/2, un + self.h * Y2/2)
        Z3 = self.ivp.f(tn + self.h, un - self.h * Y1 + 2*self.h*Y2)
        Y4 = self.ivp.f(tn + self.h, un + self.h * Y3)
        unext = un + self.h/6 * (Y1 + 2*Y2 + 2*Y3 + Y4)

        self.errestold = self.errest
        self.rnold = self.rn
        self.errest = self.h/6 * (2*Y2 + Z3 - 2*Y3 - Y4)
        self.rn = np.linalg.norm(self.errest)

        return unext

    def newstep(self):
        # PI controller for next step
        hnext = (self.tol / self.rn)**(2/(3*self.k)) * (self.tol / self.rnold)**(-1/(3*self.k)) * self.h
        return hnext
