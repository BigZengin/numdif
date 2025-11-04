import numpy as np
import scipy as sp
import scipy . linalg
import matplotlib . pyplot as plt
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


A = np.array([[1]])
problem = LTE(0, 1, y0=np.array([2]), A=A)
ts = np.linspace(0, 1)
ys = [problem.exact(t) for t in ts]
plt.plot(ts, ys)
