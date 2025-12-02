import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
from typing import Callable
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# -------------------------
# BC classes
# -------------------------
class BC(object):
    def __init__(self, value):
        self.value = value

class Dirichlet(BC):
    def __init__(self, value):
        super().__init__(value)

class Neumann(BC):
    def __init__(self, value):
        super().__init__(value)



# -------------------------
# FDM second derivative
# -------------------------
def FDMd2dx2(L, BCleft, BCright, N=100):
    dx = L / (N + 1)
    grid = np.linspace(dx, L - dx, N)

    main = -2.0 * np.ones(N)
    off  =  1.0 * np.ones(N - 1)
    A = diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    A /= dx**2

    A = A.tolil()

    BCcorr = np.zeros(N)

    # Left boundary
    if isinstance(BCleft, Dirichlet):
        BCcorr[0] += BCleft.value / dx**2

    elif isinstance(BCleft, Neumann):
        A[0, 0] = -1.0 / dx**2
        A[0, 1] =  1.0 / dx**2
        BCcorr[0] += -BCleft.value / dx 

    # Right boundary
    if isinstance(BCright, Dirichlet):
        BCcorr[-1] += BCright.value / dx**2

    elif isinstance(BCright, Neumann):
        A[-1, -2] =  1.0 / dx**2
        A[-1, -1] = -1.0 / dx**2
        BCcorr[-1] += BCright.value / dx  

    A = A.tocsr()
    return dx, grid, A, BCcorr
