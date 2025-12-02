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


    # -------------------------
# exact problem helpers
# -------------------------
def exact_sin(x):
    return np.sin(x)

def f_from_exact(x):
    # for exact = sin(x) -> f = y'' = -sin(x)
    return -np.sin(x)

# -------------------------
# Example: small run, show solution
# -------------------------
L = np.pi
dx, grid, A, BCcorr = FDMd2dx2(L, Dirichlet(0), Neumann(-1), N=100)
RHS = f_from_exact(grid) - BCcorr
u = spsolve(A, RHS)

# plot numerical vs exact (include boundaries)
left_val = 0.0
right_deriv = -1.0
# for plotting the Dirichlet+Neumann case: left Dirichlet=0, right boundary value is not Dirichlet.
# we show interior + left Dirichlet, and for right we reconstruct u_N and skip u(L)=... (no direct value).
plt.figure()
plt.plot(np.concatenate(([0.0], grid)), np.concatenate(([left_val], u)), 'o-', label='numerical (interior + left BC)')
# exact on full domain
xx = np.linspace(0, L, 301)
plt.plot(xx, exact_sin(xx), '-', label='exact sin(x)')
plt.legend()
plt.xlabel('x'); plt.ylabel('y'); plt.grid(True)
plt.title('Dirichlet (left) + Neumann (right) example')
plt.show()

# -------------------------
# Convergence test
# -------------------------
def convergence_test(L, BCleft, BCright, exact_fun, f_fun, Ns):
    dx_list = []
    err_list = []
    for N in Ns:
        dx, grid, A, BCcorr = FDMd2dx2(L, BCleft, BCright, N)
        RHS = f_fun(grid) - BCcorr
        u = spsolve(A, RHS)
        y_exact = exact_fun(grid)
        rms = np.sqrt(np.mean((u - y_exact)**2))
        dx_list.append(dx)
        err_list.append(rms)
    return np.array(dx_list), np.array(err_list)

Ns = [20, 40, 80, 160, 320, 640]

# Dirichlet + Dirichlet (use sin on [0,pi], both ends zero)
dx_dd, err_dd = convergence_test(L=np.pi,
                                 BCleft=Dirichlet(0),
                                 BCright=Dirichlet(0),
                                 exact_fun=exact_sin,
                                 f_fun=f_from_exact,
                                 Ns=Ns)

# Dirichlet (left) + Neumann (right) (exact sin: left y(0)=0, right y'(pi) = cos(pi) = -1)
dx_dn, err_dn = convergence_test(L=np.pi,
                                 BCleft=Dirichlet(0),
                                 BCright=Neumann(-1.0),
                                 exact_fun=exact_sin,
                                 f_fun=f_from_exact,
                                 Ns=Ns)

# plot errors
plt.figure()
plt.loglog(dx_dd, err_dd, 'o-', label='Dirichlet+Dirichlet')
plt.loglog(dx_dn, err_dn, 's-', label='Dirichlet+Neumann')

# reference slope 2 line (through first point of DD)
ref = err_dd[0] * (dx_dd / dx_dd[0])**2
plt.loglog(dx_dd, ref, '--', label='slope 2')

plt.xlabel('dx'); plt.ylabel('RMS error')
plt.grid(True, which='both')
plt.legend()
plt.title('Convergence (expect slope ≈ 2)')
plt.show()


    A = A.tocsr()
    return dx, grid, A, BCcorr
