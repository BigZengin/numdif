import numpy as np
from typing import Callable
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def p_sl_simple(x):
    return 1.0

def q_sl_simple(x):
    return 0.0

class BC(object):
    def __init__(self, value: float):
        self.value = value


class Dirichlet(BC):
    def __init__(self, value):
        BC.__init__(self, value)


class Neumann(BC):
    def __init__(self, value):
        BC.__init__(self, value)


def FDMSturmLiouville(L: float, BCleft: BC, BCright: BC, p:Callable, q:Callable, N: int = 100):
    """
    Compute the FDM discretization of a Sturm-Liouville eigenvalue problem (with funtion p(x) nonlinear and q(x)) on
    the interval [0,L] given the boundary conditions BCleft at x=0 and BCright at x=L, using N computational points
    :param L: float
    :param BCleft: BC object(e.g. Dirichlet, Neumann)
    :param BCright: BC object(e.g. Dirichlet, Neumann)
    :param N: integer > 0
    :return dx : float, grid spacing
    :return grid : np.array with shape (N ,) , the computational points
    :return A : np.array or scipy.sparse matrix with shape (N,N)
    :return BCcorr : np.array with shape (N, 1) or (N, ), containing boundary condition correction terms
    """
    dx = L / (N + 1)
    grid = np.linspace(dx, L - dx, N)

    x_minus = grid - dx / 2
    x_plus = grid + dx / 2
    p_minus = np.array([p(x) for x in x_minus])  # p_{i-1/2}
    p_plus = np.array([p(x) for x in x_plus])  # p_{i+1/2}
    q_vals = np.array([q(x) for x in grid])

    main = -(p_minus + p_plus) / (dx**2) + q_vals
    lower = p_minus[1:] / (dx ** 2)
    upper = p_plus[:-1] / (dx ** 2)
    A = diags([lower, main, upper], [-1, 0, 1], format="csr")
    A = A.tolil()
    BCcorr = np.zeros(N)

    # Left boundary
    if isinstance(BCleft, Dirichlet):
        BCcorr[0] -= p_minus[0] * BCleft.value / (dx ** 2)
    elif isinstance(BCleft, Neumann):
        A[0, 0] = -(p_plus[0]) / (dx ** 2) + q_vals[0]
        BCcorr[0] = -p_minus[0] * BCleft.value / dx

    # Right boundary
    if isinstance(BCright, Dirichlet):
        BCcorr[-1] = -p_plus[-1] * BCright.value / (dx ** 2)
    elif isinstance(BCright, Neumann):
        A[-1, -1] = -(p_minus[-1]) / (dx ** 2) + q_vals[-1]
        BCcorr[-1] = p_plus[-1] * BCright.value / dx
    A = A.tocsr()
    return dx, grid, A, BCcorr


def solve_sturm_liouville(L: float, BCleft: BC, BCright: BC, p: Callable, q: Callable, N: int = 100, num_eigenvalues: int = 5):
    dx, grid, A, BCcorr = FDMSturmLiouville(L, BCleft, BCright, p, q, N)
    # We need to implement BCcorr if u(0)=!0 or u'(0)=!0
    eigenvalues, eigenvectors = eigsh(-A, k=num_eigenvalues, which='SM')
    # Normalize eigenvectors
    for i in range(eigenvectors.shape[1]):
        norm = np.linalg.norm(eigenvectors[:, i]) * np.sqrt(dx)
        eigenvectors[:, i] /= norm

    return eigenvalues, eigenvectors, grid

L_val = 1.0
N_points = 1000
k = 3

# Boundary conditions: u(0) = 0, u'(1) = 0
BC_left = Dirichlet(value=0)
BC_right = Neumann(value=0)

eigenvalues, eigenvectors, grid = solve_sturm_liouville(L_val, BC_left, BC_right, p_sl_simple, q_sl_simple, N_points,k)

# Analytic eigenvalues (lambda_k = -((2k-1)pi/2)^2)
k_values = np.arange(1, k + 1)
lambda_analytic = ((2 * k_values - 1) * np.pi / 2)**2

print(f"--- FDM solutions for u''=λu, u(0)=0, u'(1)=0 (N={N_points}) ---")

# Show eigenvalues
results = np.array([lambda_analytic, eigenvalues, np.abs(lambda_analytic - eigenvalues)])
results_table = np.vstack((lambda_analytic, eigenvalues, np.abs(lambda_analytic - eigenvalues))).T

print("\n| k |Analytic eigenvalue (λk) | FDM eigenvalue (λk) | Global error |")
print("|---|--------------------------|--------------------|----------------|")
for k, row in enumerate(results_table):
    print(f"| {k+1} | {row[0]:22.6f} | {row[1]:18.6f} | {row[2]:14.2e} |")


#Plot module functions
plt.figure(figsize=(10, 6))
plt.suptitle('First Three Eigenfunctions for u´´=λu with u(0)=0, u´(1)=0')

# The number of eigenfunctions to plot is k
num_plots = eigenvectors.shape[1]

for i in range(num_plots):
    # Plot the i-th eigenfunction (i+1 is the mode number, k)
    plt.plot(grid, eigenvectors[:, i], label=f'Mode k={i+1} ( λ={eigenvalues[i]:.2f})')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid(True, linestyle='--')
plt.legend(loc='lower left')
plt.show()

# --- Setup for Accuracy Plotting ---
N_range = np.logspace(1.5, 3.17, 10, dtype=int) # N from ~31 to ~3162
N_range = N_range[N_range % 2 != 0]  # Ensure N is odd (N+1 is even) for cleaner discretization
k_max = 3
lambda_true = -((2 * np.arange(1, k_max + 1) - 1) * np.pi / 2) ** 2

errors = np.zeros((k_max, len(N_range)))

for idx, N in enumerate(N_range):
    # Solve the problem for the current N
    lambda_N, _, _ = solve_sturm_liouville(L=1.0, BCleft=Dirichlet(0), BCright=Neumann(0),
                                           p=p_sl_simple, q=q_sl_simple, N=N, num_eigenvalues=k_max)

    # Calculate absolute error for the first k_max eigenvalues
    errors[:, idx] = np.abs(lambda_N - lambda_true)

# --- Plotting ---
plt.figure(figsize=(9, 6))

# Calculate the reference slope line: y = m * log(N) + b, where m = -2
N_ref = N_range[[0, -1]]
E_ref = errors[0, 0] * (N_ref[0] / N_ref) ** 2  # Reference line starting at the first error point and scaling by N^-2

for k in range(k_max):
    plt.loglog(N_range, errors[k, :], 'o-', label=f'$k={k + 1}$')

# Plot the reference line with slope -2
plt.loglog(N_ref, E_ref, 'k--', label='Reference Slope: $-2$')

plt.xlabel('Number of Interior Points, N (log scale)')
plt.ylabel('Absolute Error, |       λx - \λ|(log scale)')
plt.title('Log-Log Plot of Error vs. $N$ for Sturm-Liouville Eigenvalues')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()