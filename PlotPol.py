# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 09:03:47 2025
@author: diaso
"""
from  numpy import *
import matplotlib.pyplot as plt
from Main import *
from proj1 import *
#Initial case μ=100
ivp = VanderPol(0, 200, np.array([2, 0]), 100)
solver = RK34(ivp, 1, adaptive=True)
solver.integrate()

#First graph
plt.figure()
plt.plot(solver.grid, solver.us[1, :], label="y2")
plt.xlabel("t")
plt.ylabel("y2")
plt.title("RK34 adaptive approximation of Van der Pol equation μ = 100 ")
plt.legend()
plt.semilogy(solver.grid[:-1],np.diff(solver.grid))

#Second graph
plt.figure()
plt.plot(solver.us[0, :],solver.us[1, :], label="y2")
plt.xlabel("y1")
plt.ylabel("y2")
plt.title("y1 vs y2 Van der Pol equation μ = 100")
plt.legend()


#Different μs
mus=np.array([10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000])
N_steps = []
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))

for m in mus:
    ivp = VanderPol(0, 0.7*m, np.array([2, 0]), m)
    solver = RK34(ivp, 1, adaptive=True, tol=1e-10)
    solver.integrate()
    N_steps.append(shape(solver.us)[1])
    
    # y2 with respect of the time
    ax1.plot(solver.grid, solver.us[1, :], label=f"μ = {m}")
    
    # y2 with respect of y1
    ax2.plot(solver.us[0, :], solver.us[1, :], label=f"μ = {m}")

print(N_steps)
# First graph
ax1.set_xlabel("t")
ax1.set_ylabel("y2")
ax1.set_title("RK34 adaptive: y2(t) for different μ")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Second graph
ax2.set_xlabel("y1")
ax2.set_ylabel("y2")
ax2.set_title("RK34 adaptive: Phase portrait for different μ")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')
plt.show()

#Step size
plt.figure(figsize=(10, 6))
plt.loglog(mus, N_steps, 'o-', linewidth=2, markersize=8, label='Datos')
plt.xlabel('μ', fontsize=12)
plt.ylabel('Number of steps N', fontsize=12)
plt.title('Number of steps vs. parameter μ(Van der Pol)', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

#Differents initial values
iv= np.array([[0,0],[2,0],[1,1],[0,2],[3,5]])
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))

for i in range(iv.shape[0]):
    y0 = iv[i, :]
    ivp = VanderPol(0, 20, y0, 10)
    solver = RK34(ivp, 1, adaptive=True)
    solver.integrate()
    
    # y2 with respect of the time
    ax1.plot(solver.grid, solver.us[1, :], label=f"y0 = {y0}")
    
    # y2 with respect of y1
    ax2.plot(solver.us[0, :], solver.us[1, :], label=f"yo = {y0}")

# First graph
ax1.set_xlabel("t")
ax1.set_ylabel("y2")
ax1.set_title("RK34 adaptive: y2(t) for different y0")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Second graph
ax2.set_xlabel("y1")
ax2.set_ylabel("y2")
ax2.set_title("RK34 adaptive: Phase portrait for different y0")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.show()