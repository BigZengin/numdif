# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 12:45:38 2025
@author: diaso
"""
from  numpy import *
import matplotlib.pyplot as plt
from Main import *
from proj1 import *
from scipy.integrate import solve_ivp
import time

mus = np.array([10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000])
mus_extended = np.array([10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 
                         1000, 2000, 5000, 10000])

t0 = 0
T = 2000  # Tiempo largo para capturar el comportamiento
y0 = np.array([2.0, 0.0])

# Tolerancias
rtol = 1e-6
atol = 1e-9

print("="*50)
print("Comparison: BDF (implicit) vs RK45 (explicit)")
print("="*50)

# ============ EXPERIMENT 1: BDF ============
print("\n1. METHOD BDF (Backward Differentiation Formula - IMPLICIT)")
print("-"*50)

N_steps_bdf = []
times_bdf = []

for mu in mus:
    vdp = VanderPol(t0, T, y0, mu)
    
    start = time.time()
    sol = solve_ivp(
        vdp.f, 
        [t0, T], 
        y0, 
        method='BDF',
        rtol=rtol,
        atol=atol
    )
    elapsed = time.time() - start
    
    n_steps = len(sol.t)
    N_steps_bdf.append(n_steps)
    times_bdf.append(elapsed)
    print(f"μ = {mu:4d}: N = {n_steps:5d} step, time = {elapsed:.4f} s")

N_steps_bdf = np.array(N_steps_bdf)
times_bdf = np.array(times_bdf)

# ============ EXPERIMENT 2: RK45 ============
print("\n2. METHOD RK45 (Runge-Kutta - EXPLICIT)")
print("-"*50)

N_steps_rk45 = []
times_rk45 = []

for mu in mus:
    vdp = VanderPol(t0, T, y0, mu)
    
    start = time.time()
    sol = solve_ivp(
        vdp.f, 
        [t0, T], 
        y0, 
        method='RK45',
        rtol=rtol,
        atol=atol
    )
    elapsed = time.time() - start
    
    n_steps = len(sol.t)
    N_steps_rk45.append(n_steps)
    times_rk45.append(elapsed)
    print(f"μ = {mu:4d}: N = {n_steps:5d} stesps, time = {elapsed:.4f} s")

N_steps_rk45 = np.array(N_steps_rk45)
times_rk45 = np.array(times_rk45)

# ============ EXPERIMENT 3: BDF with huge μ ============
print("\n3. BDF WITH HUGE μ VALUES")
print("-"*50)

N_steps_bdf_ext = []
times_bdf_ext = []

for mu in mus_extended:
    vdp = VanderPol(t0, T, y0, mu)
    
    start = time.time()
    sol = solve_ivp(
        vdp.f, 
        [t0, T], 
        y0, 
        method='BDF',
        rtol=rtol,
        atol=atol
    )
    elapsed = time.time() - start
    
    n_steps = len(sol.t)
    N_steps_bdf_ext.append(n_steps)
    times_bdf_ext.append(elapsed)
    print(f"μ = {mu:5d}: N = {n_steps:5d} steps, time = {elapsed:.4f} s")

N_steps_bdf_ext = np.array(N_steps_bdf_ext)
times_bdf_ext = np.array(times_bdf_ext)

# ============ Graphs ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Number of steps vs μ (log-log)
ax = axes[0, 0]
ax.loglog(mus, N_steps_bdf, 'o-', linewidth=2, markersize=8, 
          label='BDF (implicit)', color='blue')
ax.loglog(mus, N_steps_rk45, 's-', linewidth=2, markersize=8, 
          label='RK45 (explicit)', color='red')
ax.set_xlabel('μ', fontsize=11)
ax.set_ylabel('Number of steps N', fontsize=11)
ax.set_title('Comparison: Necessary steps', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10)

# 2. Time vs μ
ax = axes[0, 1]
ax.loglog(mus, times_bdf, 'o-', linewidth=2, markersize=8, 
          label='BDF', color='blue')
ax.loglog(mus, times_rk45, 's-', linewidth=2, markersize=8, 
          label='RK45', color='red')
ax.set_xlabel('μ', fontsize=11)
ax.set_ylabel('Time (s)', fontsize=11)
ax.set_title('Comparison: Time of compute', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10)

# 3. BDF extended (huge μ)
ax = axes[1, 0]
ax.loglog(mus_extended, N_steps_bdf_ext, 'o-', linewidth=2, markersize=8, 
          label='BDF', color='blue')
ax.set_xlabel('μ', fontsize=11)
ax.set_ylabel('Number of steps N', fontsize=11)
ax.set_title('BDF: μ until 10,000 (impossible for RK45!)', 
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10)
ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5, 
           label='Limit RK45')

# 4. Efficiency ratio
ax = axes[1, 1]
ratio_steps = N_steps_rk45 / N_steps_bdf
ax.semilogx(mus, ratio_steps, 'o-', linewidth=2, markersize=8, color='green')
ax.set_xlabel('μ', fontsize=11)
ax.set_ylabel('N_RK45 / N_BDF', fontsize=11)
ax.set_title('Adventage of BDF over RK45', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# ============ ANALYSIS ============
print("\n" + "="*50)
print("ANALYSIS OF RESULTS")
print("="*50)
print(f"\nFor μ = {mus[-1]}:")
print(f"  BDF:  {N_steps_bdf[-1]:5d} steps, {times_bdf[-1]:.4f} s")
print(f"  RK45: {N_steps_rk45[-1]:5d} steps, {times_rk45[-1]:.4f} s")
print(f"  Step rate: {N_steps_rk45[-1]/N_steps_bdf[-1]:.2f}x")
print(f"\nFor μ = 10000 (only BDF):")
print(f"  BDF:  {N_steps_bdf_ext[-1]:5d} steps, {times_bdf_ext[-1]:.4f} s")
print("\n¿Why BDF is better?")
print("  - IMPLICIT method → better stability for stiff problems")
print("  - It has a much bigger stability region")
print("  - Adaptative order → additional efficiency")
print("="*50)