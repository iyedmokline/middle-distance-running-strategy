# fig4_fig5_fig6_simulation.py
# Author: Mohamed Iyed Mokline
# Description:
# This script generates one of the following:
# - Figure 4: evolution of df/dt over time
# - Figure 5: df/dt versus f(t)
# - Figure 6: eta(t) over time
# Use by modifying the plotting section at the end to match the figure you need.

import matplotlib.pyplot as plt
import numpy as np
import math

# --- Friction force modeling ---
def eta(t, A, w):
    """
    Computes additional energy dissipation due to muscle inefficiency when cos(wt) < 0.
    This reflects phases of deceleration or braking.
    """
    cn = 4
    if math.cos(w * t) > 0:
        return 0
    else:
        return cn * (A**2) * (w**2) * (np.cos(w * t))**2

# --- Motion equations ---
def x(t, V0, A, w):
    """
    Position of the runner at time t based on sinusoidal velocity model.
    """
    return V0 * t - (A / w) * np.cos(w * t) + (A / w)

def v(t, V0, A, w):
    """
    Velocity of the runner at time t.
    """
    return V0 + A * np.sin(w * t)

# --- Force computation ---
def f(A, V0, t, w):
    """
    Computes the total force applied at time t, based on position-dependent aerodynamic and inertial resistance.
    """
    a = x(t, V0, A, w)

    if 0 <= a <= 1500:
        # Standard resistance zones (alternating every 100m)
        if (0 <= a < 100 or 200 <= a < 300 or 400 <= a < 500 or
            600 <= a < 700 or 800 <= a < 900 or 1000 <= a < 1100 or
            1200 <= a < 1300 or 1400 <= a < 1500):

            force = 81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w) ** 2)
        else:
            # Enhanced resistance zones
            force = np.sqrt(
                3.6725 * (v(t, V0, A, w) ** 4) +
                (81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w) ** 2)) ** 2
            ) / 2
    else:
        force = 0

    return abs(force)

# --- Derivative of force with respect to time ---
def derivative_f(A, V0, delta_t, t, w):
    """
    Approximates df/dt at time t using finite differences.
    """
    return (f(A, V0, t + delta_t, w) - f(A, V0, t, w)) / delta_t

# --- Parameters from optimal strategy (Slide 16) ---
delta_t = 1e-8
V0 = 6.764650588808889
A = 0.4723350215209585
w = 0.4973095108162272
T = 221.64516551008255

# --- Time vector for simulation ---
t_values = [0.01 * i for i in range(int(100 * T))]

# --- Simulation targets: choose what to plot ---
# Option 1: df/dt over time (→ Figure 4)
dfdt_values = [derivative_f(A, V0, delta_t, t, w) for t in t_values]

# Option 2: eta(t) over time (→ Figure 6)
# eta_values = [eta(t, A, w) for t in t_values]

# Option 3: df/dt as function of f(t) (→ Figure 5)
# f_values = [f(A, V0, t, w) for t in t_values]
# dfdt_values = [derivative_f(A, V0, delta_t, t, w) for t in t_values]
# plt.plot(f_values, dfdt_values)

# --- Plot (default: Figure 4) ---
plt.plot(t_values, dfdt_values)

# French labels for consistency with paper
plt.xlabel("Temps en s")             # Time in seconds
plt.ylabel("df/dt en N/s")           # Derivative of force in N/s
plt.title("Évolution de la dérivée de f en fonction du temps", size=10, color='C3')
plt.grid(True)
plt.tight_layout()
plt.show()
