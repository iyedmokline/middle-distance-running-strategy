# fig5_simulation.py
# Author: Mohamed Iyed Mokline
# Description: Generates Figure 5 — Plots the derivative of force with respect to force (df/dt vs f),
#              based on a position- and speed-dependent force model used in the 1500m strategy simulation.

import numpy as np
import matplotlib.pyplot as plt

# --- Motion functions (position and velocity) ---

def x(t, V0, A, w):
    """
    Position function based on sinusoidal velocity integration.
    Returns the runner's position at time t.
    """
    return V0 * t - (A / w) * np.cos(w * t) + (A / w)

def v(t, V0, A, w):
    """
    Velocity function: sinusoidal fluctuation around V0.
    """
    return V0 + A * np.sin(w * t)

# --- Force function ---

def f(A, w, V0, t, v):
    """
    Computes the total force applied at time t based on:
    - Propulsive component (cosine modulation)
    - Aerodynamic drag (proportional to v^2)
    - Ground contact (modulated by runner's position x(t))
    """
    a = x(t, V0, A, w)

    if 0 <= a <= 1500:  # Only compute if runner is within 1500m
        # Zones with standard resistance (even 100m intervals)
        if (0 <= a < 100 or 200 <= a < 300 or 400 <= a < 500 or
            600 <= a < 700 or 800 <= a < 900 or 1000 <= a < 1100 or
            1200 <= a < 1300 or 1400 <= a <= 1500):

            f_value = 81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w) ** 2)
        else:
            # Zones with additional resistance → more complex model
            f_value = np.sqrt(
                3.6725 * (v(t, V0, A, w) ** 4) +
                (81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w) ** 2)) ** 2
            )
    else:
        f_value = 0

    return abs(f_value)

# --- Numerical derivative of force ---

def deriveef(f, A, w, V0, delta_t, t, v):
    """
    Approximates the derivative of f(t) using finite difference.
    df/dt ≈ [f(t + Δt) - f(t)] / Δt
    """
    return (f(A, w, V0, t + delta_t, v) - f(A, w, V0, t, v)) / delta_t

# --- Parameters obtained from optimization (Slide 16) ---
V0 = 6.760500678882389       # Mean speed (m/s)
A = 0.47323504752192025      # Amplitude of speed oscillation
w = 0.5000000000000093       # Frequency of oscillation
T = 221.64651651000005       # Total duration (s)
delta_t = 1e-7               # Time step for numerical derivative

# --- Time vector (high resolution) ---
t_values = np.linspace(0, T, 100000)

# --- Compute force and its derivative ---
f_values = np.array([f(A, w, V0, t, v) for t in t_values])
df_dt_values = np.array([deriveef(f, A, w, V0, delta_t, t, v) for t in t_values])

# --- Plot df/dt as a function of f(t) ---
plt.plot(f_values, df_dt_values, color="black")
plt.xlabel("f(t) en N")
plt.ylabel("df/dt en N/s")
plt.title("Courbe de la dérivée de f en fonction de f")
plt.grid(True)
plt.tight_layout()
plt.show()
