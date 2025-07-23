import numpy as np
import math
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# FIGURE 7 — Evolution of Anaerobic Power e(t) for Various Effort Levels
# This simulation models the mass-specific anaerobic energy reserve over time
# under different percentages of mechanical force applied during the race.
# --------------------------------------------------------------

# ----------------------------
# Physiological Power Function
# ----------------------------

def sigma(e):
    """
    Models the aerobic power sigma(e) as a piecewise linear function of anaerobic energy e.
    Based on physiological modeling (Aftalion et al.).
    """
    if 0 <= e <= 1337:
        if e <= 401.25:
            return 19.37 + 0.066679 * e
        elif e <= 668.75:
            return 22.05
        else:
            return 38.42 - 0.024479 * e
    elif e < 0:
        return 19.37
    else:
        return 5.691577

# ----------------------------
# Velocity and Position Models
# ----------------------------

def v(t, V0, A, w):
    """Velocity profile: sinusoidal modulation over base velocity V0"""
    return V0 + A * np.sin(w * t)

def x(t, V0, A, w):
    """Integrated position function from v(t)"""
    return V0 * t + (A / w) * np.cos(w * t) - (A / w)

# ----------------------------
# Force Function f(t)
# ----------------------------

def f_generic(A, w, V0, t, v, z):
    """
    Calculates the total mechanical force applied at time t under effort percentage z.
    Alternates between high resistance and reduced effort zones every 100m.
    """
    a = x(t, V0, A, w)

    if 0 <= a <= 1500:
        # Zones of reduced mechanical resistance every 100m
        in_low_force_zone = any([
            0 <= a < 100, 200 <= a < 300, 400 <= a < 500,
            600 <= a < 700, 800 <= a < 900, 1000 <= a < 1100,
            1200 <= a < 1300, 1400 <= a < 1500
        ])

        if in_low_force_zone:
            f = (81.3333 * A * w * np.cos(w * t) + 0.16856 * v(t, V0, A, w)**2) * z
        else:
            f = np.sqrt(
                3.6725 * v(t, V0, A, w)**4 +
                (z * (81.3333 * A * w * np.cos(w * t) + 0.16856 * v(t, V0, A, w)**2))**2
            )
    else:
        f = 0

    return abs(f)

# Different force functions (percentage of max effort)
def f1(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=1.00)     # 100%
def f2(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=0.92)     # 92%
def f3(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=0.85)     # 85%
def f4(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=0.95)     # 95%
def f5(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=0.953125) # 95.31%
def f6(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=0.50)     # 50%

# ----------------------------
# Anaerobic Power Dissipation Term
# ----------------------------

def eta(t, A, w):
    """
    Models the dissipation of anaerobic energy due to internal force generation.
    Only active during deceleration phases (cos(wt) < 0).
    """
    cn = 4
    if math.cos(w * t) > 0:
        return 0
    else:
        return cn * (A**2) * (w**2) * (np.cos(w * t))**2

# ----------------------------
# Anaerobic Power Function e(t)
# ----------------------------

def e(sigma, eta, f, v, V0, A, w, T):
    """
    Simulates the anaerobic energy curve e(t) over time using Euler's method.
    Initial anaerobic energy e0 = 1337 J/kg.
    """
    e0 = 1337
    m = 61                      # Runner's mass (kg)
    N = 10000                   # Time discretization
    delta_t = T / N
    temps = np.linspace(0, T, N)
    e_values = np.zeros(N + 1)
    e_values[0] = e0
    r = 1                       # Scaling factor

    for i in range(N):
        e_values[i+1] = e_values[i] + delta_t * (
            sigma(e_values[i]) + eta(temps[i], A, w) - r * f(A, w, V0, temps[i], v) * v(temps[i], V0, A, w)
        ) / m

    return temps, e_values[:-1]  # Return times and e(t) values (aligned)

# ----------------------------
# Parameters from Optimization
# ----------------------------

V0 = 6.756005867882389
A = 0.47233504752192025
w = 0.5000000000000093
T = 221.64651651000005

# ----------------------------
# Plotting e(t) for various effort levels
# ----------------------------

plt.xlabel("Temps en (T/10000)s")
plt.ylabel("e(t) en W/kg")
plt.title("Évolution de la puissance massique anaérobie en fonction du temps", size=10, color='C3')

# Plot multiple effort scenarios
plt.plot(*e(sigma, eta, f6, v, V0, A, w, T), label='50%')
plt.plot(*e(sigma, eta, f3, v, V0, A, w, T), label='85%')
plt.plot(*e(sigma, eta, f2, v, V0, A, w, T), label='92%')
plt.plot(*e(sigma, eta, f4, v, V0, A, w, T), label='95%')
plt.plot(*e(sigma, eta, f5, v, V0, A, w, T), label='95.31%')
plt.plot(*e(sigma, eta, f1, v, V0, A, w, T), label='100%')

plt.legend()
plt.show()
