import numpy as np
import math
from scipy.optimize import minimize, NonlinearConstraint
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Middle-Distance Running Simulation — Slide 16 Optimization
# Objective: Find the optimal parameters (V0, A, w, T) that minimize race time T
# while satisfying physiological and mechanical constraints
# ---------------------------------------------------------------

# ------------------------
# MODEL DEFINITIONS
# ------------------------

def v(t, V0, A, w):
    """
    Instantaneous velocity function at time t.
    Modeled as a sinusoidal oscillation around a base speed V0.
    """
    return V0 + A * np.sin(w * t)

def f(A, V0, t, w):
    """
    Computes the absolute value of the propulsive force at time t.
    The expression varies depending on the segment of the race.
    """
    a = V0 * t + (A / w) * np.cos(w * t) - (A / w)  # primitive of v(t)

    if 0 <= a <= 1500:  # race interval [0m, 1500m]
        # If in linear zones (every 100m interval)
        if any(lower <= a < lower + 100 for lower in range(0, 1500, 200)):
            return abs(81.3333 * A * w * np.cos(w * t) + 0.16856 * v(t, V0, A, w)**2)
        else:
            term = 81.3333 * A * w * np.cos(w * t) + 0.16856 * v(t, V0, A, w)**2
            return abs(np.sqrt(3.6725 * v(t, V0, A, w)**4 + term**2) / 2)
    else:
        return 0  # Outside of the race interval

def sigma(e):
    """
    Aerobic power function depending on anaerobic energy e.
    Piecewise-defined, continuous by construction.
    """
    if 0 <= e < 401.25:
        return 19.37 + 0.066679 * e
    elif 401.25 <= e <= 668.75:
        return 22.05
    elif 668.75 < e <= 1337.5:
        return -0.024479 * e + 39.37
    else:
        return 5.691577  # Outside modeled range

def eta(A, w, t):
    """
    Recovery term (mechanical dissipation when force is negative).
    Only applies when cosine is negative.
    """
    cn = 4
    return cn * (A**2) * (w**2) * (np.cos(w * t))**2 if np.cos(w * t) <= 0 else 0

def e(sigma, eta, f, v, A, w, T):
    """
    Computes the anaerobic energy e(t) over time using forward Euler integration.
    """
    e0 = 1337.5  # Initial anaerobic energy (in J/kg)
    m = 61       # Athlete's mass in kg
    N = 10000
    temps = np.linspace(0, T, N)
    delta_t = T / N

    e_values = np.zeros(N + 1)
    e_values[0] = e0

    for i in range(N):
        power = f(A, V0, temps[i], w) * v(temps[i], V0, A, w)
        recovery = eta(A, w, temps[i])
        e_values[i + 1] = e_values[i] + delta_t * (sigma(e_values[i]) - power - recovery) / m

    return temps, e_values

def deriveef(f, A, V0, delta_t, t, v):
    """
    Numerical approximation of the derivative of the force over time (df/dt).
    """
    return (f(A, V0, t + delta_t, v) - f(A, V0, t, v)) / delta_t

# ------------------------
# OPTIMIZATION TARGET & CONSTRAINTS
# ------------------------

def objective(x):
    """
    Objective function to minimize: total race time T.
    """
    V0, A, w, T = x
    return T

# Constraints to ensure physically meaningful and safe solutions
def constraint1(x): return x[1]               # A > 0
def constraint2(x): return x[0] - x[1]        # V0 > A
def constraint3(x): return x[0]               # V0 > 0
def constraint4(x):                            # Total distance must be 1500m
    V0, A, w, T = x
    return V0 * T + (A / w) * np.cos(w * T) + (A / w) - 1500
def constraint5(x): return x[3]               # T > 0

def constraint6(x, sigma, eta, f, v):
    """
    Ensures anaerobic energy e(t) stays non-negative during race.
    """
    _, e_values = e(sigma, eta, f, v, x[1], x[2], x[3])
    return np.min(e_values)

def fliste(A, V0, v, f, T):
    """
    Compute force values over time for constraint7.
    """
    t_values = np.linspace(0, T, 100)
    return [f(A, V0, t, v) for t in t_values]

def constraint7(x, f, v):
    """
    Ensures that the peak propulsive force is below 578.6 N.
    """
    A, V0, w, T = x[1], x[0], x[2], x[3]
    f_values = fliste(A, V0, v, f, T)
    return 578.6 - np.max(f_values)

def constraint8(x, delta_t, v):
    """
    Ensures that the rate of force increase df/dt stays under 186.3 N/s.
    """
    V0, A, w, T = x
    t_values = np.linspace(0, T, 10000)
    derivatives = [deriveef(f, A, V0, delta_t, t, v) for t in t_values]
    return 186.3 - np.max(derivatives)

# ------------------------
# OPTIMIZATION EXECUTION
# ------------------------

x0 = [7.1, 1.0, 200.1, 1.0]  # Initial guess: [V0, A, w, T]

constraints = [
    {'type': 'ineq', 'fun': constraint1},
    {'type': 'ineq', 'fun': constraint2},
    {'type': 'ineq', 'fun': constraint3},
    {'type': 'ineq', 'fun': constraint4},
    {'type': 'ineq', 'fun': constraint5},
    NonlinearConstraint(lambda x: constraint6(x, sigma, eta, f, v), 0, np.inf),
    NonlinearConstraint(lambda x: constraint7(x, f, v), 0, np.inf),
    NonlinearConstraint(lambda x: constraint8(x, 1e-3, v), 0, np.inf)
]

# Launch optimization
result = minimize(objective, x0, constraints=constraints, method='SLSQP')

# Display result
print("Résultat de l’optimisation :")
print("V0 =", result.x[0])
print("A  =", result.x[1])
print("w  =", result.x[2])
print("T  =", result.x[3])
