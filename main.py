#Figure 3.................................................................................

import matplotlib.pyplot as plt
import numpy as np

def fc_affine(x, a, b):
    return a * x + b

sigma_max = 22.05
sigma_final = 19.37
sigma_rest = 5.68
energy_total = 1337.5
energy_phi = 0.5 * energy_total
energy_critical = 0.3 * energy_total

range1 = np.linspace(0, energy_critical, 100)
range2 = np.linspace(energy_critical, energy_phi, 100)
range3 = np.linspace(energy_phi, energy_total, 100)

sigma1 = fc_affine(range1, (sigma_max - sigma_final) / energy_critical, sigma_final)
sigma2 = fc_affine(range2, 0, sigma_max)
sigma3 = fc_affine(range3, -(sigma_max - sigma_rest) / (energy_total - energy_phi),
                   sigma_rest + energy_total * (sigma_max - sigma_rest) / (energy_total - energy_phi))

plt.plot(range1, sigma1, color='k')
plt.plot(range2, sigma2, color='k')
plt.plot(range3, sigma3, color='k')

plt.ylabel('Puissance aérobie en W/kg', color='C4')
plt.xlabel("Énergie anaérobie restante en J/kg", color='C4')

plt.fill_between(range1, sigma1, where=(range1 >= 0) & (range1 <= energy_critical), color='red', alpha=0.3)

plt.title("Évolution de la puissance aérobie en fonction de l'énergie anaérobie restante", size=10, color='C3')
plt.grid()
plt.show()

# Figure 5 ...............................................................................

import numpy as np
import matplotlib.pyplot as plt

# Position function
def x(t, V0, A, w):
    return V0 * t - (A / w) * np.cos(w * t) + (A / w)

# Speed function
def v(t, V0, A, w):
    return V0 + A * np.sin(w * t)

# Force function
def f(A, w, V0, t, v):
    a = x(t, V0, A, w)
    if 0 <= a <= 1500:
        if (0 <= a < 100 or 200 <= a < 300 or 400 <= a < 500 or 600 <= a < 700 or
            800 <= a < 900 or 1000 <= a < 1100 or 1200 <= a < 1300 or 1400 <= a <= 1500):
            f_value = 81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w) ** 2)
        else:
            f_value = np.sqrt(3.6725 * (v(t, V0, A, w) ** 4) +
                              (81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w) ** 2)) ** 2)
    else:
        f_value = 0
    return abs(f_value)

# Derivative of f with respect to time
def deriveef(f, A, w, V0, delta_t, t, v):
    return (f(A, w, V0, t + delta_t, v) - f(A, w, V0, t, v)) / delta_t

# Parameters
V0 = 6.760500678882389
A = 0.47323504752192025
w = 0.5000000000000093
T = 221.64651651000005
delta_t = 0.0000001

t_values = np.linspace(0, T, 100000)

# Compute force and its derivative
f_values = np.array([f(A, w, V0, t, v) for t in t_values])
df_dt_values = np.array([deriveef(f, A, w, V0, delta_t, t, v) for t in t_values])

# Plot
plt.plot(f_values, df_dt_values)
plt.xlabel("f(t) en N")
plt.ylabel("df/dt en N/s")
plt.title("Courbe de la dérivée de f en fonction de f")
plt.grid(True)
plt.show()

#Figure 4, 5 & 6..........................................................................

import matplotlib.pyplot as plt
import math
import numpy as np

def eta(t, A, w):
    cn = 4
    if math.cos(w * t) > 0:
        eta = 0
    else:
        eta = cn * (A**2) * (w**2) * (np.cos(w * t))**2
    return eta

def x(t, V0, A, w):
    return V0 * t - (A / w) * np.cos(w * t) + (A / w)

def v(t, V0, A, w):
    return V0 + A * np.sin(w * t)

def f(A, V0, t, w):
    a = x(t, V0, A, w)
    if (0 <= a <= 1500):
        if 0 <= a < 100 or 200 <= a < 300 or 400 <= a < 500 or 600 <= a < 700 or 800 <= a < 900 \
            or 1000 <= a < 1100 or 1200 <= a < 1300 or 1400 <= a < 1500:
            f = 81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w)**2)
        else:
            f = (np.sqrt(3.6725 * (v(t, V0, A, w)**4) +
                         (81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w)**2))**2)) / 2
    else:
        f = 0
    return abs(f)

def derivative_f(A, V0, delta_t, t, v):
    return (f(A, V0, t + delta_t, v) - f(A, V0, t, v)) / delta_t

delta_t = 10**(-8)
V0 = 6.764650588808889
A = 0.4723350215209585
w = 0.4973095108162272
T = 221.64516551008255

t = [0.01 * i for i in range(int(100 * T))]
L = [derivative_f(A, V0, delta_t, t[i], w) for i in range(int(100 * T))]

plt.plot(t, L)
# → each time I change this line to display the corresponding curve

plt.xlabel("Temps en s")
#Time in seconds

plt.ylabel("df/dt en N/s")
# → "df/dt in N/s"

plt.title("Évolution de la dérivée de f en fonction du temps", size=10, color='C3')
#Evolution of the derivative of f as a function of time

plt.show()

# Slide 16 ...............................................................................

import numpy as np
import math
from scipy.optimize import minimize, NonlinearConstraint
import matplotlib.pyplot as plt

def v(t, V0, A, w):  # la vitesse à l’instant t → speed at time t
    return V0 + A * np.sin(w * t)

def f(A, V0, t, w):
    # absolute value of propulsive force at time t
    a = V0 * t + (A / w) * np.cos(w * t) - (A / w)
    if (0 <= a <= 1500):  # primitive of v(t) for circular paths
        if 0 <= a < 100 or 200 <= a < 300 or 400 <= a < 500 or 600 <= a < 700 or 800 <= a < 900 or 1000 <= a < 1100 or 1200 <= a < 1300 or 1400 <= a <= 1500:
            return abs(81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w)**2))
        else:
            return abs((np.sqrt(3.6725 * (v(t, V0, A, w)**4) + (81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w)**2))**2)) / 2)
    else:  # pour t<0 et t>T → for t<0 and t>T
        return 0

def sigma(e):  # aerobic power
    if (0 <= e < 401.25):
        return 19.37 + 0.066679 * e
    elif (401.25 <= e <= 668.75):
        return 22.05
    elif (668.75 < e <= 1337.5):
        return -0.024479 * e + 39.37
    else:
        return 5.691577  # sigma assumed continuous and constant outside study range

def eta(A, w, t):  # recovery term
    cn = 4
    if np.cos(w * t) > 0:
        return 0
    else:
        return cn * (A**2) * (w**2) * (np.cos(w * t))**2

def e(sigma, eta, f, v, A, w, T):  # anaerobic power
    e0 = 1337.5
    m = 61
    temps = np.linspace(0, T, 10000)
    delta_t = T / 100
    e_values = np.zeros(len(temps)+1)
    e_values[0] = e0
    for i in range(0, len(temps)):
        e_values[i+1] = e_values[i] + delta_t * (sigma(e_values[i]) - f(A, V0, temps[i], w) * v(temps[i], V0, A, w) - eta(A, w, temps[i])) / m
    return temps, e_values

def deriveef(f, A, V0, delta_t, t, v):
    # Rate of increase of force at time t
    return (f(A, V0, t + delta_t, v) - f(A, V0, t, v)) / delta_t

# --------------------------
# Main Program
# --------------------------

def objective(x):
    V0, A, w, T = x
    return T

def constraint1(x):  # A > 0
    V0, A, w, T = x
    return A

def constraint2(x):  # V0 > A
    V0, A, w, T = x
    return V0 - A

def constraint3(x):  # V0 > 0
    V0, A, w, T = x
    return V0

def constraint4(x):  # b = 1500 → V0 * T + (A / w) * np.cos(w * T) + (A / w)
    V0, A, w, T = x
    b = V0 * T + (A / w) * np.cos(w * T) + (A / w)
    return b - 1500

def constraint5(x):  # T > 0
    V0, A, w, T = x
    return T

def constraint6(x, sigma, eta, f, v):  # e(t) > 0 pour tout t tq 0 <= t <= T
    V0, A, w, T = x
    _, e_values = e(sigma, eta, f, v, A, w, T)
    return np.min(e_values)

def fliste(A, V0, v, f, T):
    t_values = np.linspace(0, T, 100)
    f_values = []
    for t in t_values:
        f_values.append(f(A, V0, t, v))
    return f_values

def constraint7(x, f, v):  # f(t) < 578.6
    V0, A, w, T = x
    f_values = fliste(A, V0, v, f, T)
    return 578.6 - np.max(f_values)

def constraint8(x, delta_t, v):  # df/dt < 186.3
    V0, A, w, T = x
    t_values = np.linspace(0, T, 10000)
    f_values = [deriveef(f, A, V0, delta_t, t, v) for t in t_values]
    return 186.3 - np.max(f_values)

x0 = [7.1, 1.0, 200.1, 1.0]  # [V0, A, w, T]

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

result = minimize(objective, x0, constraints=constraints, method='SLSQP')

print("Résultat de l’optimisation :")
print("V0 =", result.x[0])
print("A =", result.x[1])
print("w =", result.x[2])
print("T =", result.x[3])

#Figure 7 ................................................................................

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

def sigma(e):  # aerobic power
    if (0 <= e <= 1337):
        if (0 <= e <= 401.25):
            sigma = 19.37 + 0.066679 * e
        elif (401.25 < e <= 668.75):
            sigma = 22.05
        else:
            sigma = 38.42 - 0.024479 * e
    elif (e < 0):
        sigma = 19.37  # constant outside study interval
    else:
        sigma = 5.691577  # constant outside study interval
    return sigma

def v(t, V0, A, w):
    return V0 + A * np.sin(w * t)

def x(t, V0, A, w):
    return V0 * t + (A / w) * np.cos(w * t) - (A / w)

# Define 6 force functions f1 to f6 with different z values (correspond to % effort)

def f_generic(A, w, V0, t, v, z):
    a = x(t, V0, A, w)
    if (0 <= a <= 1500):
        if 0 <= a < 100 or 200 <= a < 300 or 400 <= a < 500 or 600 <= a < 700 or \
           800 <= a < 900 or 1000 <= a < 1100 or 1200 <= a < 1300 or 1400 <= a < 1500:
            f = (81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w)**2)) * z
        else:
            f = np.sqrt(3.6725 * (v(t, V0, A, w)**4) +
                        (z * (81.3333 * A * w * np.cos(w * t) + 0.16856 * (v(t, V0, A, w)**2)))**2)
    else:
        f = 0
    return abs(f)

def f1(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=1.0)      # 100%
def f2(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=0.92)     # 92%
def f3(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=0.85)     # 85%
def f4(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=0.95)     # 95%
def f5(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=0.953125) # 95.31%
def f6(A, w, V0, t, v): return f_generic(A, w, V0, t, v, z=0.50)     # 50%

def eta(t, A, w):
    cn = 4
    if math.cos(w * t) > 0:
        return 0
    else:
        return cn * (A**2) * (w**2) * (np.cos(w * t))**2

def e(sigma, eta, f, v, V0, A, w, T):
    e0 = 1337
    m = 61
    temps = np.linspace(0, T, 10000)
    delta_t = T / 10000
    e_values = np.zeros(len(temps) + 1)
    e_values[0] = e0
    r = 1  # scaling factor

    for i in range(len(temps)):
        e_values[i+1] = e_values[i] + delta_t * (
            sigma(e_values[i]) + eta(temps[i], A, w) - r * f(A, w, V0, temps[i], v) * v(temps[i], V0, A, w)
        ) / m

    return temps, e_values[:-1]  # drop last to match time length

# --- Main parameters
V0 = 6.756005867882389
A = 0.47233504752192025
w = 0.5000000000000093
T = 221.64651651000005

# --- Plotting
plt.xlabel("Temps en (T/10000)s")
plt.ylabel("e(t) en W/kg")
plt.title("Évolution de la puissance massique anaérobie en fonction du temps", size=10, color='C3')

plt.plot(*e(sigma, eta, f6, v, V0, A, w, T), label='50%')
plt.plot(*e(sigma, eta, f3, v, V0, A, w, T), label='85%')
plt.plot(*e(sigma, eta, f2, v, V0, A, w, T), label='92%')
plt.plot(*e(sigma, eta, f4, v, V0, A, w, T), label='95%')
plt.plot(*e(sigma, eta, f5, v, V0, A, w, T), label='95.31%')
plt.plot(*e(sigma, eta, f1, v, V0, A, w, T), label='100%')

plt.legend()
plt.show()

#Figure 8 ................................................................................

import numpy as np
import matplotlib.pyplot as plt

# Parameters
V0 = 6.756005867882389
A = 0.47323504752192025
w = 0.5000000000000093
T = 221.64651651000005

# Position function
def x(t, V0, A, w):
    return V0 * t - (A / w) * np.cos(w * t) + (A / w)

# Experimental time and position data
t1 = [0, 26.23, 56.89, 89.38, 119.65, 151.17, 182.92, 213.42, 232.91]
x1 = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1500]

time_uncertainty = 0.0058  # ± uncertainty on time

# Time and model-generated position data
t2 = np.linspace(0, T, 1000)
x2 = [x(t2[i], V0, A, w) for i in range(len(t2))]

# Plotting
plt.errorbar(t1, x1, xerr=time_uncertainty, fmt='o', color='red', capsize=5,
             capthick=1, ecolor='black', elinewidth=1, label="expérience")
plt.plot(t2, x2, label="modèle")

plt.axhline(y=1500, color='blue', linestyle='--', label='x=1500m')
plt.scatter([T], [1500], color='blue', zorder=5)

plt.legend()
plt.title("Position du coureur en fonction du temps")  # Runner's position vs. time
plt.xlabel("Temps (s)")  # Time (s)
plt.ylabel("Position (m)")  # Position (m)
plt.grid(True)
plt.show()

