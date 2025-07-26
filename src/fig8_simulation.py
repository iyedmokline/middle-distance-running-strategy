import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# FIGURE 8 – COMPARISON: MODEL VS. EXPERIMENTAL POSITION-TIME DATA
# This script compares the runner's position predicted by the sinusoidal model
# to actual experimental data with uncertainty on timing.
# ----------------------------------------------------------------------

# -------------------------------
# PARAMETERS FROM OPTIMIZATION
# -------------------------------
V0 = 6.756005867882389           # Base velocity (m/s)
A = 0.47323504752192025          # Amplitude of speed oscillation
w = 0.5000000000000093           # Angular frequency (rad/s)
T = 221.64651651000005           # Final time to reach 1500m (s)

# -------------------------------
# POSITION MODEL
# -------------------------------

def x(t, V0, A, w):
    """
    Position function x(t) derived from the velocity profile:
    Integrates v(t) = V0 + A*sin(w*t)
    The primitive yields a sinusoidal progression of distance.
    """
    return V0 * t - (A / w) * np.cos(w * t) + (A / w)

# -------------------------------
# EXPERIMENTAL DATA
# -------------------------------
# Measured time (t1) to reach distances (x1), with time uncertainty ±0.0058 s
t1 = [0, 26.23, 56.89, 89.38, 119.65, 151.17, 182.92, 213.42, 232.91]  # seconds
x1 = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1500]                   # meters
time_uncertainty = 0.0058  # Time uncertainty for error bars

# -------------------------------
# SIMULATED DATA FROM MODEL
# -------------------------------
t2 = np.linspace(0, T, 1000)  # Time points for smooth simulation
x2 = [x(t, V0, A, w) for t in t2]  # Position at each time point

# -------------------------------
# PLOTTING
# -------------------------------

# Plot experimental points with horizontal error bars (time uncertainty)
plt.errorbar(
    t1, x1, xerr=time_uncertainty, fmt='o', color='red',
    capsize=5, capthick=1, ecolor='black', elinewidth=1,
    label="expérience"
)

# Plot model curve
plt.plot(t2, x2, label="modèle")

# Highlight finish line at x = 1500m
plt.axhline(y=1500, color='blue', linestyle='--', label='x=1500m')
plt.scatter([T], [1500], color='blue', zorder=5)  # Mark final time

# Axis labels and title (French for report consistency)
plt.xlabel("Temps (s)")         # Time in seconds
plt.ylabel("Position (m)")      # Position in meters
plt.title("Position du coureur en fonction du temps")  # Title in French

# Display legend and grid
plt.legend()
plt.grid(True)

# Show the figure
plt.show()
