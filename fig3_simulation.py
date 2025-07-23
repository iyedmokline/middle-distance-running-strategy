# fig3_simulation.py
# Author: Mohamed Iyed Mokline
# Description: Generates Figure 3 — plots aerobic power as a function of remaining anaerobic energy.
#              Based on a piecewise physiological model of energy phases used in middle-distance running.
#              Inspired by the framework of Aftalion & Bonnans (2013).

import matplotlib.pyplot as plt
import numpy as np

def fc_affine(x, a, b):
    """
    Computes an affine function f(x) = ax + b.
    Used to model different linear phases of the aerobic power response.
    """
    return a * x + b

# --- Constants (physiological parameters in W/kg or J/kg) ---
sigma_max = 22.05       # Maximum aerobic power at the peak
sigma_final = 19.37     # Aerobic power at the end of phase 1
sigma_rest = 5.68       # Minimal aerobic power at exhaustion
energy_total = 1337.5   # Total anaerobic energy capacity (J/kg)

# Thresholds separating the 3 physiological phases
energy_phi = 0.5 * energy_total         # End of phase 2
energy_critical = 0.3 * energy_total    # End of phase 1

# --- Domains for the 3 linear segments ---
range1 = np.linspace(0, energy_critical, 100)          # Phase 1: linear increase
range2 = np.linspace(energy_critical, energy_phi, 100) # Phase 2: plateau
range3 = np.linspace(energy_phi, energy_total, 100)    # Phase 3: linear decrease

# --- Piecewise definition of aerobic power sigma(e) ---
# Phase 1: increasing from sigma_final to sigma_max
sigma1 = fc_affine(
    range1,
    (sigma_max - sigma_final) / energy_critical,
    sigma_final
)

# Phase 2: constant at sigma_max
sigma2 = fc_affine(range2, 0, sigma_max)

# Phase 3: decreasing from sigma_max to sigma_rest
sigma3 = fc_affine(
    range3,
    -(sigma_max - sigma_rest) / (energy_total - energy_phi),
    sigma_rest + energy_total * (sigma_max - sigma_rest) / (energy_total - energy_phi)
)

# --- Plotting the 3 segments ---
plt.plot(range1, sigma1, color='k')
plt.plot(range2, sigma2, color='k')
plt.plot(range3, sigma3, color='k')

# Axis labels (French kept for consistency with report)
plt.ylabel('Puissance aérobie en W/kg', color='C4')
plt.xlabel("Énergie anaérobie restante en J/kg", color='C4')

# Highlight Phase 1 with a transparent red area
plt.fill_between(
    range1,
    sigma1,
    where=(range1 >= 0) & (range1 <= energy_critical),
    color='red',
    alpha=0.3
)

# Title and visual adjustments
plt.title("Évolution de la puissance aérobie en fonction de l'énergie anaérobie restante", size=10, color='C3')
plt.grid()
plt.tight_layout()
plt.show()
