# 🏃 Middle-Distance Running: The Keys to a Winning Strategy

This project explores mathematical modeling and optimization of a 1500-meter running strategy using real physiological data and Python simulation. The work is inspired by models from Keller, Aftalion, and Bonnans, and was developed during my final year in CPGE (MP option SI track, 2023/2024).

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Objective

Determine the optimal pacing profile to minimize 1500m race time without exceeding physiological constraints such as VO2 max, anaerobic capacity & force generation thresholds.

---

## 🧪 Real Athlete Data

All physiological parameters used in this model are based on data from elite Tunisian middle-distance runner [**Oussama Laajili**](https://worldathletics.org/athletes/tunisia/osama-al-ajili-15036538), collected and validated by the [**Centre National de la Médecine et des Sciences du Sport (CNMSS)**](https://cnmss.tn/), Tunisia’s official institution for elite sport medicine and performance.

> The collaboration ensured high-quality data on VO₂ max, maximum force, and anaerobic capacity under real training conditions.

His coach, [**Sofiane Labidi**](https://fr.wikipedia.org/wiki/Sofiane_Labidi), former elite 400m runner whose national record has stood unbeaten in Tunisia for over 20 years, directly contributed to the validation of the pacing strategy and interpretation of performance metrics.

---

## 🧠 Modeling Principles

- The runner is modeled as a point mass subject to propulsion force and air resistance.
- Speed is modeled using a sinusoidal pacing strategy:  **V(t) = V₀ + A · sin(ω·t)**
- Both aerobic and anaerobic energy systems are modeled with real physiological dynamics.
- Energy recovery during deceleration is included, improving on the classic Keller model (based on Aftalion–Fiorini).

## 📊 Physiological Parameters (From Real Athlete Evaluation)

The following values were measured for a national-level middle-distance runner (400–1500m) during a sports performance evaluation conducted in July 2025 by CNMSS (Centre National de Médecine du Sport et de la Santé):

| Parameter                     | Value        |
|------------------------------|--------------|
| Mass (m)                     | 61 kg        |
| Initial Energy (e₀)          | 1337.5 J/kg  |
| Max Aerobic Power (σₘₐₓ)     | 22.05 W/kg   |
| Final Power (σf)             | 19.37 W/kg   |
| Max Force                    | 578.6 N      |
| Max Rate of Force Dev.       | 186.3 N/s    |

*Athlete data anonymized and used with explicit consent for academic simulation purposes.*

## ⚙️ Optimization

The optimization problem is solved using `scipy.optimize.minimize` with nonlinear constraints:

- Total distance covered = 1500m
- \( A > 0 \), \( V_0 > A \), \( T > 0 \)
- Anaerobic energy must remain ≥ 0 throughout
- Force and force rate must stay below physiological limits

🎯 **Objective**: Minimize total race time \( T \)

## 📈 Results

- **Simulated race time**: 3 minutes 41 seconds 65
- **Real athlete time**: 3 minutes 52 seconds 91
- **Time gain potential**: ~11 seconds using optimized pacing strategy

## 📁 Project Structure

```plaintext
middle-distance-running-strategy/
├── fig3_simulation.py               # Force and energy dynamics simulation (Figure 3)
├── fig4_fig5_fig6_simulation.py     # Strategy variation simulations (Figures 4–6)
├── fig5_simulation.py               # Validation or sensitivity analysis (Figure 5)
├── fig7_simulation.py               # Anaerobic energy profile analysis (Figure 7)
├── fig8_simulation.py               # Strategy robustness under constraints (Figure 8)
├── slide16_illustration.py          # Physiological limit illustration (Slide 16)
├── report/
│   ├── performance_evaluation_ousama_laajili.pdf  # Medical lab results
│   ├── TIPE.pdf                     # Final presentation slides
│   └── MCOT.pdf                     # Project summary form
├── LICENSE
├── README.md
└── requirements.txt
```

## 🔧 Installation

To run the code, install the required Python packages:

```bash
pip install -r requirements.txt
