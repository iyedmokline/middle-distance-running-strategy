# 🏃 Middle-Distance Running: The Keys to a Winning Strategy

This project explores mathematical modeling and optimization of a 1500-meter running strategy using real physiological data and Python simulation. The work is inspired by models from Keller, Aftalion, and Bonnans, and was developed during my final year in CPGE (MP option SI track, 2023–2024).

## 🎯 Objective

Determine how a runner can structure their effort to minimize race time without exceeding physiological constraints like VO2 max, anaerobic energy, and maximal force.

## 📐 Modeling Approach

- The runner is modeled as a mass subject to propulsion force and air resistance.
- Speed is modeled as a sinusoidal function:
  \[
  V(t) = V_0 + A \sin(\omega t)
  \]
- The model incorporates both aerobic and anaerobic energy dynamics based on real physiology.
- A recovery term is included when speed decreases (following Aftalion–Fiorini's improvement to Keller's model).

## 📊 Physiological Parameters (Measured)

| Parameter                     | Value        |
|------------------------------|--------------|
| Mass (m)                     | 61 kg        |
| Initial Energy (e₀)          | 1337.5 J/kg  |
| Max Aerobic Power (sigma max)| 22.05 W/kg   |
| Final Power (sigma final)    | 19.37 W/kg   |
| Max Force                    | 578.6 N      |
| Max dF/dt                    | 186.3 N/s    |

## ⚙️ Optimization

The optimization uses `scipy.optimize.minimize` with nonlinear constraints:

- Distance covered = 1500m
- Constraints on \( A > 0 \), \( V_0 > A \), \( T > 0 \)
- Anaerobic energy must remain ≥ 0 throughout
- Force and force rate must not exceed physiological limits

🎯 **Goal**: Minimize total race time \( T \)

## 📈 Results

- **Simulated race time**: 3 minutes 41 seconds 65
- **Real athlete time**: 3 minutes 52 seconds 91
- **Improvement**: ~11 seconds using optimized pacing strategy

## 📁 Project Structure

middle-distance-running-strategy/

├── main.py # Python simulation and optimization

├── README.md # This file

├── report/

│ ├── TIPE.pdf # Project presentation slides

│ └── MCOT.pdf # Project summary form

## 🛠️ Tech Stack

- Python 3
- NumPy
- Matplotlib
- SciPy (minimization, constraints)

## 📚 References

- Keller, J.B. (1974). *Optimal velocity in a race* [DOI: 10.2307/2318584]
- Aftalion & Bonnans (2013). *Energy-based optimization of race strategy*
- Aftalion & Fiorini (2015). *Modeling energy recovery during deceleration*

## 📜 License

Distributed under the MIT License. Feel free to reuse or adapt for academic purposes.
