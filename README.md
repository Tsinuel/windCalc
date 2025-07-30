# windCalc: A Python module for wind load analysis

**windTools** is a Python toolkit for wind engineering and atmospheric boundary layer (ABL) analysis. The `wind.py` module provides functions and classes for statistical and spectral analysis of wind velocity and pressure data, including integral scales, turbulence intensities, spectral densities, and peak pressure estimation using Gumbel distributions.

## Features

- 📊 **Wind velocity and turbulence statistics** (`get_velTH_stats_1pt`)
- 📈 **Spectral analysis** of wind velocity components (`spectra` class)
- 🔁 **Auto-correlation and integral time/length scale estimation**
- 🔬 **Fitting of logarithmic wind profiles** to velocity data
- 📐 **Von Kármán turbulence spectra and Davenport coherence models**
- 🌬️ **Pressure time-history analysis** and Gumbel-based peak estimation
- 🧮 **Error metrics** for CFD and experimental data comparison
- 🎨 **Matplotlib-based plotting utilities**

## Installation

```bash
git clone https://github.com/yourusername/windTools.git
cd windTools
pip install -r requirements.txt
```
Note: This package depends on `numpy`, `scipy`, `pandas`, `matplotlib`, `sklearn`. Make sure these are available or included in your project.

## Usage

```python
import wind

# Example: Compute wind statistics
UofT = ...  # np.ndarray of u-component time histories
stats = wind.get_velTH_stats_1pt(UofT=UofT, dt=0.01)
print(stats)

# Example: Fit a log profile
Z = np.array([1, 2, 5, 10])
U = np.array([2.5, 3.0, 4.0, 5.0])
z0, u_star, U_fit = wind.fitVelDataToLogProfile(Z, U, debugMode=True)
```

## Repository Structure

```bash
windTools/
├── wind.py               # Main module with all core functionality
├── windCAD.py            # Geometry and line sampling tools
├── windCodes.py          # Theoretical spectrum and turbulence models
├── windPlotting.py       # Visualization and plot formatting utilities
├── refData/
│   └── bluecoeff.json    # BLUE coefficients for Gumbel peak estimation
├── tests/                # Test scripts (optional)
└── examples/             # Example notebooks or scripts
```

## Citing
If you use this module for published research, please consider citing the author:
```arduino
T. Geleta, "windCalc: A Python module for wind load analysis", 2022.
```

