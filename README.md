# windCalc

windCalc is a Python toolkit for data processing and analysis in wind engineering, with a focus on wind load calculations for structures. The repository provides tools for handling experimental and simulation data, performing statistical analysis, and generating wind field models.

## Features

- Compute wind velocity statistics and turbulence quantities (mean, standard deviation, turbulence intensity, integral length scales, spectra).
- Fit wind velocity profiles using log-law and power-law models, and ESDU 74/85 standards.
- Calculate wind-induced surface pressures and loads on building faces.
- Analyze, visualize, and export wind data, including support for plotting profiles and spectra.
- Utilities for processing wind tunnel and CFD datasets.
- Extensible class structure for wind field, pressure, and building modeling.

## Directory Structure

- `src/` - Core Python modules for wind field/statistical analysis, wind load computations, and plotting.
- `runScripts/` - Example or batch scripts for end-to-end analysis (customize as needed).
- `ThirdParty/` - External libraries or resources used by the project.
- `notebooks/`, `refData/`, `scripts/` - Jupyter notebooks, reference datasets, and additional scripts.

## Getting Started

Clone the repository and install required dependencies (see `requirements.txt` if available or pip install common packages such as numpy, scipy, pandas, matplotlib, scikit-learn).

```bash
git clone https://github.com/Tsinuel/windCalc.git
cd windCalc
pip install -r requirements.txt  # or install packages manually
```

## Example Usage

```python
from src import wind

# Load your wind data: e.g., time series of wind speed
UofT = ...  # shape [n_points, n_time]
dt = 0.01

# Compute velocity statistics
stats = wind.get_velTH_stats_1pt(UofT=UofT, dt=dt)

# Fit a log wind profile
z = ...
U = ...
z0, uStar, U_fit = wind.fitVelDataToLogProfile(z, U)

# Plot results (see windPlotting.py)
```

## License

Distributed under the MIT License. See `LICENSE` for details.

---

*For more details, see the code in `src/` or contact the repository maintainer.*
