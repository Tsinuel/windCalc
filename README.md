# windTools

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


## Usage
