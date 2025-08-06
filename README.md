# windCalc

**Version:** 0.1  
**Author:** Tsinuel Geleta

---

## Overview

`windCalc` is a comprehensive Python package designed for advanced wind engineering analysis. It supports the calculation, analysis, and visualization of wind pressure coefficients, wind profiles, structural response, and pressure averaging on building surfaces. The tool is ideal for researchers and engineers working in wind tunnel testing, building aerodynamics, structural wind loading, and related disciplines.

The package includes modules for spectral analysis, coherence functions, structural modeling, data processing, and CAD-based panel/tap management, following standards like ESDU 74030 and ESDU 85020.

---

## Features

- **Wind Load Processing:**
  - Support for multiple pressure tap configurations and averaging strategies.
  - Tools for computing and plotting pressure coefficient statistics (`CpStats`) under various AoA (Angle of Attack) configurations.

- **Turbulence Modeling:**
  - Implementations of spectral models such as Von Kármán, ESDU 74, and ESDU 85.
  - Turbulence intensity (`Iu`, `Iv`, `Iw`), power spectral densities (`Suu`, `Svv`, `Sww`), and integral length/time scales.

- **Structural Module:**
  - 2D frame element formulation with local and global stiffness matrices.
  - Transformation and load applications for structural elements, nodes, and panels.

- **Profile Management:**
  - Definition and visualization of wind velocity profiles.
  - Log-law and power-law fitting, shear velocity computation, and coherence calculations.

- **Validation Tools:**
  - Error quantification between modeled and measured data.
  - Visual comparison tools for velocity and pressure fields.

- **CAD Utilities:**
  - Detailed face, panel, tap, and zone management.
  - Geometry parsing and plotting capabilities.

---

## Modules Summary

### `src.wind`
Handles wind-specific functions and statistical operations:
- `BldgCp_cummulative`, `BldgCps`: Compute cumulative and per-building Cp statistics.
- `profile`, `Profiles`: Wind profile creation, analysis, and plotting.
- `ESDU74`, `ESDU85`: Turbulence modeling functions from ESDU standards.
- `spectra`, `vonKarmanSpectra`: Frequency domain analyses and plotting.
- `validator`: Validation utilities for experimental/model comparison.

### `src.structure`
Provides finite element-like structural representation:
- `element`, `node`, `panel`, `frame2D`: Represent structural members and their mechanical properties.
- Methods to compute stiffness matrices and structural responses under aerodynamic loads.

### `src.spatialAvg_FFS`
- `average_data()`: Computes spatial averages using Full-Field Sampling methods.

### `src.calcSSL`
- `SSL`, `curvedCoordSys`: Utilities for streamlined coordinate transformations and structural line elements.

### `src.windCAD`
- `Faces`, `Taps`, `Zones`: Interface with geometric and panelized building models.
- Visualization and error-checking tools for CAD data.

### `src.pyRunWind`
- `main()`: Execution entry-point for wind analysis automation.

---

## Installation

```bash
pip install windCalc
```

Or clone the repository and install manually:

```bash
git clone https://github.com/your-repo/windCalc.git
cd windCalc
pip install .
```

## Usage Example

```python
from windCalc.wind import BldgCps

# Load and process data
bldgs = BldgCps()
bldgs.CpStats()
bldgs.plotTapCpStatsPerAoA()
```

## Documentation

Full API documentation is available in HTML format. Open docs/index.html in your browser or rebuild with:
```bash
make html
```
