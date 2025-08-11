# windCalc

**Version:** 1.0  
**Author:** Tsinuel Geleta

---

## Overview

`windCalc` is a comprehensive Python package designed for advanced wind engineering analysis. It supports the calculation, analysis, and visualization of wind pressure coefficients, wind profiles, structural response, and pressure averaging on building surfaces. The tool is ideal for researchers and engineers working in wind tunnel testing, building aerodynamics, structural wind loading, and related disciplines.

The package includes modules for statistical analysis, structural modeling, data processing, and CAD-based panel/tap management, following standards like ESDU, ASCE, NBCC etc.

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
This core module handles wind engineering computations, including pressure coefficient (Cp) analysis, turbulence modeling, coherence, spectral analysis, and wind profile fitting. Major components include:

- **Cp Analysis**:
  - `bldgCp`, `BldgCps`, `BldgCp_cummulative`: Handle individual and ensemble building Cp statistics. Functions include envelope extraction, peak and average Cp computations, and Cp data export to Excel.
  - `SampleLinesCp`, `faceCp`, `line`: Manage Cp statistics along sample lines, building faces, or specific coordinates.

- **Wind Profile Modeling**:
  - `profile`, `Profiles`: Compute and visualize velocity profiles. Fit experimental or CFD velocity data to log-law or power-law profiles.
  - `logProfile()`, `fitVelToPowerLawProfile()`, `fitVelDataToLogProfile()`: Fit velocity data and extract roughness parameters (`z0`, `u*`).

- **Spectral Analysis**:
  - `spectra`, `vonKarmanSpectra`, `vonKarmanSuu/Svv/Sww`: Compute wind spectra using von Kármán or ESDU models.
  - `ESDU74`, `ESDU85`: Provide all relevant functions and parameterizations for turbulence intensities (`Iu`, `Iv`, `Iw`), power spectral densities, and integral scales.

- **Coherence and Time Scales**:
  - `coherence()`, `Coh_Davenport()`: Calculate coherence between points based on distance and frequency.
  - `integLengthScale()`, `integTimeScale()`: Estimate turbulence length and time scales.

- **Validation Tools**:
  - `validator`: Provides error metrics and plotting tools for comparing modeled vs. experimental Cp or velocity statistics (e.g., bar charts, contours, line plots).
  - `measureError()`, `plotError_CpStats()`, `plotError_velStats()`: Visual diagnostics for assessing model accuracy.

- **Utility Functions**:
  - `LaTeXise()`, `mathName()`, `fullName()`: Formatting functions for labeling.
  - `lowpass()`, `smooth()`, `psd()`: Signal filtering and spectral density estimations.

---

### `src.windCAD`
This module interfaces wind analysis with CAD-like representations of surfaces and pressure tap distributions. It enables detailed paneling, geometric zone definition, and visualization:

- **Panel and Tap Management**:
  - `Faces`, `Taps`, `Zones`: Classes for defining the geometry and attributes of building faces, pressure taps, and aerodynamic zones.
  - Functions include `plotPanels()`, `plotTaps()`, `plotZones()`, `boundingBoxPlt()`, and tap/panel field plotting.

- **Geometry and Quality Checks**:
  - `panelingErrors`, `error_in_panels`, `error_in_zones`: Tools for verifying spatial consistency and tap assignments.
  - `Refresh()`, `RemovedBadTaps`: Update and clean geometry definitions.

- **Panel Attributes**:
  - `panelAreas_groupAvg`, `panelIdxRanges`, `panelArea_nominal`: Compute and organize panel properties by face and zone.

---

### `src.windCodes`
A standards-compliant module for implementing wind loading rules from international codes and guidelines:

- **Code-Specific Implementations** (depending on availability):
  - Compute reference wind pressures, gust factors, and terrain multipliers.
  - Evaluate code-specific Cp or Cf distributions for simplified design.

- **Potential Support**:
  - Standards such as ASCE 7, NBCC, or EN 1991 may be implemented here or extended.

---

### `src.windIO`
Handles data input/output for wind analysis tasks. Facilitates efficient loading, parsing, and exporting of wind data in structured formats:

- **Supported Formats**:
  - Time history data, Cp tables, processed stats, and structured JSON.

- **Export Functions**:
  - `writeCpStatsToXLSX()`, `writeCandCloadToXLSX()`: Export Cp statistics and derived load values to Excel.
  - `writeDataToFile()`, `writeToJSON()`: Save computed or validated profiles and spectra to files for later use.

- **Reading Utilities**:
  - `readFromFile()`, `readFromJSON()`: Ingest saved datasets into the analysis pipeline.

---

### `src.windOF`
This module acts as a bridge between windCalc and **OpenFOAM** (OF) outputs. It parses CFD results and enables streamlined post-processing:

- **Typical Use Cases**:
  - Extract Cp or velocity fields from OF output files.
  - Convert sampled data into `profile`, `spectra`, or `bldgCp` objects for direct analysis.

- **Functions**:
  - Parsing `sampleDict`, `probeData`, or other post-processing outputs.
  - Matching geometry or time-series data with CAD panels/taps.

---

### `src.windWT`
Provides interfaces and utilities tailored for **Wind Tunnel (WT)** data processing:

- **WT-Specific Workflows**:
  - Reads Cp time histories and metadata from pressure tap files.
  - Matches tap locations with CAD models (`windCAD`) and formats time series for analysis.

- **Data Conditioning**:
  - Performs cleaning, interpolation, and filtering on WT time history data.
  - Converts raw tunnel data into `bldgCp`, `profile`, or `spectra` formats.

- **Calibration and Setup**:
  - Tools for handling Reynolds number matching, AoA variation, and reference pressure conversions.

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

Full documentation is available in HTML format. Open docs/index.html in your browser or rebuild with:
```bash
make html
```
