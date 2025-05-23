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

---

## Deep Dive: Class Structure (`src/`)

### Geometry & Data Classes

- **`face`**: Models a 2D building surface (wall, roof), storing geometry, tap locations, zoning and paneling info, and supports Voronoi-based tap tributary and panel mesh generation. Provides geometric queries, error reporting, serialization, and advanced plotting utilities.
- **`Faces`**: A collection of `face` objects, aggregating taps, panels, and zones across faces. Supports collective error reporting, bounding boxes, serialization, and visualization.
- **`samplingLine` / `SamplingLines`**: Represent lines (sampling regions) on a face, used for selecting taps along a path, with fringe zones and plotting support.

### Building & Structural Classes

- **`building`**: Extends `Faces` to include building-level metadata (name, height, width, etc.) and 3D visualization.
- **CAD Classes**:
  - **`node_CAD`**: 1D/2D/3D point with connectivity (elements, panels), DOF info, and plotting.
  - **`panel_CAD`**: Polygonal panel with support nodes, area share calculations, tap overlaps, and plotting.
  - **`element_CAD`**: Structural element (beam, column) connecting nodes with orientation, length, and connected panel queries.
  - **`frame2D_CAD`**: A frame of nodes/elements forming the building skeleton; provides local axes, geometric properties, and panel aggregation.

### Wind Engineering Extensions (`structure.py`)

- **`panel`**: Inherits `panel_CAD`, adds reference to the parent building and area-weighted force coefficient calculations from tap data.
- **`node`**: Inherits `node_CAD`, aggregates force coefficient time histories from connected panels.
- **`element`**: Inherits `element_CAD`, adds material and section properties, and computes local/global stiffness matrices.
- **`frame2D`**: Inherits `frame2D_CAD`, aggregates force coefficients and statistics for the frame.

### Design Patterns

- **Geometry-first, then physics**: Define geometry (faces, panels, nodes) first, then attach wind/structural data using subclassing.
- **Extensive use of properties**: Many computed attributes are exposed as properties for lazy evaluation.
- **Rich plotting support**: Most classes have 2D/3D plotting methods for geometry, data, and results.
- **Serialization**: All major classes can be serialized/deserialized for reproducibility.
- **Aggregation**: Collection classes group geometry and data for building- and frame-level analyses.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---
*For more details, see the code in `src/` or contact the repository maintainer.*
