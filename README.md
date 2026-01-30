# rock-daisy-viz

Visualization and analysis toolkit for **Rock Daisies (tribe Perityleae)** in North America, supporting Isaac Lichter-Marck's research at the California Academy of Sciences.

This project provides tools for managing botanical nomenclature, querying species occurrence data from GBIF, and producing geographic visualizations — 2D maps, density heatmaps, and interactive 3D relief plots — of rock daisy distributions across the Americas.

## Project Structure

```
rock-daisy-viz/
├── rockdaisy/                  # Python package
│   ├── nomenclator.py          # Taxonomic nomenclature parser
│   ├── plotting.py             # Geographic visualization functions
│   └── main.py                 # Entry point (stub)
├── notebooks/                  # Jupyter analysis notebooks
│   ├── species_groupings_gbif.ipynb   # Primary: genus-level maps from GBIF data
│   ├── species_groupings.ipynb        # Geographic maps by genus
│   ├── rock_daisy_eda.ipynb           # Exploratory data analysis
│   ├── sky_island_eda.ipynb           # Sky island geographic analysis
│   ├── nomenclator.ipynb             # Nomenclator class demo/testing
│   ├── gbif.ipynb                     # GBIF API demo
│   └── img/                           # Generated PDF maps
├── scripts/
│   └── gbif_pull.py            # GBIF backbone API query script
├── data/                       # Data files (git-ignored)
│   ├── nomenclator.txt         # Curated taxonomy with synonymy
│   ├── nomenclator.csv         # CSV export of nomenclature
│   ├── gbif/                   # GBIF occurrence records
│   ├── great_basin/            # Great Basin DEM rasters
│   ├── guadalupe/              # Guadalupe Mountains rasters
│   ├── sky_island/             # Sky island boundaries (KMZ)
│   ├── HYP_HR_SR_OB_DR/       # NASA high-resolution DEM raster
│   └── perityleae_distribution_data.xlsx
├── pyproject.toml
├── uv.lock
└── .python-version
```

## Prerequisites

- **Python 3.13+**
- [**uv**](https://docs.astral.sh/uv/) package manager
- GDAL/PROJ system libraries (required by `rasterio` and `cartopy`)

On macOS with Homebrew:

```bash
brew install gdal proj
```

## Installation

```bash
git clone https://github.com/dangause/rock-daisy-viz.git
cd rock-daisy-viz
uv sync
```

This creates a virtual environment and installs all dependencies from the lockfile.

## Data Setup

The `data/` directory is git-ignored due to file sizes. You will need to obtain the following:

| Path | Description | Source |
|------|-------------|--------|
| `data/nomenclator.txt` | Curated Perityleae taxonomy with synonymy | Included in repo |
| `data/gbif/gbif_occurrences.csv` | Species occurrence records (~8,700 rows) | [GBIF](https://www.gbif.org/) export or `scripts/gbif_pull.py` |
| `data/HYP_HR_SR_OB_DR/` | NASA high-resolution elevation raster | [Natural Earth](https://www.naturalearthdata.com/downloads/10m-raster-data/) |
| `data/great_basin/` | Great Basin regional DEM | USGS or similar |
| `data/perityleae_distribution_data.xlsx` | Curated distribution spreadsheet | Project-internal |

## Usage

### Nomenclator

The `Nomenclator` class parses a curated taxonomy file that maps species names (including synonyms and varieties) to their accepted names.

```python
from rockdaisy.nomenclator import Nomenclator

nomen = Nomenclator("data/nomenclator.txt")

# Look up a species
nomen.lookup("Perityle emoryi")

# Get all accepted names grouped with their synonyms
nomen.accepted_with_synonyms()

# Filter by genus
nomen.filter_by(genus="Laphamia")

# Export to DataFrame
df = nomen.to_dataframe()
```

### GBIF Querying

Match nomenclator names against the GBIF backbone taxonomy:

```bash
uv run python scripts/gbif_pull.py
```

This queries the GBIF API for all names in the nomenclator and saves matches to `gbif_matches.json`.

### Plotting

The `rockdaisy.plotting` module provides three visualization functions:

**Geographic scatter maps** — plot species occurrences on a cartopy basemap with optional DEM raster backgrounds, hillshade, roads, and rivers:

```python
from rockdaisy.plotting import plot_geographical_positions

plot_geographical_positions(
    df,
    group_col="species",
    raster_path="data/HYP_HR_SR_OB_DR/HYP_HR_SR_OB_DR.tif",
    bbox=[-120, -95, 25, 45],
    hillshade_overlay=True,
    save_path="notebooks/img/my_map.pdf"
)
```

**Density heatmaps** — overlaid per-species heatmaps with logarithmic normalization:

```python
from rockdaisy.plotting import plot_geographical_heatmap_overlay

plot_geographical_heatmap_overlay(
    df,
    group_col="species",
    grid_size=100,
    save_path="notebooks/img/heatmap.pdf"
)
```

**3D relief plots** — interactive PyVista plots of species on DEM surfaces:

```python
from rockdaisy.plotting import plot_3d_relief_with_species

plot_3d_relief_with_species(
    "data/great_basin/dem.tif",
    df,
    group_col="species",
    bbox=[-120, -110, 35, 42],
    elev_exaggeration=0.001
)
```

### Notebooks

The primary analysis workflow lives in the Jupyter notebooks. To launch:

```bash
uv run jupyter lab notebooks/
```

| Notebook | Description |
|----------|-------------|
| `species_groupings_gbif.ipynb` | Main notebook — generates 20+ genus-level PDF maps from GBIF occurrence data |
| `species_groupings.ipynb` | Earlier version using curated distribution spreadsheet |
| `rock_daisy_eda.ipynb` | Data quality exploration: missing values, temporal patterns, field coverage |
| `sky_island_eda.ipynb` | Analysis of sky island complex boundaries and species overlap |
| `nomenclator.ipynb` | Interactive walkthrough of the `Nomenclator` class |
| `gbif.ipynb` | GBIF backbone API query demo |

### Generated Maps

Running the species groupings notebooks produces PDF maps in `notebooks/img/`, organized by genus:

- **Laphamia** — Arizona/New Mexico sky islands, Big Bend, Great Basin, Sonora/Sinaloa/Baja
- **Perityle** — Baja/SW USA, mainland Mexico
- **Galinsogeopsis**, **Eutetras**, **Nesothamnus**, **Pericome** — full range and regional views
- **All genera** — combined scatter and heatmap views

## Dependencies

Core libraries:

| Library | Purpose |
|---------|---------|
| `cartopy` | Geographic map projections and basemaps |
| `rasterio` | DEM/raster file I/O |
| `matplotlib` | 2D plotting |
| `pyvista` | Interactive 3D visualization |
| `pandas` | Tabular data manipulation |
| `pygbif` | GBIF API client |
| `cmocean` | Scientific colormaps |
| `trame` | VTK-based web rendering for Jupyter 3D |
| `openpyxl` | Excel file reading |

## License

This project is part of ongoing research at the California Academy of Sciences.
