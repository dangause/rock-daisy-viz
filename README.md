# rock-daisy-viz

Visualization and analysis toolkit for **Rock Daisies (tribe Perityleae)** in North America, supporting Isaac Lichter-Marck's research at the California Academy of Sciences.

This project provides tools for managing botanical nomenclature, querying species occurrence data from GBIF, and producing geographic visualizations — 2D maps, density heatmaps, and interactive 3D relief plots — of rock daisy distributions across the Americas.

## Project Structure

```
rock-daisy-viz/
├── rockdaisy/                  # Python package
│   ├── __init__.py             # Package exports
│   ├── nomenclator.py          # Taxonomic nomenclature parser
│   └── plotting.py             # Geographic visualization functions
├── notebooks/
│   ├── species_groupings_gbif.ipynb   # Genus-level maps from GBIF data
│   └── img/                           # Generated PDF maps
├── scripts/
│   └── gbif_pull.py            # GBIF backbone API query script
├── data/                       # Data files (git/LFS; large rasters are local-only)
│   ├── nomenclator.txt         # Curated taxonomy with synonymy
│   ├── gbif/                   # GBIF occurrence records (raw + cleaned)
│   ├── great_basin/            # Great Basin DEM raster (local-only)
│   ├── guadalupe/              # Guadalupe Mountains DEM raster (LFS)
│   └── HYP_HR_SR_OB_DR/       # Natural Earth raster (local-only)
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
git lfs pull
uv sync
```

This pulls LFS-tracked data files, creates a virtual environment, and installs all dependencies from the lockfile.

## Data Setup

Data files are stored in `data/` and tracked via **Git LFS** (binary files) or regular git (small text files). After cloning, run `git lfs pull` to download the LFS objects. Large rasters are intentionally excluded from version control to keep the repo small.

| Path | Description | Tracked |
|------|-------------|---------|
| `data/nomenclator.txt` | Curated Perityleae taxonomy with synonymy | LFS |
| `data/gbif/gbif_occurrences.csv` | Raw GBIF occurrence records (~8,700 rows) | LFS |
| `data/gbif/gbif_occurrences_cleaned.csv` | Cleaned occurrences (outliers removed, synonyms mapped) | LFS |
| `data/gbif/outliers.png` | Identified coordinate outliers | Git |
| `data/guadalupe/output_SRTMGL3.tif` | Guadalupe Mountains DEM | LFS |

### Large raster prerequisites (not versioned)

These rasters are required for some maps but are **not stored in git/LFS**. Download or generate them and place them at the paths below:

- `data/HYP_HR_SR_OB_DR/HYP_HR_SR_OB_DR.tif` — Natural Earth 1:10m raster **“Cross Blended Hypso with Relief, Water, Drains, and Ocean Bottom”** (`HYP_HR_SR_OB_DR`). Download the large GeoTIFF from Natural Earth and extract it into `data/HYP_HR_SR_OB_DR/`.
- `data/great_basin/output_be.tif` — Great Basin regional DEM used for the Laphamia Great Basin panels. Generate this by clipping a DEM (e.g., NASADEM) to the Great Basin extent, or update the notebook `raster_path` to point to your local DEM.

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
    "data/great_basin/output_be.tif",
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

The main notebook is `species_groupings_gbif.ipynb`, which generates 20+ genus-level PDF maps from GBIF occurrence data. It loads the nomenclator, reads GBIF occurrences, curates outliers, and produces scatter maps, heatmaps, and 3D relief plots for each genus.

### Generated Maps

Running the notebook produces PDF maps in `notebooks/img/`, organized by genus:

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
