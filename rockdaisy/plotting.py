import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from cartopy.io.shapereader import Reader
import cartopy.io.shapereader as shpreader
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

def plot_time_histogram(df, datetime_col='datetime', bins='auto'):
    """
    Plots a histogram of records over time.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a datetime column.
    - datetime_col (str): Name of the datetime column.
    - bins (str or int): Number of bins ('auto' for automatic binning or int for manual).
    """
    # Ensure the datetime column is in datetime format
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot histogram
    counts, bin_edges, _ = ax.hist(
        df[datetime_col], bins=bins, edgecolor='black', alpha=0.7
    )

    # Format x-axis for date readability
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Records')
    ax.set_title('Records Over Time')

    # Rotate x-ticks for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio.plot import show as rio_show
from cartopy.io.shapereader import Reader
import cartopy.io.shapereader as shpreader
from matplotlib import colormaps

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio.plot import show as rio_show
from cartopy.io.shapereader import Reader
import cartopy.io.shapereader as shpreader
from matplotlib import colormaps

def plot_geographical_positions(
    df, 
    lat_col='decimallatitude_wgs', 
    lon_col='decimallongitude_wgs',
    group_col='speciesCurated',
    group_label='Species',  # NEW: custom name for legend and title
    zoom='auto',
    cluster_line=False,
    plot_roads=False,
    plot_rivers=False,
    raster_path=None,
    bbox=None
):
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': projection})

    # Filter data to only what's inside the bbox
    if bbox:
        min_lon, max_lon, min_lat, max_lat = bbox
        df = df[
            (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon) &
            (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat)
        ]

    if df.empty:
        print("No data inside bounding box. Nothing to plot.")
        return

    # Raster background
    if raster_path:
        with rasterio.open(raster_path) as src:
            rio_show(src, ax=ax, transform=src.transform)

    # Base map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)

    if plot_rivers:
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.7, linewidth=0.5)
    if plot_roads:
        roads_shp = shpreader.natural_earth(category='cultural', name='roads', resolution='10m')
        roads = Reader(roads_shp).geometries()
        ax.add_geometries(roads, crs=projection, edgecolor='brown', facecolor='none', linewidth=0.5, alpha=0.7)

    # Color by group
    group_list = df[group_col].dropna().unique()
    cmap = colormaps.get_cmap('tab20')
    color_map = {group: cmap(i / len(group_list)) for i, group in enumerate(group_list)}

    for group in group_list:
        subset = df[df[group_col] == group]
        ax.scatter(
            subset[lon_col], subset[lat_col],
            color=color_map[group],
            s=50, alpha=0.8, edgecolor='k',
            label=group,
            transform=ccrs.PlateCarree()
        )

        if cluster_line and len(subset) > 1:
            subset = subset.sort_values(by=[lat_col, lon_col])
            for i in range(len(subset) - 1):
                ax.plot(
                    [subset.iloc[i][lon_col], subset.iloc[i + 1][lon_col]],
                    [subset.iloc[i][lat_col], subset.iloc[i + 1][lat_col]],
                    color=color_map[group],
                    linewidth=1.5, alpha=0.6,
                    transform=ccrs.PlateCarree()
                )

    # Set extent
    if bbox:
        ax.set_extent(bbox, crs=projection)
    else:
        min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
        min_lon, max_lon = df[lon_col].min(), df[lon_col].max()
        lat_buffer = (max_lat - min_lat) * 0.1 if max_lat > min_lat else 1
        lon_buffer = (max_lon - min_lon) * 0.1 if max_lon > min_lon else 1
        extent = [min_lon - lon_buffer, max_lon + lon_buffer, min_lat - lat_buffer, max_lat + lat_buffer]
        ax.set_extent(extent, crs=projection)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    ax.set_title(f'Geographical Positions by {group_label}')
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title=group_label,
        frameon=True
    )

    plt.tight_layout()
    plt.show()


def plot_geographical_heatmap_overlay(
    df,
    lat_col='decimallatitude_wgs',
    lon_col='decimallongitude_wgs',
    group_col='speciesCurated',
    group_label='Species',
    raster_path=None,
    grid_size=100,
    bbox=None,
    plot_rivers=False,
    plot_roads=False,
    max_groups=10,
    show_density_labels=False
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import rasterio
    from rasterio.plot import show as rio_show
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgb
    from cartopy.io.shapereader import Reader
    import cartopy.io.shapereader as shpreader

    base_colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#a65628', '#f781bf', '#999999', '#66c2a5', '#d95f02'
    ]

    # Filter to bbox
    if bbox:
        min_lon, max_lon, min_lat, max_lat = bbox
        df = df[
            (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon) &
            (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat)
        ]

    if df.empty:
        print("No data in bounding box. Nothing to plot.")
        return

    # Get unique groups
    group_list = df[group_col].dropna().unique()
    if len(group_list) > max_groups:
        print(f"[info] Limiting to first {max_groups} {group_label.lower()}s for clarity.")
        group_list = group_list[:max_groups]

    # Grid setup
    lon_all = df[lon_col].values
    lat_all = df[lat_col].values
    lon_bins = np.linspace(lon_all.min(), lon_all.max(), grid_size)
    lat_bins = np.linspace(lat_all.min(), lat_all.max(), grid_size)
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'projection': projection})

    # Raster background
    if raster_path:
        with rasterio.open(raster_path) as src:
            rio_show(src, ax=ax, transform=src.transform)

    # Basemap features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    if plot_rivers:
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.6, linewidth=0.4)
    if plot_roads:
        roads_shp = shpreader.natural_earth(category='cultural', name='roads', resolution='10m')
        roads = Reader(roads_shp).geometries()
        ax.add_geometries(roads, crs=projection, edgecolor='brown', facecolor='none', linewidth=0.5, alpha=0.6)

    # Create color maps
    color_map = {}
    legend_patches = []
    for idx, group in enumerate(group_list):
        base_rgb = np.array(to_rgb(base_colors[idx % len(base_colors)]))
        light_rgb = base_rgb + (1 - base_rgb) * 0.4
        cmap = LinearSegmentedColormap.from_list(
            f"{group}_fade", [light_rgb, base_rgb], N=256
        )
        color_map[group] = cmap
        legend_patches.append(mpatches.Patch(color=base_rgb, label=group))

    # Plot heatmaps
    for group in group_list:
        cmap = color_map[group]
        subset = df[df[group_col] == group]
        if subset.empty:
            continue

        heatmap, _, _ = np.histogram2d(
            subset[lon_col], subset[lat_col],
            bins=[lon_bins, lat_bins]
        )

        if np.sum(heatmap) == 0:
            continue

        ax.pcolormesh(
            lon_grid, lat_grid, heatmap.T,
            cmap=cmap,
            norm=LogNorm(vmin=1, vmax=np.max(heatmap) + 1),
            alpha=0.85,
            transform=projection
        )

        # Density labels
        if show_density_labels:
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    val = int(heatmap[i, j])
                    if val > 0:
                        ax.text(
                            lon_centers[i], lat_centers[j], str(val),
                            fontsize=6, color='black', weight='bold',
                            ha='center', va='center',
                            transform=projection,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=0.5)
                        )

        # Max-density label
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        label_lon = lon_centers[max_idx[0]]
        label_lat = lat_centers[max_idx[1]]
        ax.text(
            label_lon, label_lat, group,
            fontsize=9, weight='bold', ha='center', va='center',
            transform=projection,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.25, boxstyle='round,pad=0.3')
        )

    # Set extent
    if bbox:
        ax.set_extent(bbox, crs=projection)
    else:
        lat_buffer = (lat_all.max() - lat_all.min()) * 0.1
        lon_buffer = (lon_all.max() - lon_all.min()) * 0.1
        ax.set_extent([
            lon_all.min() - lon_buffer, lon_all.max() + lon_buffer,
            lat_all.min() - lat_buffer, lat_all.max() + lat_buffer
        ], crs=projection)

    # Gridlines and legend
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    ax.set_title(f"Overlaid {group_label} Heatmaps with Density Labels", fontsize=14)
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.01, 0.5), title=group_label)
    plt.tight_layout()
    plt.show()
