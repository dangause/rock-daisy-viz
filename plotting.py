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


def plot_geographical_positions(
    df, 
    lat_col='decimallatitude_wgs', 
    lon_col='decimallongitude_wgs',
    species_col='speciesCurated',
    zoom='auto',
    cluster_line=False,
    plot_towns=False,
    plot_roads=False,
    plot_rivers=False
):
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.5)

    if plot_rivers:
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.7, linewidth=0.5)
    if plot_roads:
        roads_shp = shpreader.natural_earth(category='cultural', name='roads', resolution='10m')
        roads = Reader(roads_shp).geometries()
        ax.add_geometries(roads, crs=ccrs.PlateCarree(), edgecolor='brown', facecolor='none', linewidth=0.5, alpha=0.7)

    # Assign unique colors to species
    species_list = df[species_col].unique()
    cmap = cm.get_cmap('tab20', len(species_list))
    color_map = {species: cmap(i) for i, species in enumerate(species_list)}

    # Plot points, grouped by species
    for species in species_list:
        subset = df[df[species_col] == species]
        ax.scatter(
            subset[lon_col],
            subset[lat_col],
            color=color_map[species],
            s=50,
            alpha=0.7,
            edgecolor='k',
            label=species,
            transform=ccrs.PlateCarree()
        )

        if cluster_line and len(subset) > 1:
            subset = subset.sort_values(by=[lat_col, lon_col])
            for i in range(len(subset) - 1):
                ax.plot(
                    [subset.iloc[i][lon_col], subset.iloc[i + 1][lon_col]],
                    [subset.iloc[i][lat_col], subset.iloc[i + 1][lat_col]],
                    color=color_map[species],
                    linewidth=1.5,
                    alpha=0.6,
                    transform=ccrs.PlateCarree()
                )

    # Determine map extent
    min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
    min_lon, max_lon = df[lon_col].min(), df[lon_col].max()
    lat_buffer = (max_lat - min_lat) * 0.1 if max_lat > min_lat else 0.5
    lon_buffer = (max_lon - min_lon) * 0.1 if max_lon > min_lon else 0.5

    if zoom == 'auto' or isinstance(zoom, (int, float)):
        extent = [
            min_lon - lon_buffer, max_lon + lon_buffer,
            min_lat - lat_buffer, max_lat + lat_buffer
        ]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    elif zoom == 'california':
        ax.set_extent([-125, -113, 32, 42], crs=ccrs.PlateCarree())
    elif zoom == 'us':
        ax.set_extent([-130, -60, 24, 50], crs=ccrs.PlateCarree())
    elif zoom == 'world':
        ax.set_global()
    else:
        raise ValueError("Invalid zoom option.")

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    ax.set_title('Geographical Positions by Species')

    # Legend on the right
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
        title='Species'
    )

    plt.tight_layout()
    plt.show()


def plot_geographical_heatmap(df, lat_col='lat', lon_col='lon', zoom='auto', grid_size=100, plot_rivers=False, plot_roads=False):
    """
    Plots a heatmap of geographical positions from a DataFrame based on record density overlaid on a detailed world map.

    Parameters:
    - df (pd.DataFrame): DataFrame containing latitude and longitude columns.
    - lat_col (str): Name of the latitude column.
    - lon_col (str): Name of the longitude column.
    - zoom (str or float): Zoom level ('auto', 'california', 'us', 'world') or a numeric value to control lat/lon buffers inversely.
    - grid_size (int): Number of bins for the heatmap in each dimension (higher values result in finer grids).
    - plot_rivers (bool): Whether to plot rivers on the map.
    - plot_roads (bool): Whether to plot roads on the map.
    """
    # Create the figure and axis using PlateCarree projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add map features (coastlines, countries, etc.)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.5)

    # Add rivers and roads
    if plot_rivers:
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.7, linewidth=0.5, label='Rivers')
    if plot_roads:
        pass

    # Determine the map extent based on zoom level
    if isinstance(zoom, (int, float)):
        zoom_factor = 1 / zoom  # Inverse relationship
        # Calculate bounds for zoom
        min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
        min_lon, max_lon = df[lon_col].min(), df[lon_col].max()

        # Buffer for visualization
        lat_buffer = (max_lat - min_lat) * zoom_factor
        lon_buffer = (max_lon - min_lon) * zoom_factor

        extent = [min_lon - lon_buffer, max_lon + lon_buffer,
                  min_lat - lat_buffer, max_lat + lat_buffer]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    elif zoom == 'auto':
        min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
        min_lon, max_lon = df[lon_col].min(), df[lon_col].max()
        lat_buffer = (max_lat - min_lat) * 0.1
        lon_buffer = (max_lon - min_lon) * 0.1
        extent = [min_lon - lon_buffer, max_lon + lon_buffer,
                  min_lat - lat_buffer, max_lat + lat_buffer]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    elif zoom == 'california':
        extent = [-125, -113, 32, 42]  # California region
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    elif zoom == 'us':
        extent = [-130, -60, 24, 50]  # Continental US
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    elif zoom == 'world':
        ax.set_global()
    else:
        raise ValueError("Invalid zoom option. Choose from 'auto', 'california', 'us', 'world', or a numeric value.")

    # Create a 2D histogram for the heatmap
    lon_bins = np.linspace(df[lon_col].min(), df[lon_col].max(), grid_size)
    lat_bins = np.linspace(df[lat_col].min(), df[lat_col].max(), grid_size)
    heatmap, lon_edges, lat_edges = np.histogram2d(df[lon_col], df[lat_col], bins=[lon_bins, lat_bins])

    # Plot the heatmap
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    pcm = ax.pcolormesh(lon_grid, lat_grid, heatmap.T, cmap='winter', norm=LogNorm(), transform=ccrs.PlateCarree(), alpha=0.8)

    # Add a color bar
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label('Number of Records')

    # Add lat/lon gridlines and labels
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add a distance scale in the bottom left corner
    if isinstance(zoom, (int, float)) or zoom == 'auto':
        # Calculate the approximate scale for the distance
        scale_km = (extent[1] - extent[0]) * 111 / 10  # Approximate 1 degree as 111 km
        scale_label = f'{int(scale_km)} km'
        ax.plot([extent[0] + 0.05 * (extent[1] - extent[0]), extent[0] + 0.15 * (extent[1] - extent[0])],
                [extent[2] + 0.05 * (extent[3] - extent[2]), extent[2] + 0.05 * (extent[3] - extent[2])],
                transform=ccrs.PlateCarree(), color='black', linewidth=2)
        ax.text(extent[0] + 0.16 * (extent[1] - extent[0]),
                extent[2] + 0.05 * (extent[3] - extent[2]),
                scale_label,
                transform=ccrs.PlateCarree(), fontsize=10, color='black')

    ax.set_title('Geographical Density Heatmap')
    plt.show()

def plot_geographical_heatmap_by_day(df, lat_col='lat', lon_col='lon', datetime_col='datetime', zoom='auto', grid_size=100, plot_rivers=False, plot_roads=False):
    """
    Wrapper function to plot a heatmap for each day in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing latitude, longitude, and datetime columns.
    - lat_col (str): Name of the latitude column.
    - lon_col (str): Name of the longitude column.
    - datetime_col (str): Name of the datetime column.
    - zoom (str or float): Zoom level ('auto', 'california', 'us', 'world') or a numeric value to control lat/lon buffers inversely.
    - grid_size (int): Number of bins for the heatmap in each dimension (higher values result in finer grids).
    - plot_rivers (bool): Whether to plot rivers on the map.
    - plot_roads (bool): Whether to plot roads on the map.
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['date'] = df[datetime_col].dt.date

    unique_dates = sorted(df['date'].unique())
    num_days = len(unique_dates)

    # Determine the overall extent to ensure consistent zoom across subplots
    min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
    min_lon, max_lon = df[lon_col].min(), df[lon_col].max()
    lat_buffer = (max_lat - min_lat) * 0.1
    lon_buffer = (max_lon - min_lon) * 0.1
    extent = [min_lon - lon_buffer, max_lon + lon_buffer, min_lat - lat_buffer, max_lat + lat_buffer]

    # Calculate global min and max for heatmap values
    lon_bins = np.linspace(min_lon, max_lon, grid_size)
    lat_bins = np.linspace(min_lat, max_lat, grid_size)
    global_heatmap = np.histogram2d(df[lon_col], df[lat_col], bins=[lon_bins, lat_bins])[0]
    vmin = global_heatmap[global_heatmap > 0].min() if np.any(global_heatmap > 0) else 1
    vmax = global_heatmap.max() if np.isfinite(global_heatmap.max()) else 10

    # Create subplots with 3 per row
    ncols = 3
    nrows = (num_days + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    for ax, current_date in zip(axes, unique_dates):
        daily_df = df[df['date'] == current_date]

        # Add map features (coastlines, countries, etc.)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, alpha=0.5)

        # Add rivers and roads
        if plot_rivers:
            ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.7, linewidth=0.5, label='Rivers')
        if plot_roads:
            pass

        # Set the consistent extent
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Create a 2D histogram for the heatmap
        heatmap, lon_edges, lat_edges = np.histogram2d(daily_df[lon_col], daily_df[lat_col], bins=[lon_bins, lat_bins])

        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
        lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

        pcm = ax.pcolormesh(lon_grid, lat_grid, heatmap.T, cmap='winter', norm=LogNorm(vmin=vmin, vmax=vmax), transform=ccrs.PlateCarree(), alpha=0.8)

        # Add a color bar
        cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.7)
        cbar.set_label('Number of Records')

        # Add lat/lon gridlines and labels
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.7)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        ax.set_title(f'Heatmap for {current_date}')

    # Turn off unused subplots
    for ax in axes[len(unique_dates):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
