import pandas as pd
import os
import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import DBSCAN

import seaborn as sns


def cut_obs_df(obs_df, from_ts, to_ts):
    if obs_df.index.tz is None:
        obs_df = obs_df.tz_localize("Europe/Berlin")

    from_dt = pd.to_datetime(from_ts, unit="ms", utc=True).tz_convert("Europe/Berlin")
    to_dt = pd.to_datetime(to_ts, unit="ms", utc=True).tz_convert("Europe/Berlin")

    return obs_df.loc[from_dt:to_dt]


def aggr_df_by(obs_df, aggr_by = "month", aggr_method="sum"):
    assert aggr_by == "month" or aggr_by == "hour"
    assert aggr_method in {"sum", "mean"}, "aggr_method must be 'sum' or 'mean'"
    
    obs_df_by_ = obs_df.copy()
    obs_df_by_[aggr_by] = obs_df_by_.index.month if aggr_by == "month" else obs_df_by_.index.hour
    
    if aggr_method == "sum":
        aggr_obs_by_ = obs_df_by_.groupby(aggr_by).sum()
    else:
        aggr_obs_by_ = obs_df_by_.groupby(aggr_by).mean()
    
    return aggr_obs_by_, obs_df_by_
    

def plot_heatmap(obs_df, aggr_by = "month", aggr_method="sum"):
    aggr_obs_by_, _ = aggr_df_by(obs_df, aggr_by, aggr_method=aggr_method)
    
    plt.figure(figsize=(12,6))
    sns.heatmap(aggr_obs_by_.T, cmap='viridis', linewidths=0.3, linecolor='gray')

    aggr_by_str = "month" if aggr_by == "month" else "hour"
    plt.title(f"Number of vehicles per sensor per {aggr_by_str}")
    plt.xlabel(f"{aggr_by_str}")
    plt.ylabel("Sensor")
    plt.xticks()
    plt.tight_layout()
    plt.show()
    
    
    
def plot_hourly_total(obs_df, aggr_method="sum", save_path = None):

    hourly_totals = obs_df.sum(axis=1).resample('h').sum()
    
    if aggr_method == "mean":
        hourly_totals_by_hour = hourly_totals.groupby(hourly_totals.index.hour).mean()
        hourly_std_by_hour = hourly_totals.groupby(hourly_totals.index.hour).std()
    elif aggr_method == "sum":
        hourly_totals_by_hour = hourly_totals.groupby(hourly_totals.index.hour).sum()
        hourly_std_by_hour = hourly_totals.groupby(hourly_totals.index.hour).std()
    else:
        raise ValueError("aggr_method must be 'sum' or 'mean'")

    
    plt.figure(figsize=(14, 6))
    
    line_color = '#0077BE'  # Deep blue
    fill_color = '#4ECDC4'  # Turquoise
    
    plt.bar(hourly_totals_by_hour.index, hourly_totals_by_hour.values,
            yerr=hourly_std_by_hour.values,
            color=fill_color, edgecolor=line_color, linewidth=2, alpha=0.8,
            error_kw=dict(ecolor='#E63946', elinewidth=1.5, capsize=5, capthick=1.5))

    
    plt.title("Average Hourly Traffic Volume Across All Sensors", 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Hour of Day", fontsize=13, fontweight='bold')
    plt.ylabel("Average Traffic Volume", fontsize=13, fontweight='bold')
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, axis='y')
    
    # Set background color
    ax = plt.gca()
    ax.set_facecolor('#FAFAFA')
    plt.gcf().patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        
    plt.show()
    
    
def plot_mean_std_veh_counts(obs_df, aggr_by="hour", aggr_method="sum"):

    _, df_by = aggr_df_by(obs_df, aggr_by, aggr_method)
    df_long = df_by.melt(id_vars=aggr_by, var_name='sensor', value_name='count')
    
    # Calculate mean and standard deviation for each time period
    stats = df_long.groupby(aggr_by)['count'].agg(['mean', 'std']).reset_index()
    
    # Create the plot
    plt.figure(figsize=(14, 6))
    
    # Define fancy colors (using a nice gradient)
    line_color = '#2E86AB'  # Deep blue
    fill_color = '#A23B72'  # Purple-pink
    
    # Plot mean line
    plt.plot(stats[aggr_by], stats['mean'], 
             color=line_color, linewidth=2.5, label='Mean', marker='o', 
             markersize=6, markerfacecolor='white', markeredgewidth=2)
    
    # Plot shaded standard deviation area
    plt.fill_between(stats[aggr_by], 
                     stats['mean'] - stats['std'], 
                     stats['mean'] + stats['std'],
                     alpha=0.3, color=fill_color, label='±1 Std Dev')
    
    # Styling
    plt.title(f"Mean Vehicle Counts every 10 minutes per {aggr_by.capitalize()} with Standard Deviation", 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f"{aggr_by.capitalize()}", fontsize=13, fontweight='bold')
    plt.ylabel("Vehicle Count", fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Set background color
    ax = plt.gca()
    ax.set_facecolor('#F8F9FA')
    plt.gcf().patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.show()
    

def get_zero_val_mask(df, zero_run_threshold):
    mask = pd.DataFrame(1, index=df.index, columns=df.columns)

    for col in df.columns:
        z = (df[col] == 0)
        group_id = (z != z.shift()).cumsum() # assigns a unique integer ID to each contiguous "run" of equal boolean values
        run_lengths = group_id.map(group_id.value_counts()) # replace each entry of group_id with the size of its run
        
        long_zero_runs = z & (run_lengths >= zero_run_threshold) # If zero and it is part of a long run, mask it
        mask.loc[long_zero_runs, col] = 0
        
    return mask.values


def from_df_to_geopandas(
    data,
    lon_key="longitude",
    lat_key="latitude",
    crs="EPSG:4326"
):
    df = pd.DataFrame(data)

    # Ensure coordinates are numeric
    df[lon_key] = pd.to_numeric(df[lon_key])
    df[lat_key] = pd.to_numeric(df[lat_key])

    # Create geometry column
    geometry = gpd.points_from_xy(df[lon_key], df[lat_key])

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

    return gdf


def merge_data_by_junc(data, metadata, use_direction = False, offset = 0, dir_column = "reg_dir", eps = 50):
    
    def normalize_direction(dir_str):
        if pd.isna(dir_str) or dir_str.strip() == "":
            return ""
        parts = [d.strip() for d in dir_str.split(",") if d.strip()]
        return ",".join(sorted(parts))
    
    meta_gdf = metadata.to_crs(epsg=3003)
    
    coords = np.vstack(meta_gdf.geometry.apply(lambda p: (p.x, p.y)))
    db = DBSCAN(eps=eps, min_samples=1, metric='euclidean') # Cluster closer sensors
    meta_gdf['junction_id'] = db.fit_predict(coords)
    
    if use_direction:
        meta_gdf['direction_norm'] = meta_gdf[dir_column].apply(normalize_direction)
        meta_gdf['group_id'] = meta_gdf['direction_norm'].astype(str) + "_" + meta_gdf['junction_id'].astype(str)
    else:
        meta_gdf['group_id'] = meta_gdf['junction_id'].astype(str)
    
    unique_groups = meta_gdf['group_id'].unique()
    group_to_fancy = {gid: f"agg_{i + offset}" for i, gid in enumerate(unique_groups)}
    
    agg_list = []
    for gid, group_df in meta_gdf.groupby('group_id'):
        fancy_id = group_to_fancy[gid]
        mean_point = group_df.to_crs(epsg = 4326).geometry.union_all().centroid  # back to lat/lon
        
        agg_info = {
            "id": fancy_id,
            "reg_dir": group_df["direction_norm"].iloc[0] if use_direction else "NA",
            "latitude": mean_point.y,
            "longitude": mean_point.x,
        }
        agg_list.append(agg_info)
    
    sensor_to_group = meta_gdf.set_index('id')['group_id'].to_dict()
    mapping = {s: group_to_fancy[sensor_to_group[s]] for s in data.columns}
    grouped_df = data.T.groupby(mapping).sum()
    grouped_df = grouped_df.T
    
    return grouped_df, agg_list, mapping