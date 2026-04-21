import pandas as pd
import os
import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
import osmnx as ox

import seaborn as sns


def filter_outliers(data, m=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower = q1 - m * iqr
    upper = q3 + m * iqr
    return lower, upper


def filter_cameras(traffic_cams, m = 2):
    for cam in traffic_cams:
        cam['latitude'] = float(cam['latitude'])
        cam['longitude'] = float(cam['longitude'])
    
    latitudes = np.array([cam['latitude'] for cam in traffic_cams])
    longitudes = np.array([cam['longitude'] for cam in traffic_cams])
    
    lat_lower, lat_upper = filter_outliers(latitudes, m=m)
    lon_lower, lon_upper = filter_outliers(longitudes, m=m)
    
    # Filter cameras within bounds
    filtered_cams = [
        cam for cam in traffic_cams
        if lat_lower <= cam['latitude'] <= lat_upper and lon_lower <= cam['longitude'] <= lon_upper
    ]
    
    return filtered_cams

    
def from_json_to_geojson(traffic_json_data):
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    for cam in traffic_json_data:
        feature = {
            "type": "Feature",
            "properties": {
                "name": cam.get("name", "-"),
                "id": cam.get("id", "-"),
                "veh_count": cam.get("veh_count", 0)
            },
            "geometry": {
                "type": "Point",
                "coordinates": [cam["longitude"], cam["latitude"]]  # GeoJSON uses [lon, lat] order
            }
        }
        geojson["features"].append(feature)
    
    return geojson


def from_csv_to_geojson(df):
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "properties": {
                "name": row.get("name", "-"),
                "id": row.get("id", "-")
            },
            "geometry": {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]]  # GeoJSON uses [lon, lat] order
            }
        }
        geojson["features"].append(feature)
    
    return geojson


def load_time_series(files, name_to_idx_mapping, valid_ids = None):
    # Define percentage bins
    bins = [(0, 10), (10, 30), (30, 70), (70, 100)]

    # Store per-month results
    monthly_distribution = {}
    sensors_time_series = defaultdict(dict)
    sensors_month_occurrences = {}
    sensors_veh_count = defaultdict(int)
    sensors_missing_values_stats = defaultdict(dict)
    sensors_directions = {}


    for file_path in files:

        file = os.path.basename(file_path)
        month_name = file.replace(".json", "")
        # file_path = os.path.join(data_folder, file)

        with open(file_path, "r") as f:
            data = json.load(f)

        percentages = []

        for sensor_id, sensor_data in data.items():
                    
            obs = sensor_data.get("aggr_observations", {})
            values = list(obs.values())

            if not values:
                continue
                
            if (valid_ids is not None) and (not (sensor_id in valid_ids)):
                continue
            
            for t, v in obs.items():
                sensors_time_series[sensor_id][t] = v
                
            if sensor_id not in sensors_month_occurrences:
                sensors_month_occurrences[sensor_id] = [False] * len(name_to_idx_mapping)
            
            if len(sensor_data.get("vehicle_count", {})) > 0:
                sensors_veh_count[sensor_id] += sum(sensor_data["vehicle_count"].values())
            else:
                sensors_veh_count[sensor_id] += sum(values)
                
            sensors_month_occurrences[sensor_id][name_to_idx_mapping[month_name]] = True

            directions = sensor_data.get("directions", [])
            sensors_directions[sensor_id] = directions

            total = len(values)
            missing = sum(1 for v in values if v == 0)
            pct_missing = (missing / total) * 100
            percentages.append(pct_missing)

            sensors_missing_values_stats[sensor_id][month_name] = pct_missing

        distribution = {f"{lo}-{hi}%": 0 for lo, hi in bins}
        for pct in percentages:
            for lo, hi in bins:
                if lo <= pct < hi or (hi == 100 and pct == 100):
                    distribution[f"{lo}-{hi}%"] += 1            
                    break

        monthly_distribution[month_name] = distribution
        monthly_distribution[month_name]["month_idx"] = name_to_idx_mapping[month_name]
        
    return monthly_distribution, sensors_time_series, sensors_month_occurrences, sensors_veh_count, sensors_missing_values_stats, sensors_directions


def get_distr_df(monthly_distribution):
    df = pd.DataFrame(monthly_distribution).T
    df = df.sort_values("month_idx")
    df.drop(columns=["month_idx"], inplace=True)
    
    return df
    
    
def plot_monthly_distr(mnthly_distr_df):
    # --- Plotting ---
    mnthly_distr_df.plot(kind="bar", stacked=True, figsize=(12,6))
    plt.ylabel("Number of Sensors")
    plt.xlabel("Month")
    plt.title("Distribution of sensors by % of zero values")
    # plt.legend(title="% Zero Measurements")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4
    )

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()
    
    
def get_obs_df(sensors_time_series, sensors_month_occurrences, name_to_idx_mapping, from_month = "apr_may_25", to_month = "sep_today"):
    obs_df = pd.DataFrame(sensors_time_series).sort_index()
    # obs_df.fillna(0, inplace=True)
    obs_df.index = pd.to_datetime(obs_df.index, utc=True).tz_convert('Europe/Berlin')
    
    from_idx = name_to_idx_mapping[from_month]
    to_idx = name_to_idx_mapping[to_month]
    consistent_sensors = [k for k, v in sensors_month_occurrences.items() if all(v[from_idx:to_idx])]
    obs_df_consistent = obs_df[consistent_sensors]
    
    return obs_df_consistent


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


def get_poi_feature_matrix(gdf_nodes, df_pois, radius=200):
    
    gdf_pois = gpd.GeoDataFrame(
        df_pois.drop(columns=["id"]),
        geometry=gpd.points_from_xy(df_pois.longitude, df_pois.latitude),
        crs="EPSG:4326"
    )
    
    gdf_nodes = gdf_nodes.to_crs(epsg=3857) # To express distance in meters
    gdf_pois = gdf_pois.to_crs(epsg=3857)

    
    gdf_nodes["buffer"] = gdf_nodes.geometry.buffer(radius)
    gdf_buffers = gdf_nodes[["id", "buffer"]].set_geometry("buffer")
    
    join_result = gpd.sjoin(
        gdf_pois,
        gdf_buffers,
        how="inner",
        predicate="within"
    )
    
    poi_features = (
        join_result.groupby(["id", "category_level_0"])
        .size()
        .unstack(fill_value=0)
        .reindex(gdf_nodes["id"], fill_value=0)
        .reset_index()
    )
        
    return poi_features


def get_weather_data(file_path):
    def classify_weather(row):
        precip = row["PREC"]

        if precip == 0:
            return "Soleggiato"
        elif precip < 5:
            return "Piovoso"
        else:
            return "Fortemente piovoso"
    
    weather_var = pd.read_csv(file_path, delimiter=";")
    weather_var["TIMESTAMP"] = pd.to_datetime(
        weather_var[["ANNO", "MESE", "GIORNO", "ORA"]].rename(
            columns={"ANNO": "year", "MESE": "month", "GIORNO": "day", "ORA": "hour"}
        ),
        format="%Y-%m-%d %H"
    )
    
    weather_var["TIMESTAMP"] = weather_var["TIMESTAMP"].apply(lambda x: x.tz_localize("Europe/Rome").tz_convert("Europe/Berlin"))
    weather_var.drop(columns=["ANNO", "MESE", "GIORNO", "ORA"], inplace=True)
    weather_var = weather_var[:-1] # Remove last row (1st Sep)
    
    weather_var["WEATHER_CLASS"] = weather_var.apply(classify_weather, axis=1)
    
    return weather_var


def get_top_k_zones_pop(zone_file_path, pop_file_path, meta_gdf, k=5):
    def get_nearest(src_points, candidates, k_neighbors=1):
        """Find nearest neighbors for all source points from a set of candidate points"""

        # Create tree from the candidate points
        tree = BallTree(candidates, leaf_size=15, metric='euclidean')

        # Find closest points and distances
        _, indices = tree.query(src_points, k=k_neighbors)

        return indices # Shape (N, K)
    
    zones = gpd.read_file(zone_file_path)
    residents = pd.read_csv(pop_file_path, delimiter=";")
    
    zones_filtered = zones[zones['SEZ21'].isin(residents['Sezioni 2021 attribuite'])].copy()
    mapping = residents.set_index("Sezioni 2021 attribuite")["Somma - Residenti"]
    zones_filtered["POP21"] = zones_filtered["SEZ21"].map(mapping).fillna(zones_filtered["POP21"])
    meta_gdf_proj = meta_gdf.to_crs(zones.crs)
    
    zones_subset = zones.copy()[['POP21', 'SHAPE_Area', 'geometry']]

    zones_subset["centroid"] = zones_subset.geometry.centroid
    centroids = zones_subset.set_geometry("centroid")[["centroid", "POP21"]]

    centroid_coords = np.vstack([
        centroids.geometry.y.values,
        centroids.geometry.x.values
    ]).T


    point_coords = np.vstack([
        meta_gdf_proj.geometry.y.values,
        meta_gdf_proj.geometry.x.values
    ]).T
    
    nearest_idx = get_nearest(point_coords, centroid_coords, k_neighbors=k)

    pop21_array = centroids["POP21"].values
    nearest_pop21 = pop21_array[nearest_idx]
    
    for i in range(k):
        meta_gdf_proj[f"POP21_{i+1}"] = nearest_pop21[:, i]
    
    meta_gdf_proj["geometry"] = meta_gdf["geometry"]
    
    return meta_gdf_proj


def get_road_types_lengths(
    G,
    meta_gdf,
    radius_m=500,
    highway_col="highway"
):

    sensors_enriched = meta_gdf.copy()
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    edges = edges.to_crs(epsg=3003)
    sensors_enriched = sensors_enriched.to_crs(epsg=3003)

    edges[highway_col] = edges[highway_col].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    edges["length_m"] = edges.geometry.length
    feature_rows = []

    for _, sensor in sensors_enriched.iterrows():
        buffer_geom = sensor.geometry.buffer(radius_m)

        nearby_edges = edges[edges.intersects(buffer_geom)]

        lengths = (
            nearby_edges
            .groupby(highway_col)["length_m"]
            .sum()
            .to_dict()
        )

        lengths = {f"road_len_{k}": v for k, v in lengths.items()}
        feature_rows.append(lengths)

    features_df = pd.DataFrame(feature_rows).fillna(0.0)

    sensors_enriched = sensors_enriched.reset_index(drop=True).join(features_df)

    return sensors_enriched