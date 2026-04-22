from .SpatioTemporalGraph import SpatioTemporalData
import pandas as pd
import geopandas as gpd
import torch
import numpy as np
from scripts.get_adj_matrix import get_adj_matrix, networkx_to_gdfs
import os
from sklearn.neighbors import BallTree
import osmnx as ox
import json
from shapely.geometry import LineString


class TrafficData(SpatioTemporalData):
    """
        TrafficData
        ===========

        Concrete subclass of :class:`SpatioTemporalData` for urban traffic-flow
        forecasting.  Reads sensor measurements, spatial metadata, and contextual
        features from local CSV / Shapefile / GraphML sources, constructs a graph,
        and assembles a ready-to-use PyG dataset.

        Parameters
        ----------
        root : str
            Root directory for raw and processed data storage.
        name : str
            Dataset identifier (used for processed-file naming).
        device : str, optional
            PyTorch device string. Default: ``'cpu'``.
        history : int, optional
            Input sequence length (number of 5-min intervals). Default: ``12``
            (≈ 1 hour).
        horizon : int, optional
            Forecast horizon (number of 5-min intervals). Default: ``12``.
        stride : int, optional
            Window stride. Default: ``1``.
        zero_run_threshold : int, optional
            Consecutive zeros that trigger a masking event in the validity mask.
            Sensor readings showing ``>= zero_run_threshold`` consecutive zeros are
            assumed to represent sensor inactivity and are masked out.
            Default: ``12``.
        flow_adj : bool, optional
            If ``True``, build the graph from vehicle-flow transition probabilities
            (or average travel times); if ``False``, fall back to road-network
            proximity. Default: ``True``.
        dyn_adj : bool, optional
            If ``True``, use time-varying (hourly) adjacency matrices derived from
            hourly transition probabilities. Requires ``flow_adj=True``.
            Default: ``False``.
        flow_threshold : float, optional
            Minimum transition probability ``P_ij`` for an edge to be included when
            using flow-based adjacency. Default: ``0.0``.
        use_avg_travel_times : bool, optional
            If ``True``, use mean travel time as the edge weight instead of the
            transition probability ``P_ij``. Default: ``False``.
        nan_values_handling : {'zero', 'rm'}, optional
            Strategy for missing sensor readings.

            * ``'zero'`` - Replace NaN with 0.
            * ``'rm'``   - Drop any time step that contains at least one NaN.

            Default: ``'zero'``.
    """
    def __init__(
        self, 
        root, 
        name,
        device='cpu', 
        history=12, 
        horizon=12, 
        stride=1,
        zero_run_threshold = 12,
        flow_adj = True,
        dyn_adj = False,
        flow_threshold = 0.0,
        use_avg_travel_times = False,
        nan_values_handling = "zero"
    ):

        self.zero_run_threshold = zero_run_threshold
        
        assert nan_values_handling in ["zero", "rm"], "nan_values_handling must be either 'zero' (replace NaNs with zeros) or 'rm' (remove rows with NaNs)."
        
        self.nan_values_handling = nan_values_handling
        self.flow_threshold = flow_threshold
        self.use_avg_travel_times = use_avg_travel_times
        
        self.flow_adj = flow_adj
        self.dyn_adj = dyn_adj
        
        super().__init__(root, name, device, history, horizon, stride)
    
    
    def get_raw_data(self):
        return self.get_raw_data_pd()
        
        
    def get_raw_data_pd(self):
        time_series_df, nodes_shp, nodes_json = self.get_sensors_data()
        
        if self.nan_values_handling == "zero":
            time_series_df = time_series_df.fillna(0)
        elif self.nan_values_handling == "rm":
            time_series_df = time_series_df.dropna(axis=0, how="any")
        
        timestamps_ms = pd.to_datetime(time_series_df.index).astype(int) // 10**6
        timestamps = pd.to_datetime(time_series_df.index)
        
        if not self.flow_adj:
            edges_shp = self.get_dist_connectivity(
                meta_json=nodes_json,
                k = 2
            )
        else:
            edges_shp = self.get_flow_connectivity(
                meta_json=nodes_json,
                dyn_adj=self.dyn_adj,
                use_avg_travel_times = self.use_avg_travel_times,
                threshold=self.flow_threshold
            )
        
        poi_static = self.get_poi_feature_matrix(nodes_shp, radius=500)
        
        zones_pop = self.get_top_k_zones_pop(nodes_shp, k = 5)
        zones_pop = zones_pop[["id", "POP21_1", "POP21_2", "POP21_3", "POP21_4", "POP21_5"]]
        
        roads = self.get_road_types_lengths(nodes_shp, radius_m=500)
        roads = roads.drop(columns = ["reg_dir", "latitude", "longitude", "geometry"])
        
        time_series_df, nodes_shp, poi_static, zones_pop, roads = self.align_node_ids(time_series_df, nodes_shp, poi_static, zones_pop, roads)
        
        mask = self.get_mask(time_series_df, self.zero_run_threshold) # (T, N)
        
        X = time_series_df.values # (T, N) time series data
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # (T, N, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(-1) # (T, N, 1)
        
        X_poi = torch.tensor(poi_static.values, dtype=torch.float32) # (N, F_poi)
        X_zones = torch.tensor(zones_pop.values, dtype=torch.float32)
        X_roads = torch.tensor(roads.values, dtype=torch.float32)
        X_roads = self.normalize_road_lengths(X_roads)
        
        X_static = torch.cat([X_poi, X_zones, X_roads], dim=-1)
        
        weather_data = self.get_weather_data() 
        weather_data["TIMESTAMP"] = pd.to_datetime(weather_data["TIMESTAMP"])
        weather_data = weather_data.sort_values("TIMESTAMP")
        N = X.shape[1]
        time_ms = (weather_data["TIMESTAMP"].astype("int64") // 10**6).to_numpy()
        prec = torch.tensor(weather_data["PREC"].to_numpy(), dtype=torch.float32)
        prec_expanded = prec.view(-1, 1, 1).expand(-1, N, 1)    # Shape (T_hour, N, 1)
        prec_by_time = dict(zip(time_ms, prec_expanded))
        
        node_id_map = {nid: i for i, nid in enumerate(nodes_shp["id"])}            
        
        if not self.dyn_adj:
            src = edges_shp["source"].map(node_id_map).values
            dst = edges_shp["target"].map(node_id_map).values
        
            edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
            edge_attr = torch.tensor(edges_shp["weight"].values, dtype=torch.float32).unsqueeze(-1)
            
            adj = (edge_index, edge_attr)
        else:
            assert "time_bin" in edges_shp.columns, "Dynamic adjacency requires 'time_bin' column in edges shapefile."
            adj = {}
            t_prev = -1
            HOUR_MS = 3600000
            
            last_edge_idx = None
            last_edge_attr = None
            
            for tbin, group in edges_shp.groupby("time_bin"):
                src = group["source"].map(node_id_map).values
                dst = group["target"].map(node_id_map).values
                
                current_edge_idx = torch.tensor(np.array([src, dst]), dtype=torch.long)
                current_edge_attr = torch.tensor(group["weight"].values, dtype=torch.float32).unsqueeze(-1)
                
                t_ms = int(pd.to_datetime(tbin).timestamp() * 1000)
                
                if t_prev != -1:    # Handling missing time_bin
                    t_gap = t_prev + HOUR_MS
                    while t_gap < t_ms:
                        # print(t_gap)
                        adj[t_gap] = (last_edge_idx, last_edge_attr)
                        t_gap += HOUR_MS

                adj[t_ms] = (current_edge_idx, current_edge_attr)
                t_prev = t_ms
                last_edge_idx = current_edge_idx
                last_edge_attr = current_edge_attr
        
        return adj, X, mask, X_static, prec_by_time, timestamps
        
    
    def get_sensors_data(self):
        f_time_series = "./data/prod/pre-process/traffic_cams_by_junc/pd_time_series_group_by_junc.csv"
        f_shp = "./data/prod/pre-process/traffic_cams_by_junc/gpd/traffic_cam_metadata_by_junc.shp"
        f_json = "./data/prod/pre-process/traffic_cams_by_junc/traffic_cam_metadata_by_junc.json"

        time_series_df = pd.read_csv(f_time_series, index_col=0)
        time_series_df.index = pd.to_datetime(time_series_df.index, utc=True).tz_convert("Europe/Rome")
        meta_gdf = gpd.read_file(f_shp)
        with open(f_json, "r") as f:
            meta_json = json.load(f)

        return time_series_df, meta_gdf, meta_json
    
    
    def get_dist_connectivity(self, meta_json, k = 2):
        graph_path = "./data/prod/pre-process/road_network/osmnx.graphml"
        G_road = ox.load_graphml(graph_path)
        
        _, G = get_adj_matrix(
            meta_json, 
            distance_threshold_km=k, 
            distance_type="road",
            osmnx_graph=G_road, 
            directed=True
        )
        
        _, edges = networkx_to_gdfs(G)
        
        return edges


    def get_flow_connectivity(self, meta_json, dyn_adj=False, use_avg_travel_times = False, threshold=0.1):
        
        if not dyn_adj:
            transition_probs = pd.read_csv("./data/prod/pre-process/plate_hash_by_junc/transition_probs.csv")
            avg_travel_times = pd.read_csv("./data/prod/pre-process/plate_hash_by_junc/avg_travel_time.csv")
        else:
            transition_probs = pd.read_csv("./data/prod/pre-process/plate_hash_by_junc/transition_probs_hour.csv")
            avg_travel_times = pd.read_csv("./data/prod/pre-process/plate_hash_by_junc/avg_travel_time_hour.csv")
        
        assert avg_travel_times[["from", "to"]].to_numpy().tolist() == transition_probs[["from", "to"]].to_numpy().tolist()
        
        source_adj = transition_probs if not use_avg_travel_times else avg_travel_times
        weight_col = "P_ij" if not use_avg_travel_times else "mean"
        source_adj["time_bin"] = pd.to_datetime(source_adj["time_bin"].values, utc=True).tz_convert("Europe/Rome")
        
        if self.nan_values_handling == "zero":
            source_adj = source_adj.fillna(0)
        elif self.nan_values_handling == "rm":
            source_adj = source_adj.dropna(axis=0, how="any")
        
        edges = (
            source_adj
                .loc[transition_probs["P_ij"] > threshold]      # keep rows above threshold
                .rename(columns={
                    "from": "source",
                    "to": "target",
                    weight_col: "weight"
                })
                .reset_index(drop=True)
        )
        
        junction_coords = {
            j["id"]: (j["longitude"], j["latitude"])
            for j in meta_json
        }

        edges["geometry"] = edges.apply(
            lambda row: LineString([
                junction_coords[row["source"]],
                junction_coords[row["target"]],
            ]),
            axis=1
        )

        edges_gdf = gpd.GeoDataFrame(
            edges,
            geometry="geometry",
            crs="EPSG:4326"
        )
        
        return edges_gdf
    
    
    def get_poi_feature_matrix(self, meta_gdf, radius = 500):
        df_pois = pd.read_csv("./data/prod/pre-process/context/filtered_grouped_poi.csv", index_col=0)
        gdf_pois = gpd.GeoDataFrame(
            df_pois.drop(columns=["id"]),
            geometry=gpd.points_from_xy(df_pois.longitude, df_pois.latitude),
            crs="EPSG:4326"
        )

        meta_gdf = meta_gdf.to_crs(epsg=3003) # To express distance in meters
        gdf_pois = gdf_pois.to_crs(epsg=3003)
        
        meta_gdf["buffer"] = meta_gdf.geometry.buffer(radius)
        gdf_buffers = meta_gdf[["id", "buffer"]].set_geometry("buffer")

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
            .reindex(meta_gdf["id"], fill_value=0)
            .reset_index()
        )
            
        return poi_features
    
    
    
    def get_top_k_zones_pop(self, meta_gdf, k=5):
        def get_nearest(src_points, candidates, k_neighbors=1):
            tree = BallTree(candidates, leaf_size=15, metric='euclidean')
            _, indices = tree.query(src_points, k=k_neighbors)
            return indices # Shape (N, K)
        
        zone_file_path = "./data/prod/pre-process/context/SIT_SEZIONI_2021/SIT_SEZIONI_2021.shp"
        pop_file_path = "./data/prod/pre-process/context/SIT_SEZIONI_2021/residenti_x_sezioneISTAT-2021.csv"
        
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
        self,
        meta_gdf,
        radius_m=500,
        highway_col="highway"
    ):
        
        graph_path = "./data/prod/pre-process/road_network/osmnx.graphml"
        G = ox.load_graphml(graph_path)
        
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
        sensors_enriched = sensors_enriched.drop(columns = ["road_len_unclassified"])
        
        return sensors_enriched
    
    
    def get_weather_data(self):
        def classify_weather(row):
            precip = row["PREC"]

            if precip == 0:
                return "Sunny"
            elif precip < 5:
                return "Rainy"
            else:
                return "Strongly rainy"

        file_path = "./data/prod/pre-process/context/arpav_pd.csv"
        weather_var = pd.read_csv(file_path, delimiter=";")
        weather_var["TIMESTAMP"] = pd.to_datetime(
            weather_var[["ANNO", "MESE", "GIORNO", "ORA"]].rename(
                columns={"ANNO": "year", "MESE": "month", "GIORNO": "day", "ORA": "hour"}
                )
        )

        weather_var["TIMESTAMP"] = weather_var["TIMESTAMP"].dt.tz_localize("Etc/GMT-1")
        weather_var["TIMESTAMP"] = weather_var["TIMESTAMP"].dt.tz_convert("Europe/Rome")     
           
        weather_var.drop(columns=["ANNO", "MESE", "GIORNO", "ORA"], inplace=True)
    
        weather_var["WEATHER_CLASS"] = weather_var.apply(classify_weather, axis=1)

        return weather_var
    
    # def get_weather_data(self):
    #     weather_data = pd.read_csv("./data/prod/pre-process/context/open-meteo-45.40N11.88E18m.csv")
    #     return weather_data
    
    
    def get_mask(self, df, zero_run_threshold = 12):
        # Create a binary mask of shape (T, N) with 0 values for unusual long series of consecutive zero measurements (probably due to inactive sensors) 
        mask = pd.DataFrame(1, index=df.index, columns=df.columns)

        for col in df.columns:
            z = (df[col] == 0)
            group_id = (z != z.shift()).cumsum() # assigns a unique integer ID to each contiguous "run" of equal boolean values
            run_lengths = group_id.map(group_id.value_counts()) # replace each entry of group_id with the size of its run
            
            long_zero_runs = z & (run_lengths >= zero_run_threshold) # If zero and it is part of a long run, mask it
            mask.loc[long_zero_runs, col] = 0
            
        return mask.values
        
        
    def align_node_ids(self, time_series_df, nodes_meta, poi_static, zones_pop, roads):
        ref_ids = time_series_df.columns.astype(str).tolist()

        nodes_meta["id"] = nodes_meta["id"].astype(str)
        poi_static["id"] = poi_static["id"].astype(str)
        zones_pop["id"] = zones_pop["id"].astype(str)
        roads["id"] = roads["id"].astype(str)

        nodes_meta = nodes_meta[nodes_meta["id"].isin(ref_ids)]
        poi_static = poi_static[poi_static["id"].isin(ref_ids)]
        zones_pop = zones_pop[zones_pop["id"].isin(ref_ids)]
        roads = roads[roads["id"].isin(ref_ids)]

        nodes_meta = nodes_meta.set_index("id").loc[ref_ids].reset_index()
        poi_static = poi_static.set_index("id").loc[ref_ids].reset_index()
        zones_pop = zones_pop.set_index("id").loc[ref_ids].reset_index()
        roads = roads.set_index("id").loc[ref_ids].reset_index()
        
        return time_series_df, nodes_meta, poi_static.drop(columns=["id"]), zones_pop.drop(columns = ["id"]), roads.drop(columns = ["id"])
        
    
    def normalize_road_lengths(self, roads):
        mean = roads.mean(dim=0, keepdim=True)
        std  = roads.std(dim=0, keepdim=True) + 1e-6
        x_norm = (roads - mean) / std
        
        return x_norm