import osmnx as ox
import networkx as nx
from geopy.distance import geodesic
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import os

def get_osmnx_graph(
    traffic_cams,
    graph_dir="graphs",
    graph_name="road_network.graphml"
):
    
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, graph_name)
    
    if os.path.exists(graph_path):
        print(f"Loading cached graph from {graph_path}")
        return ox.load_graphml(graph_path) 
    
    max_lat, min_lat = np.max([cam['latitude'] for cam in traffic_cams]), np.min([cam['latitude'] for cam in traffic_cams])
    max_lon, min_lon = np.max([cam['longitude'] for cam in traffic_cams]), np.min([cam['longitude'] for cam in traffic_cams])

    bbox = geodesic((max_lat, min_lon), (min_lat, max_lon)).km
    bbox = bbox * 1000
    
    center_lat = (max_lat + min_lat) / 2
    center_lon = (max_lon + min_lon) / 2

    G_road = ox.graph_from_point((center_lat, center_lon), dist=bbox/2, network_type='drive')
    
    ox.save_graphml(G_road, graph_path)
    
    return G_road


def extract_directions_from_registry(registry_direction):
    return [p.strip() for p in registry_direction.split(",") if p.strip() != "Tutte"]
    

def is_in_direction(a, b, use_registry = True):
    lat_a, lon_a = a['latitude'], a['longitude']
    lat_b, lon_b = b['latitude'], b['longitude']

    if use_registry:
        directions = extract_directions_from_registry(a.get("reg_dir", []))
    else:
        directions = a.get('direction', [])
        
    if not directions:
        return True # If no direction is specified, consider only distance

    for d in directions:
        if (d == 'NordSud' or d == "SUD") and lat_b < lat_a:
            return True
        elif (d == 'SudNord' or d == "NORD") and lat_b > lat_a:
            return True
        elif (d == 'OvestEst' or d == "EST") and lon_b > lon_a:
            return True
        elif (d == 'EstOvest' or d == "OVEST") and lon_b < lon_a:
            return True
        elif d == 'NordEst' and lat_b > lat_a and lon_b > lon_a:
            return True
        elif d == 'SudOvest' and lat_b < lat_a and lon_b < lon_a:
            return True
        elif d == 'NordOvest' and lat_b > lat_a and lon_b < lon_a:
            return True
        elif d == 'SudEst' and lat_b < lat_a and lon_b > lon_a:
            return True

    return False


def map_lat_lon_to_osmnx_nodes(traffic_cams, osmnx_graph, lat_key = "latitude", lon_key = "longitude"):
    cam_to_osmnx = {}
    for cam in traffic_cams:
        node = ox.distance.nearest_nodes(osmnx_graph, cam[lon_key], cam[lat_key])
        cam_to_osmnx[cam['id']] = (node, cam[lat_key], cam[lon_key])
    
    return cam_to_osmnx


def get_distance_graph(traffic_cams, distance_threshold_km=2, distance_type="geodesic", osmnx_graph=None, directed=False):
    nodes = [(cam['id'], {k:v for k, v in cam.items() if k != "id"}) for cam in traffic_cams]
    
    
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nodes)
    
    if osmnx_graph is not None:
        cam_to_osmnx = map_lat_lon_to_osmnx_nodes(traffic_cams, osmnx_graph)
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            id1, attr1 = nodes[i]
            id2, attr2 = nodes[j]
            
            # check directional constraint
            if directed and not is_in_direction(attr1, attr2):
                continue
            
            if distance_type == "geodesic":
                coord1 = (attr1['latitude'], attr1['longitude'])
                coord2 = (attr2['latitude'], attr2['longitude'])
                distance = geodesic(coord1, coord2).km
            elif distance_type == "road":
                assert osmnx_graph is not None, "OSMnx graph must be provided for road distance calculation."
                node1, node2 = cam_to_osmnx[id1][0], cam_to_osmnx[id2][0]
                try:
                    distance = nx.shortest_path_length(osmnx_graph, node1, node2, weight='length') / 1000
                except nx.NetworkXNoPath:
                    distance = np.inf
                distance = max(1e-3, distance)
            else:
                raise ValueError("Invalid distance_type. Choose 'geodesic' or 'road'.")
            
            if distance <= distance_threshold_km:
                G.add_edge(id1, id2, weight=distance)
    
    nodelist = [cam['id'] for cam in traffic_cams]
    A = nx.to_numpy_array(G, nodelist=nodelist)
    return A, G, nodelist



def networkx_to_gdfs(G):
    # Nodes 
    node_records = []
    for node, attr in G.nodes(data=True):
        node_records.append({
            "id": node,
            **{k: v for k, v in attr.items() if k not in ["latitude", "longitude"]},
            "geometry": Point(attr["longitude"], attr["latitude"])
        })
    gdf_nodes = gpd.GeoDataFrame(node_records, crs="EPSG:4326")

    # Edges
    edge_records = []
    for u, v, attr in G.edges(data=True):
        p1 = Point(G.nodes[u]["longitude"], G.nodes[u]["latitude"])
        p2 = Point(G.nodes[v]["longitude"], G.nodes[v]["latitude"])
        edge_records.append({
            "source": u,
            "target": v,
            **attr,
            "geometry": LineString([p1, p2])
        })
    gdf_edges = gpd.GeoDataFrame(edge_records, crs="EPSG:4326")

    return gdf_nodes, gdf_edges


def get_adj_matrix(traffic_cams, distance_threshold_km = 2, distance_type = "geodesic", osmnx_graph = None, directed=False):
    A, G, nodelist = get_distance_graph(
        traffic_cams, 
        distance_threshold_km, 
        distance_type=distance_type,
        osmnx_graph=osmnx_graph,
        directed=directed
    )
    
    if A.sum() == 0:
        return A, G
    
    non_zeros = np.nonzero(A)
    distances = A[non_zeros].flatten()
    std = distances.std()
        
    A[non_zeros] = np.exp(-np.square(A[non_zeros] / std))
    
    for i, j in zip(*non_zeros):
        node1 = nodelist[i]
        node2 = nodelist[j]
        G[node1][node2]['weight'] = A[i, j]
        
    return A, G


def get_flow_based_adj(traffic_cams, transition_probs_df, avg_travel_times_df, acceptance_threshold=0.01, src_col="junctions", dst_col="dst", prob_col="P_ij"):
    nodes = [(cam['id'], {k:v for k, v in cam.items() if k != "id"}) for cam in traffic_cams]
    
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            id1, _ = nodes[i]
            id2, _ = nodes[j]
            
            prob = transition_probs_df[
                (transition_probs_df[src_col] == id1) & 
                (transition_probs_df[dst_col] == id2)
            ][prob_col]
            
            if not prob.empty and prob.values[0] >= acceptance_threshold:
                G.add_edge(id1, id2, weight=prob.values[0])
    
    
    nodelist = [cam['id'] for cam in traffic_cams]
    A = nx.to_numpy_array(G, nodelist=nodelist)
    
    return A, G



