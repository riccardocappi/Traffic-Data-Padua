# Repo organization

## Data
 
- data/prod/pre-process folder contains all the raw files. This folder includes:

- context folder: This folder contains raw data of contextual information, such as POI, ARPAV metheorological variables and census data.

- plate_hash folder: Contains the flow statistics derived from vehicles plate hashes.

- plate_hash_by_junc: Contains the flow statistics derived from vehicles plate hashes at junction-level

- road_network folder: Contains the underlying OSMNX road network object.

- traffic_cams folder: Contains time series related to time series detected by TrafficCams

- traffic_cams_by_junc folder: Contains times series of traffic cams clustered by position.


# Requirements
- Python 3.12.0
- CUDA 11.8
- Conda

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/riccardocappi/Traffic-Data-Padua.git
cd Traffic-Data-Padua
```

### 2. Create and activate the conda environment

```bash
conda create -n traffic_data_pd python=3.12.0 
conda activate traffic_data_pd
```

### 3. Install requirements
```bash
source install_new.sh
```

# Usage Notes

## Basic Usage
 
```python
from data_loaders.TrafficData import TrafficData
 
dataset = TrafficData(
    root="./data",
    name="traffic_pd",
)
 
print(f"Number of samples: {len(dataset)}")
 
sample = dataset[0]
print(sample.x.shape)        # (N, history, 1)  — input traffic flow
print(sample.y.shape)        # (N, horizon, 1)  — target traffic flow
print(sample.edge_index)     # graph connectivity
```

## Constructor Parameters
 
| Parameter | Type | Default | Description |
|---|---|---|---|
| `root` | `str` | — | Root directory for raw and processed data |
| `name` | `str` | — | Dataset identifier; processed file is cached as `<root>/<name>/processed/<name>.pt` |
| `device` | `str` | `'cpu'` | PyTorch device for tensor loading |
| `history` | `int` | `6` | Input sequence length (× 10 min intervals, so 6 = 1 hour) |
| `horizon` | `int` | `6` | Forecast horizon |
| `stride` | `int` | `1` | Step size between consecutive sliding windows |
| `zero_run_threshold` | `int` | `6` | Consecutive zeros before a sensor reading is masked out |
| `flow_adj` | `bool` | `True` | Use vehicle-flow transition probabilities for edges; if `False`, falls back to road-network proximity |
| `dyn_adj` | `bool` | `False` | Use time-varying (hourly) adjacency matrices; requires `flow_adj=True` |
| `flow_threshold` | `float` | `0.0` | Minimum transition probability `P_ij` for an edge to be included |
| `use_avg_travel_times` | `bool` | `False` | Use mean travel time as edge weight instead of `P_ij` |
| `nan_values_handling` | `str` | `'rm'` | `'zero'` replaces NaNs with 0; `'rm'` drops any timestep containing a NaN |
| `radius` | `int` | `500` | Radius in metres for POI and road-type feature enrichment |
| `k_zones` | `int` | `5` | Number of nearest population zones per sensor node |
| `k_km` | `int` | `2` | Distance threshold (km) for proximity-based adjacency (`flow_adj=False`) |
 
---


## Each Dataset Sample
 
Each item returned by `dataset[i]` is a `torch_geometric.data.Data` object with the following attributes:
 
| Attribute | Shape | Description |
|---|---|---|
| `x` | `(N, history, 1)` | Input traffic-flow readings |
| `y` | `(N, horizon, 1)` | Target traffic-flow readings |
| `mask_in` | `(N, history, 1)` | Validity mask for the input window (`0` = inactive sensor) |
| `mask_out` | `(N, horizon, 1)` | Validity mask for the forecast window |
| `edge_index` | `(2, E)` or list | Graph connectivity in COO format; list of per-step tensors when `dyn_adj=True` |
| `edge_attr` | `(E, 1)` or list | Edge weights (transition probability or travel time) |
| `edge_index_out` | list | Per-step edge indices for the horizon (`dyn_adj=True` only) |
| `edge_attr_out` | list | Per-step edge attributes for the horizon (`dyn_adj=True` only) |
| `poi_static` | `(N, F_poi)` | Static POI category counts within `radius` metres of each node |
| `enc_ts` | `(N, history+horizon, 6)` | Cyclic sin/cos encodings of hour, day, and month |
| `x_dyn` | `(N, history+horizon, F_dyn)` | Hourly precipitation feature broadcast to all nodes; empty tensor if unavailable |
 
---


## Using with a DataLoader
 
```python
from torch_geometric.loader import DataLoader
 
loader = DataLoader(dataset, batch_size=32, shuffle=True)
 
for batch in loader:
    print(batch)
```