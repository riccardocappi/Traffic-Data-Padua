# Repo organization
The repo is full of useless things (sorry for that, i will clean it soon :) ). However, the useful folders/files are the following:
## Data
 
- data/prod/pre-process folder contains all the files obtained by preprocessing raw data. Such preprocessing is performed in the preprocessing_prod.ipynb and preprocessing_prod_spire.ipynb files. This folder includes:

- context folder: This folder contains raw data of contextual information, such as POI, ARPAV metheorological variables and census data (SIT_SEZIONI_2021).

- loops folder: Contains information related to time series detected by loops sensors on ring roads.

- plate_hash folder: Contains the aggregated flow statistics derived from vehicles plate hashes.

- road_network folder: Contains the underlying OSMNX road network object.

- traffic_cams folder: Contains information related to time series detected by TrafficCams

- traffic_cams_by_junc folder: Contains times series of traffic cams clustered by position.



## Torch Data Loaders
The data_loaders/ folder contains the code of Pytorch Geometric Data Loaders. Time series data are loaded with a sliding window style, thats is, each entry of the loader contains: 
    
    - x: a given history of observations, let's say 1 hour (6 10-min steps).

    - y: the prediction horizon, let's say the next hour after x. These are the values to be predicted by the model.

    - edge_index and edge_attrs: graph topology

    - masks: binary masks for unusual long series of consecutive zero measurements (dynamically created).

    - Other fields related to contextual information for each node.

Data Loaders are saved to the given root-name path.
