from torch_geometric.data import Data, InMemoryDataset
import torch
import os
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd


class SpatioTemporalData(InMemoryDataset, ABC):
    def __init__(
        self, 
        root,
        name,
        device='cpu',
        history = 12,
        horizon = 12,
        stride=1
    ):
        
        self.name = name
        self.device = device
        self.horizon = horizon
        self.history = history
        self.stride = stride
        super().__init__(root)
        self.data, self.slices, self.raw_data = torch.load(
            self.processed_paths[0],
            map_location=torch.device(device),
            weights_only=None
        )
        
        
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')
    
    
    @property
    def processed_file_names(self):
        name = f'{self.name}.pt'
        return [name]
    
    
    @abstractmethod
    def get_raw_data(self):
        raise NotImplementedError()


    def process(self):
        print('Building the dataset...')
        
        adj, raw_data, mask, static_features, dyn_dict, timestamps = self.get_raw_data()
        first_ts = timestamps[0]
        
        if mask is None:
            mask = torch.ones_like(raw_data)
        
        input_length = self.history
        target_length = self.horizon
        total_seq_len = input_length + target_length
        
        data = []
        
        for ts in range(0, raw_data.size(0) - total_seq_len + 1, self.stride):
            
            idx_input = slice(ts, ts + input_length)
            idx_target = slice(ts + input_length, ts + total_seq_len)
            idx_window = slice(ts, ts + total_seq_len)
            
            if isinstance(adj, dict):
                adj_list_in = self._resample_dyn_features(
                    dyn_features=adj,
                    timestamps=timestamps,
                    indices=idx_input,
                    first_ts=first_ts
                )

                edge_index = [a[0] for a in adj_list_in]
                edge_attr  = [a[1] for a in adj_list_in]
                
                target_edge_ts = timestamps[idx_target]
                target_edge_hour_ms = [
                    int(t_.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
                    for t_ in target_edge_ts
                ]

                adj_list_out = [adj[t_] for t_ in target_edge_hour_ms]

                edge_index_out = [a[0] for a in adj_list_out]
                edge_attr_out  = [a[1] for a in adj_list_out]

            else:
                edge_index, edge_attr = adj
                edge_index_out, edge_attr_out = None, None

            
            x = raw_data[idx_input, :, :]  # Shape: (input_length, num_nodes, 1)
            y = raw_data[idx_target, :, :]  # Shape: (target_length, num_nodes, 1)
            
            mask_in = mask[idx_input, :, :]  # Shape: (input_length, num_nodes, 1)
            mask_out = mask[idx_target, :, :]  # Shape: (target_length, num_nodes, 1)
            
            enc_list = []
            window_timestamps = timestamps[idx_window]

            for t in window_timestamps:
                enc = torch.tensor([
                    np.sin(2 * np.pi * t.hour / 24),
                    np.cos(2 * np.pi * t.hour / 24),
                    np.sin(2 * np.pi * t.day / 31),
                    np.cos(2 * np.pi * t.day / 31),
                    np.sin(2 * np.pi * t.month / 12),
                    np.cos(2 * np.pi * t.month / 12),
                ], dtype=torch.float)
                enc_list.append(enc)
            
            encoded_timestamp = torch.stack(enc_list, dim=0) # Shape: (total_seq_len, 6)
            encoded_timestamp = encoded_timestamp.unsqueeze(0).repeat(x.shape[1], 1, 1) # Shape: (num_nodes, total_seq_len, 6)

            if dyn_dict is not None: 
                dyn_list = self._resample_dyn_features(
                    dyn_features=dyn_dict,
                    timestamps=timestamps,
                    indices=idx_window,
                    first_ts=first_ts
                )
                x_dyn = torch.stack(dyn_list, dim=0)
            else:
                x_dyn = torch.tensor([])
            
            data_kwargs = dict(
                edge_index=edge_index,
                edge_attr=edge_attr,
                x=x.permute(1, 0, 2), # Shape: (num_nodes, input_length, 1) This permutation is necessary for batching in torch_geometric DataLoader
                y=y.permute(1, 0, 2), # Shape: (num_nodes, target_length, 1)
                mask_in=mask_in.permute(1, 0, 2),
                mask_out=mask_out.permute(1, 0, 2),
                poi_static = static_features, # Shape: (num_nodes, F_poi)
                enc_ts = encoded_timestamp,
                x_dyn = x_dyn.permute(1, 0, 2)
            )
            
            if edge_index_out is not None:
                data_kwargs["edge_index_out"] = edge_index_out
                data_kwargs["edge_attr_out"]  = edge_attr_out
                
            data.append(Data(**data_kwargs))          
        
        data, slices = self.collate(data)
        torch.save((data, slices, raw_data), self.processed_paths[0])
    
    
    def _resample_dyn_features(self, dyn_features: dict, timestamps, indices, first_ts):
        assert timestamps is not None, "Dynamic features require timestamps"
        HOUR_MS = 3600000
        dyn_list = []
        first_ts_ms = int(pd.Timestamp(first_ts).timestamp() * 1000)
        first_ts_bin = (first_ts_ms // HOUR_MS) * HOUR_MS
        
        for t in timestamps[indices]:
            t_ms = int(pd.Timestamp(t).timestamp() * 1000)
            t_bin = (t_ms // HOUR_MS) * HOUR_MS
            t_shifted = t_bin - HOUR_MS  # Shift to the previous hour time bin
            t_shifted = max(first_ts_bin, t_shifted)
            dyn_list.append(dyn_features[t_shifted])
        
        return dyn_list