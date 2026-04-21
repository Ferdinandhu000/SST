import os
import sys
from typing import List, Tuple, Dict, Any, Literal
import shutil
from tqdm import tqdm
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import h5py

from cfd.sensors import LHS, AroundCylinder
from cfd.embedding import Voronoi, SoftVoronoi, Mask, Vector
from common.plotting import plot_frame

# ──────────────────────────────────────────────
# Dataset configuration for sst_weekly.mat
# ──────────────────────────────────────────────
SST_MAT_PATH = 'data/sst_weekly.mat'
TRAIN_END = 1500   # first 1500 weeks → training
# remaining 414 weeks (index 1500–1913) → testing


def _load_sst_data(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SST data from sst_weekly.mat (HDF5 / MATLAB v7.3).

    Parameters
    ----------
    split : str
        'train' or 'test'

    Returns
    -------
    data_np : np.ndarray, shape (T, 180, 360), NaN replaced with 0.0
    nan_mask : np.ndarray, shape (180, 360), bool  (True = land / NaN)
    """
    with h5py.File(SST_MAT_PATH, 'r') as f:
        sst_raw = f['sst'][()]                     # (1914, 64800)

    # MATLAB flattens column-major (F-order), so reading it sequentially in C-order
    # as 180x360 creates mathematically disjoint layouts. We must reshape to (360, 180),
    # transpose it to recover the true (180, 360) geographic layout, and flip it 
    # vertically so that Row 0 natively corresponds to the top (North Pole).
    sst_3d = sst_raw.reshape(sst_raw.shape[0], 360, 180).transpose(0, 2, 1).astype(np.float32)
    sst_3d = np.flip(sst_3d, axis=1).copy()

    # Build a consistent land mask from the first frame (all frames share the same NaN pattern)
    nan_mask = np.isnan(sst_3d[0])                 # (180, 360), True = land

    # Split
    if split == 'train':
        sst_3d = sst_3d[:TRAIN_END]
    else:
        sst_3d = sst_3d[TRAIN_END:]

    # Replace NaN (land) with 0.0 in-place to avoid an extra large memory copy.
    np.nan_to_num(sst_3d, nan=0.0, copy=False)

    return sst_3d, nan_mask


class DatasetMixin:

    def load2tensor(self, root: str) -> torch.Tensor:
        """
        Load SST weekly data and return as tensor (T, 1, 180, 360).
        `root` is used only to decide train vs test split.
        """
        split = 'train' if 'train' in root else 'test'
        data_np, _ = _load_sst_data(split)          # (T, 180, 360)
        data = torch.from_numpy(data_np).float().unsqueeze(1)  # (T, 1, 180, 360)
        return data

    def prepare_sensor_timeframes(self, n_chunks: int) -> torch.IntTensor:
        sensor_timeframes: torch.Tensor = (
            torch.tensor(self.init_sensor_timeframes) + torch.arange(n_chunks).unsqueeze(1)
        )
        return sensor_timeframes.int()

    def prepare_fullstate_timeframes(
        self,
        n_chunks: int,
        seed: int | None = None,
        init_fullstate_timeframes: List[int] | None = None,
    ) -> torch.IntTensor:
        if seed is None and init_fullstate_timeframes is not None:
            fullstate_timeframes: torch.Tensor = (
                torch.arange(n_chunks).unsqueeze(1) 
                + torch.tensor(init_fullstate_timeframes).unsqueeze(0)
            )
            return fullstate_timeframes
        else:
            fullstate_timeframes: torch.Tensor = torch.empty((n_chunks, self.n_fullstate_timeframes_per_chunk), dtype=torch.int)
            for chunk_idx in range(n_chunks):
                torch.random.manual_seed(seed + chunk_idx)
                if self.future_prediction_range is not None:
                    range_start, range_end = self.future_prediction_range
                    offset = max(self.init_sensor_timeframes)
                    range_size = range_end - range_start + 1
                    random_init_timeframes = offset + range_start + torch.randperm(n=range_size)[:self.n_fullstate_timeframes_per_chunk].sort()[0]
                else:
                    random_init_timeframes: torch.Tensor = torch.randperm(n=max(self.init_sensor_timeframes))[:self.n_fullstate_timeframes_per_chunk].sort()[0]
                fullstate_timeframes[chunk_idx] = random_init_timeframes + chunk_idx
            return fullstate_timeframes


class CFDDataset(Dataset, DatasetMixin):

    def __init__(
        self, 
        root: str, 
        init_sensor_timeframes: List[int],
        future_prediction_range: List[int] | None,
        n_fullstate_timeframes_per_chunk: int,
        n_samplings_per_chunk: int,
        resolution: Tuple[int, int],
        n_sensors: int,
        dropout_probabilities: List[float],
        noise_level: float,
        sensor_generator: Literal['LHS', 'AroundCylinder'], 
        embedding_generator: Literal['Voronoi', 'SoftVoronoi', 'Mask', 'Vector'],
        init_fullstate_timeframes: List[int] | None,
        seed: int,
        write_to_disk: bool = True,
    ) -> None:
        
        super().__init__()
        
        self.root: str = root
        self.init_sensor_timeframes: List[int] = init_sensor_timeframes
        self.future_prediction_range: List[int] | None = future_prediction_range
        self.n_fullstate_timeframes_per_chunk: int = n_fullstate_timeframes_per_chunk
        self.n_samplings_per_chunk: int = n_samplings_per_chunk
        self.resolution: Tuple[int, int] = resolution
        self.n_sensors: int = n_sensors
        self.dropout_probabilities: List[float] = dropout_probabilities
        self.noise_level: float = noise_level
        self.init_fullstate_timeframes: List[int] | None = init_fullstate_timeframes
        self.seed: int = seed
        self.is_random_fullstate_frames: bool = init_fullstate_timeframes is None
        self.write_to_disk: bool = write_to_disk

        self.H, self.W = resolution
        self.n_sensor_timeframes_per_chunk: int = len(init_sensor_timeframes)

        suffix = 'train' if 'train' in root else 'test'
        self.dest: str = os.path.join('tensors', suffix)
        self.sensor_timeframes_dest: str = os.path.join(self.dest, 'sensor_timeframes')
        self.sensor_values_dest: str = os.path.join(self.dest, 'sensor_values')
        self.fullstate_timeframes_dest: str = os.path.join(self.dest, 'fullstate_timeframes')
        self.fullstate_values_dest: str = os.path.join(self.dest, 'fullstate_values')
        self.sensor_positions_dest: str = os.path.join(self.dest, 'sensor_positions')
        self.metadata_dest: str = os.path.join(self.dest, 'metadata')

        if sensor_generator == 'LHS':
            self.sensor_generator = LHS(n_sensors=n_sensors)
        else:
            self.sensor_generator = AroundCylinder(n_sensors=n_sensors)
        
        self.sensor_generator.seed = seed
        self.sensor_generator.resolution = resolution

        self.case_names: List[str] = []
        self.sampling_ids: List[int] = []

        # 1. Prepare Mask and Sensors
        if self.write_to_disk:
            split = 'train' if 'train' in root else 'test'
            _, nan_mask = _load_sst_data(split)      # nan_mask: (180, 360), bool

            # mask convention: 1 = land (masked out), 0 = ocean (valid)
            self.mask = torch.from_numpy(nan_mask.astype(np.float32)).int()
            self.case_name = f'sst_weekly_{split}'
        else:
            self.__load_metadata()
            mask_path = os.path.join(self.metadata_dest, 'mask.pt')
            self.mask = torch.load(mask_path, weights_only=True) if os.path.exists(mask_path) else None
            self.case_name = "sst_weekly"
        
        # Pass mask to sensor generator so sensors are placed only on ocean
        self.sensor_generator.mask = self.mask
        self.sensor_positions = self.sensor_generator()

        # 2. Initialize Embedding Generator
        if embedding_generator == 'Mask':
            self.embedding_generator = Mask(resolution=resolution, sensor_positions=self.sensor_positions, dropout_probabilities=dropout_probabilities, noise_level=noise_level)
        elif embedding_generator == 'Voronoi':
            self.embedding_generator = Voronoi(resolution=resolution, sensor_positions=self.sensor_positions, dropout_probabilities=dropout_probabilities, noise_level=noise_level)
        elif embedding_generator == 'SoftVoronoi':
            self.embedding_generator = SoftVoronoi(resolution=resolution, sensor_positions=self.sensor_positions, dropout_probabilities=dropout_probabilities, noise_level=noise_level)
        else:
            self.embedding_generator = Vector(resolution=resolution, sensor_positions=self.sensor_positions, dropout_probabilities=dropout_probabilities, noise_level=noise_level)

        # 3. Finalize Write
        if self.write_to_disk:
            self.__write2disk()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix: str = f'_{self.case_names[idx]}_{self.sampling_ids[idx]}_'
        suffix: str = str(idx).zfill(6)
        st = torch.load(os.path.join(self.sensor_timeframes_dest, f'st{prefix}{suffix}.pt'), weights_only=True)
        sv = torch.load(os.path.join(self.sensor_values_dest, f'sv{prefix}{suffix}.pt'), weights_only=True).float()
        ft = torch.load(os.path.join(self.fullstate_timeframes_dest, f'ft{prefix}{suffix}.pt'), weights_only=True)
        fv = torch.load(os.path.join(self.fullstate_values_dest, f'fv{prefix}{suffix}.pt'), weights_only=True).float()
        return st, sv, ft, fv, self.case_names[idx], self.sampling_ids[idx]
    
    def __len__(self) -> int:
        return len(self.case_names)

    def __load_metadata(self) -> None:
        metadata_file = os.path.join(self.metadata_dest, 'metadata.json')
        if not os.path.exists(metadata_file):
            return
        with open(metadata_file, 'r') as f:
            records = json.load(f)
        for record in records:
            self.case_names.append(record['case_name'])
            self.sampling_ids.append(record['sampling_id'])

    def __write2disk(self) -> None:
        if os.path.isdir(self.dest): shutil.rmtree(self.dest)
        for d in [self.sensor_timeframes_dest, self.sensor_values_dest, self.fullstate_timeframes_dest, self.fullstate_values_dest, self.sensor_positions_dest, self.metadata_dest]:
            os.makedirs(d, exist_ok=True)

        torch.save(self.mask, os.path.join(self.metadata_dest, 'mask.pt'))
        
        # Load SST data from .mat file
        split = 'train' if 'train' in self.root else 'test'
        data_np, _ = _load_sst_data(split)           # (T, 180, 360), NaN→0
        data = torch.from_numpy(data_np).float().unsqueeze(1)  # (T, 1, 180, 360)

        torch.save(self.sensor_positions, os.path.join(self.sensor_positions_dest, 'pos.pt'))

        total_timeframes = data.shape[0]
        max_frame_idx = max(self.init_sensor_timeframes)
        if self.init_fullstate_timeframes: max_frame_idx = max(max_frame_idx, max(self.init_fullstate_timeframes))
        if self.future_prediction_range: max_frame_idx = max(max_frame_idx, max(self.init_sensor_timeframes) + self.future_prediction_range[1])

        n_chunks = total_timeframes - max_frame_idx
        sensor_timeframes = self.prepare_sensor_timeframes(n_chunks)
        
        sensor_timeframes_list, fullstate_timeframes_list, running_index = [], [], 0
        for sampling_id in range(self.n_samplings_per_chunk):
            fullstate_timeframes = self.prepare_fullstate_timeframes(n_chunks, seed=self.seed + sampling_id if self.is_random_fullstate_frames else None, init_fullstate_timeframes=self.init_fullstate_timeframes)
            for idx in tqdm(range(n_chunks), desc=f'SST Chunking ({self.case_name}): '):
                self.case_names.append(self.case_name); self.sampling_ids.append(sampling_id)
                prefix, suffix = f'_{self.case_name}_{sampling_id}_', str(running_index).zfill(6)
                sensor_timeframes_list.append(sensor_timeframes[idx].tolist())
                torch.save(sensor_timeframes[idx].clone(), os.path.join(self.sensor_timeframes_dest, f'st{prefix}{suffix}.pt'))
                
                current_timeframes = sensor_timeframes[idx]
                embedding_input = data[current_timeframes].unsqueeze(0)
                sensor_embedding = self.embedding_generator(data=embedding_input, seed=self.seed + idx).squeeze(0)
                torch.save(sensor_embedding.clone(), os.path.join(self.sensor_values_dest, f'sv{prefix}{suffix}.pt'))
                
                # PLOT the first generated frame for visual inspection of the sensors
                if running_index == 0:
                    split_name = 'train' if 'train' in self.root else 'test'
                    embedding_name = self.embedding_generator.__class__.__name__
                    plot_frame(
                        sensor_positions=self.sensor_positions,
                        sensor_frame=sensor_embedding[0, 0] if sensor_embedding.ndim == 4 else None,
                        fullstate_frame=data[current_timeframes[0]][0],
                        mask=self.mask,
                        title=f"{split_name.upper()} First Frame Embedding ({embedding_name})",
                        filename=f"generated_tensor_{split_name}_frame0"
                    )
                
                fullstate_timeframes_list.append(fullstate_timeframes[idx].tolist())
                torch.save(fullstate_timeframes[idx].clone(), os.path.join(self.fullstate_timeframes_dest, f'ft{prefix}{suffix}.pt'))
                fullstate_sample = data[fullstate_timeframes[idx]]
                torch.save(fullstate_sample.clone(), os.path.join(self.fullstate_values_dest, f'fv{prefix}{suffix}.pt'))
                running_index += 1
            torch.cuda.empty_cache()
        
        records = [{'case_name': c, 'sampling_id': s, 'sensor_timeframes': st, 'fullstate_timeframes': ft} for c, s, st, ft in zip(self.case_names, self.sampling_ids, sensor_timeframes_list, fullstate_timeframes_list)]
        with open(os.path.join(self.metadata_dest, 'metadata.json'), 'w') as f: json.dump(records, f, indent=2)
