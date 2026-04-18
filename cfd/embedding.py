from abc import ABC, abstractmethod
from typing import List, Tuple
from functools import cached_property
import random

import torch


class SensorEmbedding(ABC):

    def __init__(
        self,
        resolution: Tuple[int, int],
        sensor_positions: torch.Tensor, 
        dropout_probabilities: List[float] = [], 
        noise_level: float = 0.,
    ):
        self.resolution: Tuple[int, int] = resolution
        self.H, self.W = self.resolution
        assert sensor_positions.ndim == 2 and sensor_positions.shape[1] == 2
        self.sensor_positions: torch.Tensor = sensor_positions.float()
        assert sum(dropout_probabilities) <= 1, "Dropout probabilities must sum to less than 1"
        self.n_max_dropout_sensors: int = len(dropout_probabilities)
        self.dropout_probabilities: List[float] = [1. - sum(dropout_probabilities)] + dropout_probabilities
        self.noise_level: float = noise_level
        self.S: int = sensor_positions.shape[0]  # Number of sensors

    @abstractmethod
    def __call__(self, data: torch.Tensor, seed: int = 0) -> torch.Tensor:
        pass


class Voronoi(SensorEmbedding):

    def __call__(self, data: torch.Tensor, seed: int = 0) -> torch.Tensor:
        N, T, C, H, W = data.shape
        assert (H, W) == (self.H, self.W)
        device = data.device

        data = data.float()
        random.seed(seed)
        torch.manual_seed(seed)
        # Precompute dropout sensor selection for all samples and time frames
        n_dropout_sensors: torch.LongTensor = torch.multinomial(
            torch.tensor(self.dropout_probabilities, device=device),
            num_samples=N * T, replacement=True,
        ).reshape(N, T)
        # Precompute dropout masks across all samples and frames (fast)
        masks: torch.Tensor = torch.ones((N, T, self.S), dtype=torch.bool, device=device)
        for i in range(N):
            for t in range(T):
                n_sensors_to_drop: int = n_dropout_sensors[i, t].item()
                if n_sensors_to_drop > 0:
                    dropout_indices = torch.randperm(self.S, device=device)[:n_sensors_to_drop]
                    masks[i, t, dropout_indices] = False

        del n_dropout_sensors   # manual garbage collection to save memory
        masks = masks.unsqueeze(-1).expand(N, T, self.S, H * W)
        assert masks.shape == (N, T, self.S, H * W)
        
        # Ensure precomputed_distances is on the correct device
        dist = self.precomputed_distances.to(device)
        precomputed_distances_masked: torch.Tensor = dist.unsqueeze(0).unsqueeze(0).expand(N, T, self.S, H * W)
        assert precomputed_distances_masked.shape == (N, T, self.S, H * W)
        
        # Set distances of dropped sensors to infinity
        precomputed_distances_masked = precomputed_distances_masked.masked_fill(mask=~masks, value=float('inf'))
        assert precomputed_distances_masked.shape == (N, T, self.S, H * W)
        nearest_sensor_per_position: torch.Tensor = torch.argmin(precomputed_distances_masked, dim=2)
        del precomputed_distances_masked
        assert nearest_sensor_per_position.shape == (N, T, H * W)
        
        # sensor_positions must match nearest_sensor_per_position device
        sp = self.sensor_positions.to(device)
        assigned_sensor_per_position: torch.Tensor = sp[nearest_sensor_per_position.reshape(-1)].long()
        assert assigned_sensor_per_position.shape == (N * T * H * W, 2)
        assigned_sensor_per_position = assigned_sensor_per_position.reshape(N, T, H, W, 2)
        del nearest_sensor_per_position     # manual garbage collection to save memory
        
        h_indices: torch.LongTensor = assigned_sensor_per_position[..., 0]
        w_indices: torch.LongTensor = assigned_sensor_per_position[..., 1]
        output: torch.Tensor = torch.empty_like(data)
        noisy_data: torch.Tensor = data + torch.randn_like(data) * self.noise_level * data.abs()
        for i in range(N):
            for t in range(T):
                # Now h_indices and w_indices are on the same device as noisy_data
                output[i, t] = noisy_data[i, t, :, h_indices[i, t], w_indices[i, t]]

        assert output.shape == data.shape == (N, T, C, H, W)
        return output

    @cached_property
    def precomputed_distances(self) -> torch.Tensor:
        # Initial computation device
        device = self.sensor_positions.device
        # Create mesh grid for pixel positions
        grid_h, grid_w = torch.meshgrid(
            torch.arange(self.H, device=device), torch.arange(self.W, device=device), 
            indexing='ij',
        )
        grid_positions: torch.Tensor = torch.stack(tensors=[grid_h, grid_w], dim=2).reshape(self.H * self.W, 2).float()
        # Precompute distances for all sensors to all pixels
        differences: torch.Tensor = self.sensor_positions.unsqueeze(1) - grid_positions.unsqueeze(0)
        assert differences.shape == (self.S, self.H * self.W, 2)
        distance: torch.Tensor = (differences ** 2).sum(dim=2).sqrt()
        assert distance.shape == (self.S, self.H * self.W)
        return distance


class SoftVoronoi(SensorEmbedding):
    """
    Soft (distance-weighted) Voronoi embedding.

    Instead of assigning each pixel the value of its single nearest sensor
    (which produces hard, blocky Voronoi boundaries), this class blends the
    values of the ``k`` nearest *active* sensors using inverse-distance
    weighting (IDW):

        w_i  = 1 / (dist_i ^ alpha + eps)
        output[p] = Σ (w_i / Σw_j) * sensor_value[sensor_i]

    Parameters
    ----------
    k : int
        Number of nearest sensors to blend.  ``k=1`` degenerates to the
        original hard Voronoi.  Recommended: 3–6 for 128 sensors on a
        180×360 grid.
    alpha : float
        Distance decay exponent.  Higher values → sharper transition
        (closer to hard Voronoi).  ``alpha=2`` (IDW) is a good default.
        ``alpha→∞`` → hard Voronoi.
    """

    def __init__(
        self,
        resolution: Tuple[int, int],
        sensor_positions: torch.Tensor,
        dropout_probabilities: List[float] = [],
        noise_level: float = 0.,
        k: int = 4,
        alpha: float = 2.0,
    ):
        super().__init__(
            resolution=resolution,
            sensor_positions=sensor_positions,
            dropout_probabilities=dropout_probabilities,
            noise_level=noise_level,
        )
        self.k = k
        self.alpha = alpha

    def __call__(self, data: torch.Tensor, seed: int = 0) -> torch.Tensor:
        N, T, C, H, W = data.shape
        assert (H, W) == (self.H, self.W)
        device = data.device

        data = data.float()
        random.seed(seed)
        torch.manual_seed(seed)

        # ── 1. Dropout mask (same logic as hard Voronoi) ─────────────────────
        n_dropout_sensors: torch.LongTensor = torch.multinomial(
            torch.tensor(self.dropout_probabilities, device=device),
            num_samples=N * T, replacement=True,
        ).reshape(N, T)

        active_masks: torch.Tensor = torch.ones((N, T, self.S), dtype=torch.bool, device=device)
        for i in range(N):
            for t in range(T):
                n_drop: int = n_dropout_sensors[i, t].item()
                if n_drop > 0:
                    drop_idx = torch.randperm(self.S, device=device)[:n_drop]
                    active_masks[i, t, drop_idx] = False
        del n_dropout_sensors

        # ── 2. Noisy sensor readings ──────────────────────────────────────────
        noisy_data: torch.Tensor = data + torch.randn_like(data) * self.noise_level * data.abs()
        # Sensor values: (N, T, C, S) — value at each sensor location
        sp = self.sensor_positions.to(device).long()  # (S, 2)
        # Gather sensor readings for every (i, t): shape (N, T, C, S)
        sensor_vals = noisy_data[:, :, :, sp[:, 0], sp[:, 1]]   # (N, T, C, S)

        # ── 3. Distance-weighted blending ─────────────────────────────────────
        # Precomputed distances: (S, H*W) — move to device
        dist_all = self.precomputed_distances.to(device)  # (S, H*W)

        output = torch.empty(N, T, C, H * W, device=device)

        for i in range(N):
            for t in range(T):
                # Distances with dropped sensors set to inf
                dist = dist_all.clone()                         # (S, H*W)
                dropped = ~active_masks[i, t]                  # (S,) bool
                dist[dropped, :] = float('inf')

                # Effective k: cannot exceed number of active sensors
                n_active = int(active_masks[i, t].sum().item())
                k_eff = min(self.k, n_active)

                # Top-k nearest sensors for every pixel: (k_eff, H*W)
                topk_dist, topk_idx = torch.topk(dist, k=k_eff, dim=0, largest=False)
                # topk_dist: (k_eff, H*W)   topk_idx: (k_eff, H*W)

                # IDW weights: 1 / (dist^alpha + eps)
                eps = 1e-6
                weights = 1.0 / (topk_dist.pow(self.alpha) + eps)  # (k_eff, H*W)
                weights = weights / weights.sum(dim=0, keepdim=True)  # normalise

                # Gather sensor values for top-k indices
                # sensor_vals[i, t]: (C, S) → index by topk_idx (k_eff, H*W)
                sv = sensor_vals[i, t]                          # (C, S)
                # topk_idx flattened for advanced indexing
                sv_topk = sv[:, topk_idx]                       # (C, k_eff, H*W)

                # Weighted sum over k_eff dimension
                # weights: (k_eff, H*W) → broadcast over C: (1, k_eff, H*W)
                output[i, t] = (sv_topk * weights.unsqueeze(0)).sum(dim=1)   # (C, H*W)

        output = output.reshape(N, T, C, H, W)
        assert output.shape == data.shape
        return output

    @cached_property
    def precomputed_distances(self) -> torch.Tensor:
        # Reuse exactly the same distance matrix as hard Voronoi
        device = self.sensor_positions.device
        grid_h, grid_w = torch.meshgrid(
            torch.arange(self.H, device=device), torch.arange(self.W, device=device),
            indexing='ij',
        )
        grid_positions = torch.stack([grid_h, grid_w], dim=2).reshape(self.H * self.W, 2).float()
        differences = self.sensor_positions.unsqueeze(1) - grid_positions.unsqueeze(0)
        assert differences.shape == (self.S, self.H * self.W, 2)
        distance = (differences ** 2).sum(dim=2).sqrt()
        assert distance.shape == (self.S, self.H * self.W)
        return distance


class Mask(SensorEmbedding):


    def __call__(self, data: torch.Tensor, seed: int = 0) -> torch.Tensor:
        N, T, C, H, W = data.shape
        assert (H, W) == (self.H, self.W)
        device = data.device

        data = data.float()
        # Control random seed
        random.seed(seed)
        torch.manual_seed(seed)
        noisy_data: torch.Tensor = data + torch.randn_like(data) * self.noise_level * data.abs()
        output: torch.Tensor = torch.zeros_like(data, dtype=torch.float)  # Must be zeros: non-sensor pixels = 0 (no signal)
        
        sp = self.sensor_positions.to(device).long()
        
        for i in range(N):
            for t in range(T):
                n_dropout_sensors: int = random.choices(
                    population=range(0, self.n_max_dropout_sensors + 1, 1), weights=self.dropout_probabilities, k=1
                )[0]
                dropout_indices: torch.Tensor = torch.randperm(self.S, device=device)[:n_dropout_sensors]
                mask: torch.Tensor = torch.ones(self.S, dtype=torch.bool, device=device)
                mask[dropout_indices] = False
                remaining_sensor_positions: torch.Tensor = sp[mask]
                n_remaining_sensors: int = self.S - n_dropout_sensors
                assert remaining_sensor_positions.shape == (n_remaining_sensors, 2)
                h_indices: torch.Tensor = remaining_sensor_positions[:, 0]
                w_indices: torch.Tensor = remaining_sensor_positions[:, 1]
                output[i, t, :, h_indices, w_indices] = noisy_data[i, t, :, h_indices, w_indices]
        
        assert output.shape == data.shape == (N, T, C, H, W)
        return output


class Vector(SensorEmbedding):

    def __call__(self, data: torch.Tensor, seed: int = 0) -> torch.Tensor:
        N, T, C, H, W = data.shape
        assert (H, W) == (self.H, self.W)
        device = data.device

        data = data.float()
        # Control random seed
        random.seed(seed)
        torch.manual_seed(seed)
        noisy_data: torch.Tensor = data + torch.randn_like(data) * self.noise_level * data.abs()
        output: torch.Tensor = torch.zeros((N, T, C, self.S), dtype=torch.float, device=device)
        
        sp = self.sensor_positions.to(device).long()
        
        for i in range(N):
            for t in range(T):
                n_dropout_sensors: int = random.choices(
                    population=range(0, self.n_max_dropout_sensors + 1, 1), weights=self.dropout_probabilities, k=1
                )[0]
                dropout_indices: torch.Tensor = torch.randperm(self.S, device=device)[:n_dropout_sensors]
                mask: torch.Tensor = torch.ones(self.S, dtype=torch.bool, device=device)
                mask[dropout_indices] = False
                remaining_sensor_positions: torch.Tensor = sp[mask]
                n_remaining_sensors: int = self.S - n_dropout_sensors
                assert remaining_sensor_positions.shape == (n_remaining_sensors, 2)
                h_indices: torch.Tensor = remaining_sensor_positions[:, 0]
                w_indices: torch.Tensor = remaining_sensor_positions[:, 1]
                output[i, t, :, mask] = noisy_data[i, t, :, h_indices, w_indices]
        
        assert output.shape == (N, T, C, self.S)
        return output
