import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Literal
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from common.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from cfd.dataset import CFDDataset, DatasetMixin
from cfd.embedding import Voronoi, Mask, Vector
from model import FLRONetFNO, FLRONetUNet, FLRONetMLP, FNO3D, FLRONetTransolver, FNO, AFNO, Transolver
from common.plotting import plot_frame
from common.functional import compute_velocity_field


class Worker:

    def __init__(self):
        raise ValueError('Base Worker class is not meant to be instantiated')

    def _validate_inputs(
        self,
        sensor_timeframes: torch.Tensor, sensor_frames: torch.Tensor,
        fullstate_timeframes: torch.Tensor, fullstate_frames: torch.Tensor, 
    ) -> Tuple[int, int, int, int, int, int]:
        assert fullstate_frames.ndim == 5
        assert sensor_timeframes.ndim == 2 and fullstate_timeframes.ndim == 2
        n_sensor_frames, n_fullstate_frames = sensor_timeframes.shape[1], fullstate_timeframes.shape[1]
        assert sensor_frames.shape[0] == fullstate_frames.shape[0]
        batch_size: int = sensor_frames.shape[0]

        if isinstance(self.net, FLRONetMLP):
            assert sensor_frames.ndim == 4 
            n_channels, S = sensor_frames.shape[-2:]
            H, W = fullstate_frames.shape[-2:]
            assert sensor_frames.shape == (batch_size, n_sensor_frames, n_channels, S)
        else:
            assert sensor_frames.ndim == 5
            n_channels, H, W = sensor_frames.shape[-3:]
            assert sensor_frames.shape == (batch_size, n_sensor_frames, n_channels, H, W)

        assert fullstate_frames.shape == (batch_size, n_fullstate_frames, n_channels, H, W)

    def _validate_embedding_generator(self, embedding_generator: Voronoi | Mask | Vector) -> None:
        if isinstance(self.net, FLRONetMLP):
            assert isinstance(embedding_generator, Vector)
        if isinstance(self.net, (FLRONetUNet, FLRONetFNO, FNO3D, FLRONetTransolver, FNO, AFNO, Transolver)):
            assert isinstance(embedding_generator, (Voronoi, Mask))


class Trainer(Worker):

    def __init__(
        self, 
        net: FLRONetFNO | FLRONetUNet | FLRONetMLP | FNO3D | FLRONetTransolver | FNO | AFNO | Transolver,
        lr: float,
        train_dataset: CFDDataset,
        val_dataset: CFDDataset,
        train_batch_size: int,
        val_batch_size: int,
    ):
        self.net: FLRONetFNO | FLRONetUNet | FLRONetMLP | FNO3D | FLRONetTransolver | FNO | AFNO | Transolver = net
        self.lr: float = lr
        self.train_dataset: CFDDataset = train_dataset
        self.val_dataset: CFDDataset = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size

        self.mask = train_dataset.mask
        if self.mask is not None:
            self.mask = self.mask.cuda()

        self._validate_embedding_generator(embedding_generator=train_dataset.embedding_generator)
        self._validate_embedding_generator(embedding_generator=val_dataset.embedding_generator)
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
        self.loss_function: nn.Module = nn.MSELoss(reduction='none')

        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net).cuda()
        elif torch.cuda.device_count() == 1:
            self.net = self.net.cuda()
        else:
            raise ValueError('No GPUs are found in the system')
        
        self.optimizer = Adam(params=self.net.parameters(), lr=lr)

    def compute_masked_mse(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int]:
        mse = (input - target) ** 2
        if self.mask is not None:
            mask_expanded = self.mask.view(1, 1, 1, *self.mask.shape)
            mse = mse * (1 - mask_expanded)
            n_elems = (1 - mask_expanded).sum() * input.shape[0] * input.shape[1] * input.shape[2]
        else:
            n_elems = input.numel()
        return mse.sum(), int(n_elems)

    def train(
        self, 
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_path: Optional[str] = None,
        save_frequency: int = 5,
    ) -> None:
        
        train_metrics = Accumulator()
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        checkpoint_saver = CheckpointSaver(model=self.net, dirpath=checkpoint_path)
        
        if isinstance(self.net, (FLRONetFNO, FLRONetUNet, FLRONetMLP, FLRONetTransolver, FNO, AFNO, Transolver)):
            self.model_name = self.net.__class__.__name__.lower()
        else:
            self.model_name = 'fno3d'
            
        self.net.train()
        
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            for batch, (
                sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames, _, _
            ) in enumerate(tqdm(self.train_dataloader, desc=f'Epoch {epoch}/{n_epochs}: '), start=1):
                timer.start_batch(epoch, batch)
                
                device = next(self.net.parameters()).device
                sensor_timeframes = sensor_timeframes.to(device); sensor_frames = sensor_frames.to(device)
                fullstate_timeframes = fullstate_timeframes.to(device); fullstate_frames = fullstate_frames.to(device)

                self._validate_inputs(sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames)
                self.optimizer.zero_grad()
                
                if isinstance(self.net, (FLRONetFNO, FLRONetMLP, FLRONetUNet, FLRONetTransolver, FNO, AFNO, Transolver)):
                    reconstruction_frames: torch.Tensor = self.net(
                        sensor_timeframes=sensor_timeframes,
                        sensor_values=sensor_frames,
                        fullstate_timeframes=fullstate_timeframes,
                        out_resolution=None,
                    )
                else:
                    reconstruction_frames: torch.Tensor = self.net(sensor_values=sensor_frames, out_resolution=None)

                total_mse, n_elems = self.compute_masked_mse(reconstruction_frames, fullstate_frames)
                mean_mse: torch.Tensor = total_mse / n_elems
                mean_mse.backward()
                self.optimizer.step()
                
                train_metrics.add(total_mse=total_mse.item(), n_elems=n_elems)
                timer.end_batch(epoch=epoch)
                train_mean_mse: float = train_metrics['total_mse'] / train_metrics['n_elems']
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), took=timer.time_batch(epoch, batch), 
                    mean_train_rmse=train_mean_mse ** 0.5, mean_train_mse=train_mean_mse,
                )

            if checkpoint_path is not None and epoch % save_frequency == 0:
                checkpoint_saver.save(model_states=self.net.state_dict(), filename=f'{self.model_name}{epoch}.pt')

            train_metrics.reset()
            mean_val_mse: float = self.evaluate()
            mean_val_rmse: float = mean_val_mse ** 0.5
            timer.end_epoch(epoch)
            logger.log(
                epoch=epoch, n_epochs=n_epochs, took=timer.time_epoch(epoch), 
                mean_val_rmse=mean_val_rmse, mean_val_mse=mean_val_mse,
            )
            print('=' * 20)

            early_stopping(value=mean_val_rmse)
            if early_stopping:
                print('Early Stopped')
                break

        if checkpoint_path:
            checkpoint_saver.save(model_states=self.net.state_dict(), filename=f'{self.model_name}{epoch}.pt')

    def evaluate(self) -> float:
        val_metrics = Accumulator()
        self.net.eval()
        device = next(self.net.parameters()).device
        with torch.no_grad():
            for sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames, _, _ in self.val_dataloader:
                sensor_timeframes = sensor_timeframes.to(device); sensor_frames = sensor_frames.to(device)
                fullstate_timeframes = fullstate_timeframes.to(device); fullstate_frames = fullstate_frames.to(device)

                self._validate_inputs(sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames)
                if isinstance(self.net, (FLRONetFNO, FLRONetMLP, FLRONetUNet, FLRONetTransolver, FNO, AFNO, Transolver)):
                    reconstruction_frames: torch.Tensor = self.net(
                        sensor_timeframes=sensor_timeframes,
                        sensor_values=sensor_frames,
                        fullstate_timeframes=fullstate_timeframes,
                        out_resolution=None,
                    )
                else:
                    reconstruction_frames: torch.Tensor = self.net(sensor_values=sensor_frames, out_resolution=None)

                total_mse, n_elems = self.compute_masked_mse(reconstruction_frames, fullstate_frames)
                val_metrics.add(total_mse=total_mse.item(), n_elems=n_elems)

        return val_metrics['total_mse'] / val_metrics['n_elems']


class Predictor(Worker, DatasetMixin):

    def __init__(self, net: FLRONetFNO | FLRONetUNet | FLRONetMLP | FNO3D | FLRONetTransolver | FNO | AFNO | Transolver):
        self.net: FLRONetFNO | FLRONetUNet | FLRONetMLP | FNO3D | FLRONetTransolver | FNO | AFNO | Transolver = net.cuda()
        self.rmse = nn.MSELoss(reduction='sum')
        self.mae = nn.L1Loss(reduction='sum')
        
        if isinstance(self.net, (FLRONetFNO, FLRONetUNet, FLRONetMLP, FLRONetTransolver, FNO, AFNO, Transolver)):
            self.model_name = self.net.__class__.__name__.lower()
        else:
            self.model_name = 'fno3d'

    def predict_from_scratch(
        self, 
        case_dir: str, 
        sensor_timeframes: List[int],
        reconstruction_timeframes: List[float],
        sensor_position_path: str, 
        embedding_generator: Literal['Voronoi', 'Mask', 'Vector'],
        n_dropout_sensors: int,
        noise_level: float,
        in_resolution: Tuple[int, int],
        out_resolution: Tuple[int, int] | None = None,
    ):
        device = next(self.net.parameters()).device
        data: torch.Tensor = self.load2tensor(case_dir).to(device)
        n_channels = data.shape[1]
        
        mask_path = os.path.join('tensors', 'test', 'metadata', 'mask.pt')
        mask = torch.load(mask_path, weights_only=True).to(device) if os.path.exists(mask_path) else None

        reconstruction_timeframes_tensor: torch.Tensor = torch.tensor(reconstruction_timeframes, dtype=torch.float, device=device).unsqueeze(0)
        sensor_timeframes_tensor: torch.Tensor = torch.tensor(sensor_timeframes, dtype=torch.int, device=device).unsqueeze(0)
        sensor_frames: torch.Tensor = data[sensor_timeframes_tensor]
        original_sensor_positions: torch.Tensor = torch.load(sensor_position_path, weights_only=True, map_location=device).int()
        
        if n_dropout_sensors == 0:
            implied_dropout_probabilities: List[float] = []
        else:
            implied_dropout_probabilities: List[float] = [0.] * n_dropout_sensors
            implied_dropout_probabilities[-1] = 1.

        if embedding_generator == 'Mask':
            embedding_gen = Mask(resolution=in_resolution, sensor_positions=original_sensor_positions, dropout_probabilities=implied_dropout_probabilities, noise_level=noise_level)
        elif embedding_generator == 'Voronoi':
            embedding_gen = Voronoi(resolution=in_resolution, sensor_positions=original_sensor_positions, dropout_probabilities=implied_dropout_probabilities, noise_level=noise_level)
        else:
            embedding_gen = Vector(resolution=in_resolution, sensor_positions=original_sensor_positions, dropout_probabilities=implied_dropout_probabilities, noise_level=noise_level)
        
        sensor_frames = embedding_gen(data=sensor_frames)
        
        self.net.eval()
        with torch.no_grad():
            if isinstance(self.net, (FLRONetFNO, FLRONetMLP, FLRONetUNet, FLRONetTransolver, FNO, AFNO, Transolver)):
                reconstruction_frames: torch.Tensor = self.net(
                    sensor_timeframes=sensor_timeframes_tensor,
                    sensor_values=sensor_frames,
                    fullstate_timeframes=reconstruction_timeframes_tensor,
                    out_resolution=out_resolution,
                )
            else:
                reconstruction_frames: torch.Tensor = self.net(sensor_values=sensor_frames, out_resolution=(max(sensor_timeframes)-min(sensor_timeframes), *(out_resolution or in_resolution)))

        reconstruction_frames = reconstruction_frames.squeeze(dim=0)
        reconstruction_timeframes_tensor = reconstruction_timeframes_tensor.squeeze(dim=0)
        case_name = os.path.basename(case_dir)
        reduction_fn = lambda x: x[0]
        
        sensor_timeframes_list = sensor_timeframes_tensor.squeeze(0).tolist()

        for frame_idx in tqdm(range(reconstruction_frames.shape[0]), desc=f'{case_name}: '):
            t = float(reconstruction_timeframes_tensor[frame_idx].item())
            gt_frame = data[int(t)] if int(t) < data.shape[0] else None
            
            # Condition: only plot sensor_frame if t matches a sensor_timeframe
            if t in sensor_timeframes_list:
                idx_in_sensor = sensor_timeframes_list.index(int(t))
                s_frame = sensor_frames[0, idx_in_sensor]
            else:
                s_frame = None

            plot_frame(
                sensor_positions=original_sensor_positions,
                sensor_frame=s_frame,
                reconstruction_frame=reconstruction_frames[frame_idx],
                fullstate_frame=gt_frame,
                mask=mask,
                reduction=reduction_fn,
                title=f'{case_name.upper()} week={int(t)}. active sensors: {original_sensor_positions.shape[0] - n_dropout_sensors}/{original_sensor_positions.shape[0]}',
                filename=f'{self.model_name}_{case_name.lower()}_w{int(t)}_d{n_dropout_sensors}_n{int(noise_level*100)}_{in_resolution[0]}x{in_resolution[1]}',
            )

    def predict_from_dataset(self, dataset: CFDDataset) -> Tuple[float, float, float]:
        self.net.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        device = next(self.net.parameters()).device
        mask = dataset.mask.to(device) if dataset.mask is not None else None
        
        trained_H, trained_W = dataset.resolution
        n_sensors = dataset.n_sensors
        n_dropout_sensors = len(dataset.dropout_probabilities) - 1 if len(dataset.dropout_probabilities) > 0 else 0
        n_active_sensors = n_sensors - n_dropout_sensors
        
        rmse_values, mae_values, L2Loss_values = [], [], []
        
        with torch.no_grad():
            for st, sv, ft, fv, case_names, sampling_ids in tqdm(dataloader):
                st, sv, ft, fv = st.to(device), sv.to(device), ft.to(device), fv.to(device)
                
                if isinstance(self.net, (FLRONetFNO, FLRONetMLP, FLRONetUNet, FLRONetTransolver, FNO, AFNO, Transolver)):
                    preds = self.net(st, sv, ft, None)
                else:
                    preds = self.net(sv, None)
                
                sensor_frames = sv.squeeze(dim=0)
                sensor_timeframes = st.squeeze(dim=0).tolist()
                reconstruction_frames = preds.squeeze(dim=0)
                fullstate_frames = fv.squeeze(dim=0)
                fullstate_timeframes = ft.squeeze(dim=0)

                for frame_idx, timeframe in enumerate(fullstate_timeframes):
                    time_val = int(timeframe.item())
                    # Logic: Only draw sensor frame if timeframe is in sensor_timeframes
                    if isinstance(self.net, (FLRONetMLP, FNO3D)) or time_val not in sensor_timeframes:
                        sensor_frame = None
                    else:
                        idx_in_sensor = sensor_timeframes.index(time_val)
                        sensor_frame = sensor_frames[idx_in_sensor]
                    
                    reconstruction_frame = reconstruction_frames[frame_idx]
                    fullstate_frame = fullstate_frames[frame_idx]
                    
                    if mask is not None:
                        m = mask.view(1, *mask.shape)
                        diff_sq = ((reconstruction_frame - fullstate_frame) ** 2) * (1 - m)
                        diff_abs = (reconstruction_frame - fullstate_frame).abs() * (1 - m)
                        n_valid = (1 - m).sum() * reconstruction_frame.shape[0]
                        frame_mean_mse = diff_sq.sum().item() / n_valid.item()
                        frame_mean_rmse = frame_mean_mse ** 0.5
                        frame_mean_mae = diff_abs.sum().item() / n_valid.item()
                        L2_num = diff_sq.sum().item() ** 0.5
                        L2_den = (fullstate_frame**2 * (1-m)).sum().item() ** 0.5
                        frame_mean_L2 = L2_num / L2_den
                    else:
                        frame_total_mse = self.rmse(reconstruction_frame.unsqueeze(0), fullstate_frame.unsqueeze(0))
                        frame_mean_rmse = (frame_total_mse.item() / fullstate_frame.numel()) ** 0.5
                        frame_total_mae = self.mae(reconstruction_frame.unsqueeze(0), fullstate_frame.unsqueeze(0))
                        frame_mean_mae = frame_total_mae.item() / fullstate_frame.numel()
                        frame_mean_L2 = frame_mean_rmse / (torch.norm(fullstate_frame).item() / (fullstate_frame.numel()**0.5))

                    at_timeframe = time_val
                    plot_frame(
                        sensor_positions=dataset.sensor_positions,
                        sensor_frame=sensor_frame,
                        fullstate_frame=fullstate_frame, 
                        reconstruction_frame=reconstruction_frame,
                        mask=mask,
                        reduction=lambda x: x[0],
                        title=(
                            f'week={at_timeframe}, '
                            f'active sensors: {str(n_active_sensors).zfill(2)}/{str(n_sensors).zfill(2)}, '
                            f'RMSE: {frame_mean_rmse:.3f}, MAE: {frame_mean_mae:.3f}, L2Loss: {frame_mean_L2:.3f}'
                        ),
                        filename=f'{self.model_name}_{case_names[0].lower()}s{sampling_ids[0]}_w{str(at_timeframe).zfill(4)}_d{n_dropout_sensors}_n{int(dataset.noise_level*100)}_{trained_H}x{trained_W}'
                    )
                    rmse_values.append(frame_mean_rmse)
                    mae_values.append(frame_mean_mae)
                    L2Loss_values.append(frame_mean_L2)

        avg_rmse = sum(rmse_values) / len(rmse_values)
        avg_mae = sum(mae_values) / len(mae_values)
        avg_l2 = sum(L2Loss_values) / len(L2Loss_values)
        return avg_rmse, avg_mae, avg_l2
