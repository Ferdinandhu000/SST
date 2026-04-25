import os
from typing import Callable, Tuple, List
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

def plot_frame(
    sensor_positions: torch.Tensor | None = None,
    sensor_frame: torch.Tensor | None = None,
    fullstate_frame: torch.Tensor | None = None,
    reconstruction_frame: torch.Tensor | None = None,
    mask: torch.Tensor | None = None, 
    reduction: Callable[[torch.Tensor], torch.Tensor] | None = None,
    title: str = '',
    filename: str = '',
    extent = [0, 360, -90, 90], 
    origin: str = 'upper',
    loss_vmin: float = 0.0,
    loss_vmax: float = 10.0,
    dpi: int = 600,
) -> None:
    
    if sensor_positions is not None:
        sensor_positions = sensor_positions.cpu()
    
    frames_to_plot = []
    chart_titles = [] 
    
    mask_np = mask.detach().cpu().numpy().squeeze() if mask is not None else None

    def process_frame(f):
        if f is None: return None
        f = reduction(f) if reduction else f
        f_np = f.detach().cpu().numpy().squeeze()
        if mask_np is not None:
            # Mask the land (mask_np > 0 is land)
            return np.ma.masked_where(mask_np > 0.5, f_np)
        return f_np

    if sensor_frame is not None:
        frames_to_plot.append(process_frame(sensor_frame))
        chart_titles.append("Sensor Value")
    if reconstruction_frame is not None:
        frames_to_plot.append(process_frame(reconstruction_frame))
        chart_titles.append("Reconstruction") 
    if fullstate_frame is not None:
        frames_to_plot.append(process_frame(fullstate_frame))
        chart_titles.append("Full State") 
    if reconstruction_frame is not None and fullstate_frame is not None:
        err = np.abs(
            (reduction(reconstruction_frame) if reduction else reconstruction_frame).detach().cpu().numpy().squeeze() -
            (reduction(fullstate_frame) if reduction else fullstate_frame).detach().cpu().numpy().squeeze()
        )
        if mask_np is not None:
            err = np.ma.masked_where(mask_np > 0.5, err)
        frames_to_plot.append(err)
        chart_titles.append("Error") 

    num_plots = len(frames_to_plot)
    if num_plots == 0: return

    if extent is not None:
        x_span = float(extent[1] - extent[0])
        y_span = float(extent[3] - extent[2])
        aspect_ratio = y_span / x_span if x_span != 0 else (frames_to_plot[0].shape[0] / frames_to_plot[0].shape[1])
    else:
        aspect_ratio = frames_to_plot[0].shape[0] / frames_to_plot[0].shape[1]
    figwidth = 10.0
    fig, axs = plt.subplots(num_plots, 1, figsize=(figwidth, figwidth * aspect_ratio * num_plots)) 
    if num_plots == 1: axs = [axs]

    # Use a colormap with gray for NaN/land
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(color="#d3d3d3")

    for frame, ax, chart_title in zip(frames_to_plot, axs, chart_titles):
        
        if chart_title == 'Error':
            norm = matplotlib.colors.Normalize(vmin=loss_vmin, vmax=loss_vmax)
        else:
            # SST natural range limits
            norm = matplotlib.colors.Normalize(vmin=-2, vmax=35)

        im = ax.imshow(
            frame,
            cmap=cmap,
            norm=norm,
            aspect='auto',
            extent=extent,
            origin=origin,
            interpolation='nearest',
        )
        
        cbar = ax.figure.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=10)

        if sensor_positions is not None and chart_title != 'Error':
            H, W = frame.shape
            if extent is not None:
                x_min, x_max, y_min, y_max = extent
                x_scale = (x_max - x_min) / W
                y_scale = (y_max - y_min) / H
            for sp_row, sp_col in sensor_positions.numpy():
                if extent is not None:
                    plot_x = x_min + (sp_col + 0.5) * x_scale
                    if origin == 'upper':
                        plot_y = y_max - (sp_row + 0.5) * y_scale
                    else:
                        plot_y = y_min + (sp_row + 0.5) * y_scale
                else:
                    plot_x = sp_col
                    plot_y = sp_row
                # Direct plotting on array coordinate pixel space
                ax.add_patch(
                    patches.Circle(
                        xy=(plot_x, plot_y),
                        radius=2.0,
                        edgecolor='black',
                        facecolor='white',
                        fill=True,
                        linewidth=0.5,
                        alpha=0.8,
                        zorder=10
                    )
                )
        
        ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(labelbottom=False, labelleft=False)
    
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    destination_directory = './plots'
    os.makedirs(destination_directory, exist_ok=True)
    if not filename:
        timestamp = dt.datetime.now()
        filename = timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]
    
    save_path = os.path.join(destination_directory, f'{filename}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi//2)
    plt.close(fig)
