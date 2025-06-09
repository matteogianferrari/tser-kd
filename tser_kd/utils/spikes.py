from typing import List
from matplotlib import pyplot as plt
from IPython.display import HTML, display
import torch
from snntorch import spikeplot as splt


def plot_spike_train_over_channels(spike_train: torch.Tensor, cmaps: str | List[str]) -> None:
    """Visualizes spike train over each channel as animated plots.

    This function creates one animation per channel. Each animation shows the spiking activity over T time steps
    for that channel, using the specified colormap(s).

    Args:
        spike_train: A tensor of shape [T, C, H, W] containing binary or continuous spike values over T time steps,
            for C channels, each of spatial size H×W.
        cmaps: A matplotlib colormap name applied to all channels, or a sequence of length C specifying
            a colormap for each channel.
    """
    # Input tensor of shape [T, C, H, W]
    # Retrieves the number of channels
    C = spike_train.size(1)

    # Checks if the cmap must be equal for all channels
    if isinstance(cmaps, str):
        # Repeats the cmap for all channels
        cmaps = [cmaps] * C

    # Plots each channel
    for c, cmap in zip(range(C), cmaps):
        # Subplots
        fig, ax = plt.subplots()
        ax.set_title(f"Channel {c}")

        # Animates and displays the plot
        anim = splt.animator(data=spike_train[:, c], fig=fig, ax=ax, cmap=cmap)
        display(HTML(anim.to_html5_video()))

        plt.close(fig)


def plot_raster_over_channels(spike_train: torch.Tensor) -> None:
    """Plots a raster of spikes for each channel in a spatio-temporal spike train tensor.

    For each channel produces a raster plot showing either time vs. features or features vs. time,
    depending on which dimension is larger.

    Args:
        spike_train: Tensor of shape [T, C, H, W] containing spike indicators.

    """
    # Input tensor of shape [T, C, H, W]
    T, C, H, W = spike_train.shape

    # Creates a single figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="w")

    for ch in range(C):
        # Select the subplot
        ax = axes[ch]

        # Selects the channel and collapses the spatial dimension into a shape [T, N]
        data = spike_train[:, ch].reshape(T, -1)
        x_label = "Time steps"
        y_label = "Features"

        # Checks if the time steps are greater than the number of features
        if T < data.size(1):
            # Reshape the tensor to [N, T]
            data = data.T
            x_label = "Features"
            y_label = "Time steps"

        # Raster plot
        splt.raster(data, ax, s=1.5, c='black')
        ax.set_title(f"Channel {ch}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    plt.tight_layout()
    plt.show()
