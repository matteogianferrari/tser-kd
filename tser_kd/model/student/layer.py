import torch
import torch.nn as nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Pre-configured 2D 3x3 convolution to use in ResNet architectures.

    Args:
        in_channels: Number of input channels C_in.
        out_channels: Number of kernels C_out.
        stride: Stride to apply over the spatial dimensionality.

    Returns:
        nn.Conv2d: A pre-configured 2D 3x3 convolution.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Pre-configured 2D 1x1 convolution to use in ResNet architectures.

    Args:
        in_channels: Number of input channels C_in.
        out_channels: Number of kernels C_out.
        stride: Stride to apply over the spatial dimensionality.

    Returns:
        nn.Conv2d: A pre-configured 2D 1x1 convolution.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class TDBatchNorm2d(nn.BatchNorm2d):
    """Threshold Dependent Batch Normalization for 2D feature maps.

    Extends nn.BatchNorm2d to compute mean and variance across the temporal dimension (time steps)
    as well as the spatial and channel dimensions.
    Based on the tdBN implementation: https://github.com/thiswinex/STBP-simple.
    Link to related paper: Going Deeper With Directly-Trained Larger Spiking Neural Networks
    (https://arxiv.org/pdf/2011.05280).

    Attributes:
        num_features: Same as nn.BatchNorm2d.
        eps: Same as nn.BatchNorm2d.
        momentum: Same as nn.BatchNorm2d.
        affine: Same as nn.BatchNorm2d.
        track_running_stats: Same as nn.BatchNorm2d.
        alpha: Scaling factor applied before normalization and affine transform.
        V_th: Membrane threshold potential used as scaling factor during normalization.
    """

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            alpha: float = 1.0,
            V_th: float = 1.0
    ) -> None:
        """Initializes the TDBatchNorm2d.

        Args:
            num_features: Same as nn.BatchNorm2d.
            eps: Same as nn.BatchNorm2d.
            momentum: Same as nn.BatchNorm2d.
            affine: Same as nn.BatchNorm2d.
            track_running_stats: Same as nn.BatchNorm2d.
            alpha: Scaling factor applied before normalization and affine transform.
            V_th: Membrane threshold potential used as scaling factor during normalization.
        """
        super(TDBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        # SNNs hyperparameters
        self.alpha = alpha
        self.V_th = V_th

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Threshold Dependent Batch Normalization to the input tensor.

        Computes mean and variance over the temporal (time), batch, spatial height, and width
        dimensions during training, updates running statistics, and uses them during evaluation.

        Args:
            x: Input tensor of shape [T, B, C, H, W].

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        exp_avg_factor = 0.0

        # Computes average factor
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1

                if self.momentum is None:
                    # Cumulative moving average
                    exp_avg_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    # Exponential moving average
                    exp_avg_factor = self.momentum

        # Checks the model's mode
        if self.training:
            # Training
            # Computes the mean over temporal dimension, depth dimension, and spatial dimension
            mean = x.mean([0, 1, 3, 4])

            # Computes teh variance over temporal dimension, depth dimension, and spatial dimension
            var = x.var([0, 1, 3, 4], unbiased=False)

            n = x.numel() / x.size(2)
            with torch.no_grad():
                # Updates running mean
                self.running_mean = exp_avg_factor * mean + (1 - exp_avg_factor) * self.running_mean

                # Updates running variance with unbiased variance
                self.running_var = exp_avg_factor * var * n / (n - 1) + (1 - exp_avg_factor) * self.running_var
        else:
            # Evaluation
            # Uses running mean and variance
            mean = self.running_mean
            var = self.running_var

        # Normalizes the data
        x = self.alpha * self.V_th * \
            (x - mean[None, None, :, None, None]) / \
            (torch.sqrt(var[None, None, :, None, None] + self.eps))

        # Affine transformation
        if self.affine:
            x = x * self.weight[None, None, :, None, None] + self.bias[None, None, :, None, None]

        return x


class TWrapLayer(nn.Module):
    """Layer that wraps common layers to project them into the time domain.

    The input tensor has a time dimension, and requires a forward pass in the time domain for the common layer.

    Attributes:
        layer: Layer to convert.
        batch_norm: Batch normalization layer to apply after the converted layer.
    """

    def __init__(self, layer, batch_norm=None) -> None:
        """Initializes the TWrapLayer.

        Args:
            layer:
            batch_norm:
        """
        super(TWrapLayer, self).__init__()

        # Layers attributes
        self.layer = layer
        self.batch_norm = batch_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: Input tensor of shape [T, B, C, H, W].

        Returns:

        """
        # [T, B, C, H, W]
        # List of temporal outputs
        x_out = []

        # Retrieves the number of time steps
        T = x.size(0)

        for t in range(T):
            # [B, C, H, W]
            x_out.append(self.layer(x[t]))

        x_out = torch.stack(x_out)

        if self.batch_norm is not None:
            x_out = self.batch_norm(x_out)

        return x_out
