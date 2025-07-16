import torch
import torch.nn as nn
import snntorch as snn


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
            torch.Tensor: Normalized tensor with the shape [T, B, C, H, W].
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
            # mean.shape: [C]
            mean = x.mean([0, 1, 3, 4])

            # Computes the variance over temporal dimension, depth dimension, and spatial dimension
            # var.shape: [C]
            var = x.var([0, 1, 3, 4], unbiased=False)

            # Computes the number of elements in the tensor
            tot_elem = x.numel()
            C = x.size(2)
            N = tot_elem / C
            with torch.no_grad():
                # Updates running mean
                self.running_mean = exp_avg_factor * mean + (1 - exp_avg_factor) * self.running_mean

                # Updates running variance with unbiased variance
                self.running_var = exp_avg_factor * var * N / (N - 1) + (1 - exp_avg_factor) * self.running_var
        else:
            # Evaluation
            # Uses running mean and variance
            mean = self.running_mean
            var = self.running_var

        # Normalizes the data, x.shape: [T, B, C, H, W]
        x = self.alpha * self.V_th * \
            (x - mean[None, None, :, None, None]) / \
            (torch.sqrt(var[None, None, :, None, None] + self.eps))

        # Affine transformation
        if self.affine:
            # x.shape: [T, B, C, H, W]
            x = x * self.weight[None, None, :, None, None] + self.bias[None, None, :, None, None]

        return x


class LayerTWrapper(nn.Module):
    """Wraps a spatial layer to apply it independently along the temporal dimension.

    Attributes:
        layer: The spatial layer to execute at each time step.
        batch_norm: Batch normalization layer that expects input of shape [T, B, C, H, W].
    """

    def __init__(self, layer: nn.Module, batch_norm: nn.Module = None) -> None:
        """Initializes the LayerTWrapper.

        Args:
            layer: A PyTorch layer that processes a single time slice of shape [B, C, H, W] or [B, N].
            batch_norm: A batch normalization layer that can accept a tensor of shape [T, B, C, H, W] or
                [T, B, N] and perform normalization across time and spatial dimensions.
        """
        super(LayerTWrapper, self).__init__()

        # Layers attributes
        self.layer = layer
        self.batch_norm = batch_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the wrapped layer across the time dimension and then optional batch normalization.

        This approach vectorizes the time dimension and allow the layer to process all the time steps in one pass.

        Args:
            x: Input tensor of shape [T, B, C, H, W] or [T, B, N].

        Returns:
            torch.Tensor: Output tensor of shape [T, B, C, H, W] or [T, B, N] normalized along
                the temporal and spatial dimensions.
        """
        # Retrieves the time steps and batch-size of the input tensor
        # 'in_spatial' could be [C, H, W] or [N]
        T, B, *in_spatial = x.shape

        # Collapses the time and batch dimensions
        x = x.reshape(T * B, *in_spatial)

        # One forward pass for the layer
        x = self.layer(x)

        # Retrieves the new spatial dimension
        _, *out_spatial = x.shape
        x = x.reshape(T, B, *out_spatial)

        # Applies batch normalization if present
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        return x


class LIFTWrapper(nn.Module):
    """Layer that wraps snnTorch Leaky layer to allow it to integrate over input spikes.

    Attributes:
        layer: An snnTorch Leaky neuron layer that performs membrane potential integration
            and spike generation per time step.
    """

    def __init__(self, layer: snn.Leaky) -> None:
        """Initializes the LIFTWrapper.

        Args:
            layer: A preconfigured snnTorch Leaky neuron layer.
        """
        super(LIFTWrapper, self).__init__()

        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the Leaky layer over the time dimension of the input spike train.

        Iterates through T time steps, feeding each slice x[t] of shape [B, C, H, W] into the Leaky
        layer. Collects the output spikes at each step, stacks them into a tensor of shape
        [T, B, C, H, W], and then resets the Leaky layer's hidden and membrane states.

        Args:
            x: Input spike train tensor of shape [T, B, C, H, W].

        Returns:
            torch.Tensor: Output spike tensor of shape [T, B, C, H, W], containing the spikes
                generated by the Leaky layer at each time step.
        """
        # List of outputs spikes
        x_out = []

        # Retrieves the number of time steps, x.shape: [T, B, C, H, W]
        T = x.size(0)

        # Forward pass in time
        for t in range(T):
            # Retrieves the sample at time step t, x_t.shape: [B, C, H, W]
            x_t = x[t]

            # Appends the layer output
            x_out.append(self.layer(x_t))

        # Converts the list into a tensor
        x_out = torch.stack(x_out)

        # Resets the membrane potential and the hidden state to avoid backprop errors
        self.layer.reset_hidden()
        self.layer.reset_mem()

        return x_out
