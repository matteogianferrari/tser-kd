import torch
import torch.nn as nn
from snntorch import spikegen


class Encoder(nn.Module):
    """Base class Encoder for converting tensors into time-dimensional spikes.

    This abstract class defines the interface for encoders that take a 4D input tensor
    [B, C, H, W] and produce a 5D tensor with time as the first dimension [T, B, C, H, W].

    Attributes:
        num_steps (int): Number of discrete time steps for which the encoder will produce output.
    """

    def __init__(self, num_steps: int) -> None:
        """Initializes the Encoder.

        Args:
            num_steps: The number of time steps to unroll the encoded representation.
        """
        super(Encoder, self).__init__()

        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor 'x' across 'num_steps' time steps.

        Subclasses must implement this method to produce a time‐major output tensor
        of shape [T, B, C, H, W] from a 4D input 'x'.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: A 5D tensor with shape [T, B, C, H, W] representing the encoded input over time.
        """
        raise NotImplementedError


class StaticEncoder(Encoder):
    """Encoder that replicates a static input frame across all time steps without modification.
    """

    def __init__(self, num_steps: int) -> None:
        """Initializes the StaticEncoder.

        Args:
            num_steps: The number of time steps to unroll the encoded representation.
        """
        super(StaticEncoder, self).__init__(num_steps=num_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeats the input tensor 'x' for every time step.

        Simply duplicates the 4D input across the time dimension 'num_steps'. The output
        will have shape [T, B, C, H, W].

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: A 5D tensor with shape [T, B, C, H, W] representing the encoded input over time.
        """
        return x.repeat(self.num_steps, 1, 1, 1, 1)


class RateEncoder(Encoder):
    """Encoder that converts a static input into a Poisson spike train over time based on a rate code.

    Uses 'snntorch.spikegen.rate' to generate spikes at each time step according to the
    values in the input tensor scaled by a gain factor.
    """

    def __init__(self, num_steps: int, gain: float) -> None:
        """Initializes the StaticEncoder.

        Args:
            num_steps: The number of time steps to unroll the encoded representation.
            gain: Scaling factor applied to the input values to set the spike generation rate.
        """
        super(RateEncoder, self).__init__(num_steps=num_steps)

        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates a Poisson spike train from the input tensor 'x' over 'num_steps'.

        The input 'x' represents an intensity map that is scaled by 'gain'.
        At each time step, spikes are sampled according to a Poisson process driven by the scaled intensity.

        Args:
            x: Input tensor of shape [B, C, H, W] containing intensity values in [0, 1] or any non-negative range.

        Returns:
            torch.Tensor: A tensor of shape [T, B, C, H, W] containing binary spike (0 or 1) at each time step.
        """
        return spikegen.rate(data=x, num_steps=self.num_steps, gain=self.gain)
