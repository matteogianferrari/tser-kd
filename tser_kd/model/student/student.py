import torch
import torch.nn as nn
import snntorch as snn
from tser_kd.model import conv3x3, conv1x1
from tser_kd.model.student import LayerTWrapper, LIFTWrapper


class SCNN(nn.Module):
    """Spiking CNN architecture.

    This spiking CNN is a test architecture that is simple without residual connections and batch normalization.


    """

    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            beta: float,
            threshold: float,
            learn_beta: bool = False,
            learn_threshold: bool = False
    ) -> None:
        """Initializes the SCNN.

        Args:
            in_channels: Number of channels in the input frames.
            num_classes: Number of output classes.
            beta: Membrane-decay constant for all LIF neurons.
            threshold: Membrane voltage threshold for all LIF neurons.
            learn_beta: Flag that allows to learn the membrane decay for all LIF neurons.
            learn_threshold: Flag that allows to learn the membrane voltage threshold for all LIF neurons.
        """
        super(SCNN, self).__init__()

        # Stem block
        self.stem = LayerTWrapper(layer=conv3x3(in_channels=in_channels, out_channels=64))
        self.lif_stem = LIFTWrapper(
            layer=snn.Leaky(beta=beta, threshold=threshold, init_hidden=True, learn_beta=learn_beta, learn_threshold=learn_threshold)
        )

        # Conv1
        self.conv1 = LayerTWrapper(layer=conv3x3(in_channels=64, out_channels=64))
        self.lif1 = LIFTWrapper(
            layer=snn.Leaky(beta=beta, threshold=threshold, init_hidden=True, learn_beta=learn_beta, learn_threshold=learn_threshold)
        )

        # Conv2
        self.conv2 = LayerTWrapper(layer=conv3x3(in_channels=64, out_channels=128, stride=2))
        self.lif2 = LIFTWrapper(
            layer=snn.Leaky(beta=beta, threshold=threshold, init_hidden=True, learn_beta=learn_beta, learn_threshold=learn_threshold)
        )

        # Conv3
        self.conv3 = LayerTWrapper(layer=conv3x3(in_channels=128, out_channels=128))
        self.lif3 = LIFTWrapper(
            layer=snn.Leaky(beta=beta, threshold=threshold, init_hidden=True, learn_beta=learn_beta, learn_threshold=learn_threshold)
        )

        # Conv4
        self.conv4 = LayerTWrapper(layer=conv3x3(in_channels=128, out_channels=256, stride=2))
        self.lif4 = LIFTWrapper(
            layer=snn.Leaky(beta=beta, threshold=threshold, init_hidden=True, learn_beta=learn_beta, learn_threshold=learn_threshold)
        )

        # Global pool
        self.avg_pool = LayerTWrapper(layer=nn.AdaptiveAvgPool2d((1, 1)))

        self.mlp = LayerTWrapper(layer=nn.Linear(in_features=256, out_features=num_classes, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [T, B, C, H, W].

        Returns:
            torch.Tensor: Logits for every time step if the model is in training mode, shape [T, B, K],
                mean of logits over time steps if the model is in eval mode, shape [B, K].
        """
        # x.shape: [T, B, 1, 28, 28]

        x = self.lif_stem(self.stem(x))
        # x.shape: [T, B, 64, 28, 28]

        x = self.lif1(self.conv1(x))
        # x.shape: [T, B, 64, 28, 28]

        x = self.lif2(self.conv2(x))
        # x.shape: [T, B, 128, 14, 14]

        x = self.lif3(self.conv3(x))
        # x.shape: [T, B, 128, 14, 14]

        x = self.lif4(self.conv4(x))
        # x.shape: [T, B, 256, 7, 7]

        # Global pool
        x = self.avg_pool(x)
        x = x.view(x.size(0), x.size(1), x.size(2))

        # MLP
        x = self.mlp(x)

        # Regulates the output based on the model current mode
        return x if self.training else x.mean(0)


def make_student_model(
        arch: str,
        in_channels: int,
        num_classes: int,
        beta: float,
        threshold: float,
        device: torch.device,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        state_dict: dict = None
) -> nn.Module:
    """Constructs and returns a Spiking student model based on the specified architecture.

    The model is then moved to the given device.
    Optionally loads a pre-trained state dictionary into the model.

    Currently, supports:
        - 'scnn': Custom test spiking CNN.

    Args:
        arch: Architecture name.
        in_channels: Number of input channels for the first convolutional layer.
        num_classes: Number of output classes for the final linear layer.
        beta: Initial membrane-decay constant for all LIF neurons.
        threshold: Membrane voltage threshold for all LIF neurons.
        device: The device (CPU or GPU) to which the model will be offloaded.
        learn_beta: Flag that allows to learn the membrane decay for all LIF neurons.
        learn_threshold: Flag that allows to learn the membrane voltage threshold for all LIF neurons.
        state_dict: A state dictionary of pre-trained weights to load into the model.

    Returns:
        nn.Module: The constructed student model on the specified device.
    """
    student = None

    # Custom architecture
    if arch == 'scnn':
        # Creates the student model architecture
        student = SCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            beta=beta,
            threshold=threshold,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold
        )

    # Checks if a parameter configuration must be loaded
    if state_dict is not None:
        # Loads the best parameters for the architecture
        student.load_state_dict(state_dict)

    # Offloads the student model to the specified device
    student = student.to(device)

    return student
