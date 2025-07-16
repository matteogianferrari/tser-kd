import torch
import torch.nn as nn
import snntorch as snn
from tser_kd.model import conv3x3, conv1x1
from tser_kd.model.student import TDBatchNorm2d, LayerTWrapper, LIFTWrapper


class SResNetBlock(nn.Module):
    """SResNet basic block.

    Spiking version of the classic ResNet basic block.

    The membrane‐decay parameter is learnable in every LIF instance, and thresholds are also
    learned during training.

    Attributes:
        t_conv_bn1: First 3x3conv followed by a temporal batch normalization.
        lif1: LIF neuron after the first convolution.
        t_conv_bn2: Second 3x3conv followed by a temporal batch normalization.
        lif2: LIF neuron after second convolution.
        shortcuts: Identity mapping if channels don't change, 1x1 projection otherwise.
        lif3: Final LIF neuron applied after summing the residual and shortcut paths.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, beta: float) -> None:
        """Initializes the SResNetBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride to apply in the first 3x3conv and in the 1x1 projection if needed.
            beta: Membrane decay parameter.
        """
        super(SResNetBlock, self).__init__()

        # Main branch
        self.t_conv_bn1 = LayerTWrapper(
            layer=conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride),
            batch_norm=TDBatchNorm2d(num_features=out_channels)
        )
        self.lif1 = LIFTWrapper(layer=snn.Leaky(beta=beta, init_hidden=True))

        self.t_conv_bn2 = LayerTWrapper(
            layer=conv3x3(in_channels=out_channels, out_channels=out_channels),
            batch_norm=TDBatchNorm2d(num_features=out_channels, alpha=1/(2**0.5))
        )
        self.lif2 = LIFTWrapper(layer=snn.Leaky(beta=beta, init_hidden=True))

        # Shortcut branch
        self.shortcuts = None
        if stride != 1 or in_channels != out_channels:
            self.shortcuts = LayerTWrapper(
                layer=conv1x1(in_channels=in_channels, out_channels=out_channels, stride=stride),
                batch_norm=TDBatchNorm2d(num_features=out_channels, alpha=1/(2**0.5))
            )

        self.lif3 = LIFTWrapper(layer=snn.Leaky(beta=beta, init_hidden=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the basic block.

        Args:
            x: Input tensor of shape [T, B, C, H, W].

        Returns:
            torch.Tensor: The output of the basic block of shape [T, B, C, H, W].
        """
        identity = x

        # Main branch
        x = self.t_conv_bn1(x)
        x = self.lif1(x)

        x = self.t_conv_bn2(x)
        x = self.lif2(x)

        # Shortcut branch
        if self.shortcuts is not None:
            identity = self.shortcuts(identity)

        # Add and final LIF
        x += identity

        return self.lif3(x)


class SResNet19(nn.Module):
    """SResNet-19 classifier.

    Spiking version of the 19-layer ResNet architecture.
    The membrane-decay parameter is learnable in every LIF instance, and thresholds are also learned during training.

    Attributes:
        stem: 3×3 convolution followed by temporal batch normalization.
        lif1: LIF neuron after the stem layer.
        block1: First residual stage with three SResNet blocks (128 channels, stride 1).
        block2: Second stage with three blocks (256 channels, first block stride 2).
        block3: Third stage with two blocks (512 channels, first block stride 2).
        t_avg_pool: Global average-pooling layer applied over time.
        t_fc1: First fully-connected layer.
        lif2: LIF neuron between the two linear layers.
        t_fc2: Classification head that maps the features to *num_classes* logits per time step.
    """

    def __init__(self, in_channels: int, num_classes: int, beta: float) -> None:
        """Initializes the SResNet19.

        Args:
            in_channels: Number of channels in the input frames.
            num_classes: Number of output classes.
            beta: Initial membrane-decay constant for all LIF neurons.
        """
        super(SResNet19, self).__init__()

        self.start_channels = 128

        # Stem block
        self.stem = LayerTWrapper(
            layer=conv3x3(in_channels=in_channels, out_channels=128),
            batch_norm=TDBatchNorm2d(num_features=128)
        )

        self.lif1 = LIFTWrapper(layer=snn.Leaky(beta=beta, init_hidden=True))

        # Block 1
        self.block1 = self._make_block(num_blocks=3, out_channels=128, beta=beta)

        # Block 2
        self.block2 = self._make_block(num_blocks=3, out_channels=256, beta=beta)

        # Block 3
        self.block3 = self._make_block(num_blocks=2, out_channels=512, beta=beta)

        # Global average pooling
        self.t_avg_pool = LayerTWrapper(layer=nn.AdaptiveAvgPool2d((1, 1)))

        # MLP
        self.t_fc1 = LayerTWrapper(layer=nn.Linear(in_features=512, out_features=256, bias=False))

        self.lif2 = LIFTWrapper(layer=snn.Leaky(beta=beta, init_hidden=True))

        self.t_fc2 = LayerTWrapper(layer=nn.Linear(in_features=256, out_features=num_classes, bias=False))

    def _make_block(self, num_blocks: int, out_channels: int, beta: float) -> nn.Sequential:
        """Builds a SResNetBlock with the specific configuration.

        Args:
            num_blocks: How many SResNet blocks to stack.
            out_channels: Number of output channels for the stage.
            beta: Initial membrane-decay constant for the stage.

        Returns:
            A sequential container with the requested blocks.
        """
        layers = []

        for _ in range(num_blocks):
            # Check for the stride
            if self.start_channels != out_channels:
                stride = 2
            else:
                stride = 1

            # Adds a block
            layers.append(
                SResNetBlock(in_channels=self.start_channels, out_channels=out_channels, stride=stride, beta=beta)
            )

            # Updates the current number of channels
            self.start_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [T, B, C, H, W].

        Returns:
            torch.Tensor: Logits for every time step, shape [T, B, K].
        """
        # Stem layer, x.shape: [T, B, 128, 32, 32]
        x = self.stem(x)

        # Leaky layer, x.shape: [T, B, 128, 32, 32]
        x = self.lif1(x)

        # First block, x.shape: [T, B, 128, 32, 32]
        x = self.block1(x)

        # Second block, x.shape: [T, B, 256, 16, 16]
        x = self.block2(x)

        # First block, x.shape: [T, B, 512, 8, 8]
        x = self.block3(x)

        # Global average pool layer, x.shape: [T, B, 512, 1, 1]
        x = self.t_avg_pool(x)

        # Reshape of tensor, x.shape [T, B, 512]
        x = x.view(x.size(0), x.size(1), x.size(2))

        # First FC, x.shape: [T, B, 256]
        x = self.t_fc1(x)

        # Leaky layer, x.shape: [T, B, 256]
        x = self.lif2(x)

        # Computes the output logits per time steps, x.shape: [T, B, K]
        x = self.t_fc2(x)

        # Checks the model mode
        if self.training:
            # Outputs the logits for each time step to compute TSCELoss or knowledge distillation
            return x
        else:
            # Outputs the time-average of the logits to compute regular CE loss.
            return x.mean(0)


def make_student_model(
        in_channels: int,
        num_classes: int,
        beta: float,
        device: torch.device,
        state_dict: dict = None
) -> nn.Module:
    """Constructs and returns a Spiking ResNet19 student model.

    The model is then moved to the given device.
    Optionally loads a pre-trained state dictionary into the model.

    Args:
        in_channels: Number of input channels for the first convolutional layer.
        num_classes: Number of output classes for the final linear layer.
        beta: Initial membrane-decay constant for all LIF neurons.
        device: The device (CPU or GPU) to which the model will be offloaded.
        state_dict: A state dictionary of pre-trained weights to load into the model.

    Returns:
        nn.Module: The constructed student model on the specified device.
    """
    # Creates the student model custom architecture
    student = SResNet19(in_channels=in_channels, num_classes=num_classes, beta=beta)

    # Offloads the student model to the specified device
    student = student.to(device)

    # Checks if a parameter configuration must be loaded
    if state_dict is not None:
        # Loads the best parameters for the architecture
        student.load_state_dict(state_dict)

    return student
