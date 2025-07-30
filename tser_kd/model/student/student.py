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


class SResNet(nn.Module):
    """Spiking ResNet generic architecture.

    This spiking version of ResNet can be customized to create different size of the architecture.

    Attributes:
        stem: 3×3 convolution followed by temporal batch normalization.
        lif_stem: LIF neuron layer after the stem.
        stages: List containing the residual stages of the model (its size depends on the architecture).
        t_avg_pool: Global average-pooling layer applied over time.
        mlp: Classification head that maps the features to 'num_classes' logits per time step.
    """

    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            beta: float,
            stem_channels: int,
            stage_blocks: list[int],
            stage_channels: list[int],
            fc_hidden_dims: list[int] | None = None
    ) -> None:
        """Initializes the SResNet.

        Args:
            in_channels: Number of channels in the input frames.
            num_classes: Number of output classes.
            beta: Membrane-decay constant for all LIF neurons.
            stem_channels: Number of channels in the stem layer.
            stage_blocks: List containing the number of basic blocks for each stage.
            stage_channels: List containing the number of channels for each stage.
            fc_hidden_dims: List containing the number of features for each intermediate MLP layer.
        """
        super(SResNet, self).__init__()

        # Stem block
        self.stem = LayerTWrapper(
            layer=conv3x3(in_channels=in_channels, out_channels=stem_channels),
            batch_norm=TDBatchNorm2d(num_features=stem_channels),
        )
        self.lif_stem = LIFTWrapper(layer=snn.Leaky(beta=beta, init_hidden=True))

        # Residual blocks
        self.stages = nn.ModuleList()

        # Creates the stages that compose the residual part of the architecture
        in_c = stem_channels
        for blocks, out_c in zip(stage_blocks, stage_channels):
            # Appends a customized stage
            self.stages.append(
                self._make_stage(num_blocks=blocks, in_channels=in_c, out_channels=out_c, beta=beta)
            )
            # Updates the input channels for the next stage
            in_c = out_c

        # Global pool
        self.t_avg_pool = LayerTWrapper(layer=nn.AdaptiveAvgPool2d((1, 1)))

        # MLP head
        # Initialization depending on if the architecture has multiple layers in the MLP or not
        fc_hidden_dims = fc_hidden_dims or []

        # Creates a list containing the number of features that the MLP will possess
        # Starts with the output of the global average pooling, then adds hidden features dimension if any,
        # then ends with the number of classes
        dims = [stage_channels[-1], *fc_hidden_dims, num_classes]

        # Creates a list containing the layers in the MLP head
        mlp_layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            # Adds the customized layer into the list
            mlp_layers.append(LayerTWrapper(layer=nn.Linear(in_features=d_in, out_features=d_out, bias=False)))

            # Adds a LIF layer between the FC layers if more than one is present
            if i < len(dims) - 2:
                mlp_layers.append(LIFTWrapper(layer=snn.Leaky(beta=beta, init_hidden=True)))

        # Creates the MLP head
        self.mlp = nn.Sequential(*mlp_layers)

    def _make_stage(self, num_blocks: int, in_channels: int, out_channels: int, beta: float) -> nn.Sequential:
        """Builds a SResNet stage with the specific configuration.

        Args:
            num_blocks: How many SResNet blocks to stack.
            in_channels: Number of input channels for the stage.
            out_channels: Number of output channels for the stage.
            beta: Initial membrane-decay constant for the stage.

        Returns:
            A sequential container with the specified configuration.
        """
        layers = []

        for block_i in range(num_blocks):
            # Selects the stride based on the block index and the matching of input and output channels
            stride = 2 if (block_i == 0 and in_channels != out_channels) else 1

            # Adds a block
            layers.append(
                SResNetBlock(in_channels=in_channels, out_channels=out_channels, stride=stride, beta=beta)
            )

            # Updates the input channels for the next block
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [T, B, C, H, W].

        Returns:
            torch.Tensor: Logits for every time step if the model is in training mode, shape [T, B, K],
                mean of logits over time steps if the model is in eval mode, shape [B, K].
        """
        # Stem block
        x = self.lif_stem(self.stem(x))

        # Residual stages
        for stage in self.stages:
            x = stage(x)

        # Global pool
        x = self.t_avg_pool(x)
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
        device: torch.device,
        state_dict: dict = None
) -> nn.Module:
    """Constructs and returns a Spiking student model based on the specified architecture.

    The model is then moved to the given device.
    Optionally loads a pre-trained state dictionary into the model.

    Currently, supports:
        - 'sresnet-18': A custom SResNet-18 backbone for CIFAR-10.
        - 'sresnet-19': A custom SResNet-19 backbone for CIFAR-10.

    Args:
        arch: Architecture name.
        in_channels: Number of input channels for the first convolutional layer.
        num_classes: Number of output classes for the final linear layer.
        beta: Initial membrane-decay constant for all LIF neurons.
        device: The device (CPU or GPU) to which the model will be offloaded.
        state_dict: A state dictionary of pre-trained weights to load into the model.

    Returns:
        nn.Module: The constructed student model on the specified device.
    """
    student = None

    # ResNet architecture
    if arch == 'sresnet-18':
        # Creates the student model architecture
        student = SResNet(
            in_channels=in_channels,
            num_classes=num_classes,
            beta=beta,
            stem_channels=64,
            stage_blocks=[2, 2, 2, 2],
            stage_channels=[64, 128, 256, 512]
        )
    elif arch == 'sresnet-19':
        # Creates the student model architecture
        student = SResNet(
            in_channels=in_channels,
            num_classes=num_classes,
            beta=beta,
            stem_channels=128,
            stage_blocks=[3, 3, 2],
            stage_channels=[128, 256, 512],
            fc_hidden_dims=[256]
        )

    # Checks if a parameter configuration must be loaded
    if state_dict is not None:
        # Loads the best parameters for the architecture
        student.load_state_dict(state_dict)

    # Offloads the student model to the specified device
    student = student.to(device)

    return student
