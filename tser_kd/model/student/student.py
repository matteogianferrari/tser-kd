import torch
import torch.nn as nn
import snntorch as snn
from tser_kd.model.student import conv3x3, conv1x1, TDBatchNorm2d, TWrapLayer


class SResNetBlock(nn.Module):
    """Basic building block for S-ResNet architectures.

    Attributes:
        main_branch:
        shortcut:
        lif:
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            beta: float
    ) -> None:
        """Initializes the SResNetBlock.

        Args:
            in_channels:
            out_channels:
            stride:
            beta:
        """
        super(SResNetBlock, self).__init__()

        # Main branch
        self.t_conv_bn1 = TWrapLayer(
            layer=conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride),
            batch_norm=TDBatchNorm2d(num_features=out_channels)
        )
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)

        self.t_conv_bn2 = TWrapLayer(
            layer=conv3x3(in_channels=out_channels, out_channels=out_channels),
            batch_norm=TDBatchNorm2d(num_features=out_channels, alpha=1/(2**0.5))
        )
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)

        # Shortcut branch
        self.shortcuts = None
        if stride != 1 or in_channels != out_channels:
            self.shortcuts = TWrapLayer(
                layer=conv1x1(in_channels=in_channels, out_channels=out_channels, stride=stride),
                batch_norm=TDBatchNorm2d(num_features=out_channels, alpha=1/(2**0.5))
            )

        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x:

        Returns:

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
    """

    """

    def __init__(self, in_channels: int, num_classes: int, beta: float) -> None:
        """

        Args:
            in_channels:
            num_classes:
            beta:
        """
        super(SResNet19, self).__init__()

        self.start_channels = 128

        self.stem = TWrapLayer(
            layer=conv3x3(in_channels=in_channels, out_channels=128),
            batch_norm=TDBatchNorm2d(num_features=128)
        )
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)

        # Block 1
        self.block1 = self._make_block(num_blocks=3, out_channels=128, beta=beta)

        # Block 2
        self.block2 = self._make_block(num_blocks=3, out_channels=256, beta=beta)

        # Block 3
        self.block3 = self._make_block(num_blocks=2, out_channels=512, beta=beta)

        self.t_avg_pool = TWrapLayer(layer=nn.AdaptiveAvgPool2d((1, 1)))

        self.t_fc1 = TWrapLayer(layer=nn.Linear(in_features=512, out_features=256, bias=False))
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)

        self.t_fc2 = TWrapLayer(layer=nn.Linear(in_features=256, out_features=num_classes, bias=False))

    def _make_block(self, num_blocks: int, out_channels: int, beta: float) -> nn.Sequential:
        """

        Args:
            num_blocks:
            out_channels:
            beta:

        Returns:

        """
        layers = []

        for _ in range(num_blocks):
            if self.start_channels != out_channels:
                stride = 2
            else:
                stride = 1

            layers.append(
                SResNetBlock(in_channels=self.start_channels, out_channels=out_channels, stride=stride, beta=beta)
            )

            self.start_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x:

        Returns:

        """
        x = self.stem(x)
        x = self.lif1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.t_avg_pool(x)
        x = x.view(x.size(0), x.size(1), x.size(2))

        x = self.t_fc1(x)
        x = self.lif2(x)

        return self.t_fc2(x).mean(0)
