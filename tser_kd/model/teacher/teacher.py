import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


def make_teacher_model(arch: str, in_channels: int, num_classes: int, device: torch.device, state_dict: dict = None) -> nn.Module:
    """Constructs and returns a teacher model based on the specified architecture.

    The architecture is adapted for a custom input channel count and number of output classes.
    The model is then moved to the given device.
    Optionally loads a pre-trained state dictionary into the model.

    Currently, supports:
      - 'resnet-34': A ResNet-34 backbone pre-trained on ImageNet, with its first convolutional layer and final
        fully connected layer modified for CIFAR-10.

    Args:
        arch: Architecture name.
        in_channels: Number of input channels for the first convolutional layer.
        num_classes: Number of output classes for the final linear layer.
        device: The device (CPU or GPU) to which the model will be offloaded.
        state_dict: A state dictionary of pre-trained weights to load into the model.

    Returns:
        nn.Module: The constructed teacher model on the specified device.
    """
    teacher = None

    # ResNet architecture
    if arch == 'resnet-34':
        # Creates the base teacher model architecture for ImageNet
        teacher = resnet34(progress=True, weights=ResNet34_Weights.IMAGENET1K_V1)

        # Adapts the stem layer of the architecture to CIFAR10
        teacher.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        teacher.maxpool = nn.Identity()

        # Adapts the FC layer for classification
        teacher.fc = nn.Linear(in_features=teacher.fc.in_features, out_features=num_classes)

        # Offloads the teacher model to the specified device
        teacher = teacher.to(device)

    # Checks if a parameter configuration must be loaded
    if state_dict is not None:
        # Loads the best parameters for the architecture
        teacher.load_state_dict(state_dict)

    return teacher
