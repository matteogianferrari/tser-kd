from torchvision import transforms, datasets
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from tser_kd.dataset import CutOut


def load_cifar10_data(auto_aug: bool = False, cutout: bool = False) -> tuple:
    """Loads and returns the CIFAR-10 training and validation datasets with optional augmentations.

    This function applies standard CIFAR-10 data augmentations (random crop with padding and horizontal flip),
    and optionally adds AutoAugment and CutOut augmentations. Finally, it normalizes the images using
    CIFAR-10’s channel mean and standard deviation.

    Args:
        auto_aug: If True, applies the AutoAugment policy for CIFAR-10 to each training image.
        cutout: If True, applies a single CutOut patch of size 16×16 to each training image.

    Returns:
        tuple: A 3-tuple containing:
            - train_dataset: The CIFAR-10 training set with the specified augmentations.
            - val_dataset: The CIFAR-10 validation (test) set with only normalization.
            - num_class: The number of classes.
    """
    # Number of classes
    num_classes = 10

    # Standard data augmentation for CIFAR10
    train_transform = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]

    # Adds AutoAugment if specified
    if auto_aug:
        train_transform.append(AutoAugment(policy=AutoAugmentPolicy.CIFAR10))

    # Transform image to tensor
    train_transform.append(transforms.ToTensor())

    # Adds CutOut if specified
    if cutout:
        train_transform.append(CutOut(n_patches=1, length=16))

    # Adds normalization (mean and variance)
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform.append(normalize)

    # Composes the training's data augmentation
    train_transform = transforms.Compose(train_transform)

    # Composes the validation's data augmentation
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Downloads the CIFAR10 datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    return train_dataset, val_dataset, num_classes
