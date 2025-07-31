from torchvision import transforms, datasets


def load_mnist_data() -> tuple:
    """Loads and returns the MNIST training and validation datasets.

    This function normalizes the images using MNIST’s channel mean and standard deviation.

    Returns:
        tuple: A 3-tuple containing:
            - train_dataset: The MNIST training set.
            - val_dataset: The MNIST validation (test) set.
            - num_class: The number of classes.
    """
    # Number of classes
    num_classes = 10

    # Standard data normalization for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Downloads the CIFAR10 datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_dataset, val_dataset, num_classes
