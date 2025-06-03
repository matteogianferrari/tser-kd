import torch
import numpy as np


class CutOut:
    """Randomly masks out one or more square patches from an image.

    This augmentation helps regularize CNNs by encouraging them to rely on distributed
    representations rather than focusing on a single region of the image.

    Attributes:
        n_patches: Number of square patches to cut out from each image.
        length: Side length (in pixels) of each square patch.
    """

    def __init__(self, n_patches: int, length: int) -> None:
        """Initializes the CutOut.

        Args:
            n_patches: Number of square patches to cut out from each image.
            length: Side length (in pixels) of each square patch.
        """
        self.n_patches = n_patches
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Applies CutOut to the given image tensor.

        A mask of shape (H, W) is created with ones everywhere except for `n_patches` randomly
        positioned squares, which are set to zero. The mask is then broadcast across the
        channel dimension and multiplied elementwise with the input image tensor.

        Args:
            img: Input image tensor of shape [C, H, W] (pixel values should be already normalized).

        Returns:
            torch.Tensor: The augmented image tensor of shape [C, H, W].
        """
        # Retrieves the image spatial dimensions
        H = img.size(1)
        W = img.size(2)

        # Creates the mask to apply over the image
        mask = np.ones(shape=(H, W), dtype=np.float32)

        # Applies the patches to the mask
        for _ in range(self.n_patches):
            # Randomly chooses the center of the patch
            y = np.random.randint(H)
            x = np.random.randint(W)

            # Computes the patch coordinates
            y1 = np.clip(y - self.length // 2, 0, H)
            y2 = np.clip(y + self.length // 2, 0, H)
            x1 = np.clip(x - self.length // 2, 0, W)
            x2 = np.clip(x + self.length // 2, 0, W)

            # Sets the values to 0
            mask[y1:y2, x1:x2] = 0.0

        # Converts the mask into a tensor
        mask = torch.from_numpy(mask).expand_as(img)

        # Applies the mask (broadcasting operation over the channels)
        img = img * mask

        return img
