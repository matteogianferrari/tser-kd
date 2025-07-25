import torch
import torch.nn as nn


def transfer_weights(snn: nn.Module, ann: nn.Module, flag: bool = False) -> None:
    """Transfers the learnable weights and biases of an ANN to a SNN.

    This function assumes that the 2 models share the same architecture, so that
    the weights and biases can be transferred without conflicts.

    All copy operations are wrapped in 'torch.no_grad()' so that PyTorch’s autograd
    engine does not record the in‑place writes.

    Biases in FC1 and FC2 are also copied. Other biases, batch norm statistics,
    and other buffers are not copied.

    Args:
        snn: Target Spiking Neural Network that will receive the weights.
        ann: Source Artificial Neural Network that donates the pre-trained weights.
        flag: Controls if the copied parameters in snn should remain trainable or not.
            True keeps the weights trainable, False freezes the weights.
    """
    def _copy(source: torch.Tensor, target: torch.Tensor, trainable: bool) -> None:
        """In-place copy of weights from source to target tensors.

        Args:
            source: Source tensor that donates the learned weights.
            target: Target tensor that receives the learned weights.
            trainable: True to keep the weights learnable, False to freeze them.
        """
        # Copies the tensor
        target.copy_(source)

        # Sets the flag
        if not trainable and target.requires_grad:
            target.requires_grad = False

    # Tells autograd to not track the copy operations
    # maintaining it legal and no unnecessary computation graph is built
    with torch.no_grad():
        # Stem
        # Copy operation
        _copy(source=ann.stem[0].weight, target=snn.stem.layer.weight, trainable=flag)

        # Block 1
        for i in range(3):
            # Stores the source and target
            src_block, tgt_block = ann.block1[i], snn.block1[i]

            # Copy operations
            _copy(source=src_block.conv1.weight, target=tgt_block.t_conv_bn1.layer.weight, trainable=flag)
            _copy(source=src_block.conv2.weight, target=tgt_block.t_conv_bn2.layer.weight, trainable=flag)

        # Block 2
        for i in range(3):
            # Stores the source and target
            src_block, tgt_block = ann.block2[i], snn.block2[i]

            # Shortcut in residual block
            if i == 0:
                _copy(source=src_block.shortcuts[0].weight, target=tgt_block.shortcuts.layer.weight, trainable=flag)

            # Copy operations
            _copy(source=src_block.conv1.weight, target=tgt_block.t_conv_bn1.layer.weight, trainable=flag)
            _copy(source=src_block.conv2.weight, target=tgt_block.t_conv_bn2.layer.weight, trainable=flag)

        # Block 3
        for i in range(2):
            # Stores the source and target
            src_block, tgt_block = ann.block3[i], snn.block3[i]

            # Shortcut in residual block
            if i == 0:
                _copy(source=src_block.shortcuts[0].weight, target=tgt_block.shortcuts.layer.weight, trainable=flag)

            # Copy operations
            _copy(source=src_block.conv1.weight, target=tgt_block.t_conv_bn1.layer.weight, trainable=flag)
            _copy(source=src_block.conv2.weight, target=tgt_block.t_conv_bn2.layer.weight, trainable=flag)

        # FC
        # Copy operations
        _copy(source=ann.fc1.weight, target=snn.t_fc1.layer.weight, trainable=flag)
        _copy(source=ann.fc1.bias, target=snn.t_fc1.layer.bias, trainable=flag)
        _copy(source=ann.fc2.weight, target=snn.t_fc2.layer.weight, trainable=flag)
        _copy(source=ann.fc2.bias, target=snn.t_fc2.layer.bias, trainable=flag) 
