import torch
import torch.nn as nn


def transfer_weights_resnet18_sresnet18(r18: nn.Module, sr18: nn.Module, trainable: bool = True) -> None:
    """Transfers the learned weights from ResNet18 to SResNet18.

    Batch norm parameters are excluded. Biases are disabled by architecture design.

    Args:
        r18: The trained ResNet18 model.
        sr18: The new SResNet18 model.
        trainable: Flag that allows the weights to be learnable, False freezes the weights.
    """
    # ResNet18 mapping
    r18_dict = {
        "conv1": r18.conv1.weight,
        "layer1.0.conv1": r18.layer1[0].conv1.weight,
        "layer1.0.conv2": r18.layer1[0].conv2.weight,
        "layer1.1.conv1": r18.layer1[1].conv1.weight,
        "layer1.1.conv2": r18.layer1[1].conv2.weight,
        "layer2.0.conv1": r18.layer2[0].conv1.weight,
        "layer2.0.conv2": r18.layer2[0].conv2.weight,
        "layer2.0.downsample.0": r18.layer2[0].downsample[0].weight,
        "layer2.1.conv1": r18.layer2[1].conv1.weight,
        "layer2.1.conv2": r18.layer2[1].conv2.weight,
        "layer3.0.conv1": r18.layer3[0].conv1.weight,
        "layer3.0.conv2": r18.layer3[0].conv2.weight,
        "layer3.0.downsample.0": r18.layer3[0].downsample[0].weight,
        "layer3.1.conv1": r18.layer3[1].conv1.weight,
        "layer3.1.conv2": r18.layer3[1].conv2.weight,
        "layer4.0.conv1": r18.layer4[0].conv1.weight,
        "layer4.0.conv2": r18.layer4[0].conv2.weight,
        "layer4.0.downsample.0": r18.layer4[0].downsample[0].weight,
        "layer4.1.conv1": r18.layer4[1].conv1.weight,
        "layer4.1.conv2": r18.layer4[1].conv2.weight,
        "fc": r18.fc.weight
    }

    # SResNet18 mapping
    sr18_dict = {
        "conv1": sr18.stem.layer.weight,
        "layer1.0.conv1": sr18.stages[0][0].t_conv_bn1.layer.weight,
        "layer1.0.conv2": sr18.stages[0][0].t_conv_bn2.layer.weight,
        "layer1.1.conv1": sr18.stages[0][1].t_conv_bn1.layer.weight,
        "layer1.1.conv2": sr18.stages[0][1].t_conv_bn2.layer.weight,
        "layer2.0.conv1": sr18.stages[1][0].t_conv_bn1.layer.weight,
        "layer2.0.conv2": sr18.stages[1][0].t_conv_bn2.layer.weight,
        "layer2.0.downsample.0": sr18.stages[1][0].shortcuts.layer.weight,
        "layer2.1.conv1": sr18.stages[1][1].t_conv_bn1.layer.weight,
        "layer2.1.conv2": sr18.stages[1][1].t_conv_bn2.layer.weight,
        "layer3.0.conv1": sr18.stages[2][0].t_conv_bn1.layer.weight,
        "layer3.0.conv2": sr18.stages[2][0].t_conv_bn2.layer.weight,
        "layer3.0.downsample.0": sr18.stages[2][0].shortcuts.layer.weight,
        "layer3.1.conv1": sr18.stages[2][1].t_conv_bn1.layer.weight,
        "layer3.1.conv2": sr18.stages[2][1].t_conv_bn2.layer.weight,
        "layer4.0.conv1": sr18.stages[3][0].t_conv_bn1.layer.weight,
        "layer4.0.conv2": sr18.stages[3][0].t_conv_bn2.layer.weight,
        "layer4.0.downsample.0": sr18.stages[3][0].shortcuts.layer.weight,
        "layer4.1.conv1": sr18.stages[3][1].t_conv_bn1.layer.weight,
        "layer4.1.conv2": sr18.stages[3][1].t_conv_bn2.layer.weight,
        "fc": sr18.mlp.layer.weight
    }

    # Checks for mismatches
    common_keys = r18_dict.keys() & sr18_dict.keys()
    if not common_keys:
        raise ValueError("No overlapping layer names between the supplied dictionaries.")

    with torch.no_grad():
        for k in common_keys:
            src, dst = r18_dict[k], sr18_dict[k]

            if src.shape != dst.shape:
                raise ValueError(f"Shape mismatch on layer '{k}': {src.shape} vs {dst.shape}")

            # Copy the weights and set the trainable flag
            dst.copy_(src)
            dst.requires_grad = trainable

    # Warns about any entries that were not transferred
    missing_ann = r18_dict.keys() - sr18_dict.keys()
    missing_snn = sr18_dict.keys() - r18_dict.keys()
    if missing_ann:
        print(f"[transfer] ANN layers not present in SNN mapping and therefore "
              f"not copied: {sorted(missing_ann)}")
    if missing_snn:
        print(f"[transfer] SNN layers not initialised (no source weights): "
              f"{sorted(missing_snn)}")

    if not missing_ann and not missing_snn:
        print("[transfer] Successfully transferred all weights.")

    return


def transfer_weights_resnet18_resnet19(r18: nn.Module, r19: nn.Module, trainable: bool = True) -> None:
    """Transfers the learned weights from ResNet18 to ResNet19.

    Batch norm parameters are excluded. Biases are disabled by architecture design.

    Args:
        r18: The trained ResNet18 model.
        r19: The new ResNet19 model.
        trainable: Flag that allows the weights to be learnable, False freezes the weights.
    """
    # ResNet18 mapping
    r18_dict = {
        "layer1.0.conv1": r18.layer2[1].conv1.weight,
        "layer1.0.conv2": r18.layer2[1].conv2.weight,
        "layer1.1.conv1": r18.layer2[1].conv1.weight,
        "layer1.1.conv2": r18.layer2[1].conv2.weight,
        "layer1.2.conv1": r18.layer2[1].conv1.weight,
        "layer1.2.conv2": r18.layer2[1].conv2.weight,
        "layer2.0.conv1": r18.layer3[0].conv1.weight,
        "layer2.0.conv2": r18.layer3[0].conv2.weight,
        "layer2.0.downsample.0": r18.layer3[0].downsample[0].weight,
        "layer2.1.conv1": r18.layer3[1].conv1.weight,
        "layer2.1.conv2": r18.layer3[1].conv2.weight,
        "layer2.2.conv1": r18.layer3[1].conv1.weight,
        "layer2.2.conv2": r18.layer3[1].conv2.weight,
        "layer3.0.conv1": r18.layer4[0].conv1.weight,
        "layer3.0.conv2": r18.layer4[0].conv2.weight,
        "layer3.0.downsample.0": r18.layer4[0].downsample[0].weight,
        "layer3.1.conv1": r18.layer4[1].conv1.weight,
        "layer3.1.conv2": r18.layer4[1].conv2.weight,
        "fc": r18.fc.weight
    }

    # ResNet19 mapping
    r19_dict = {
        "layer1.0.conv1": r19.block1[0].conv1.weight,
        "layer1.0.conv2": r19.block1[0].conv2.weight,
        "layer1.1.conv1": r19.block1[1].conv1.weight,
        "layer1.1.conv2": r19.block1[1].conv2.weight,
        "layer1.2.conv1": r19.block1[2].conv1.weight,
        "layer1.2.conv2": r19.block1[2].conv2.weight,
        "layer2.0.conv1": r19.block2[0].conv1.weight,
        "layer2.0.conv2": r19.block2[0].conv2.weight,
        "layer2.0.downsample.0": r19.block2[0].shortcuts[0].weight,
        "layer2.1.conv1": r19.block2[1].conv1.weight,
        "layer2.1.conv2": r19.block2[1].conv2.weight,
        "layer2.2.conv1": r19.block2[2].conv1.weight,
        "layer2.2.conv2": r19.block2[2].conv2.weight,
        "layer3.0.conv1": r19.block3[0].conv1.weight,
        "layer3.0.conv2": r19.block3[0].conv2.weight,
        "layer3.0.downsample.0": r19.block3[0].shortcuts[0].weight,
        "layer3.1.conv1": r19.block3[1].conv1.weight,
        "layer3.1.conv2": r19.block3[1].conv2.weight,
        "fc": r19.mlp.weight
    }

    # Checks for mismatches
    common_keys = r18_dict.keys() & r19_dict.keys()
    if not common_keys:
        raise ValueError("No overlapping layer names between the supplied dictionaries.")

    with torch.no_grad():
        for k in common_keys:
            src, dst = r18_dict[k], r19_dict[k]

            if src.shape != dst.shape:
                raise ValueError(f"Shape mismatch on layer '{k}': {src.shape} vs {dst.shape}")

            # Copy the weights and set the trainable flag
            dst.copy_(src)
            dst.requires_grad = trainable

    # Warns about any entries that were not transferred
    missing_ann1 = r18_dict.keys() - r19_dict.keys()
    missing_ann2 = r19_dict.keys() - r18_dict.keys()
    if missing_ann1:
        print(f"[transfer] ANN layers not present in ANN mapping and therefore "
              f"not copied: {sorted(missing_ann1)}")
    if missing_ann2:
        print(f"[transfer] ANN layers not initialised (no source weights): "
              f"{sorted(missing_ann2)}")

    if not missing_ann1 and not missing_ann2:
        print("[transfer] Successfully transferred all weights.")

    return
