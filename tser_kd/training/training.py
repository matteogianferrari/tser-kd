import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from snntorch import utils
from tqdm import tqdm
from tser_kd.eval import accuracy, accuracy_snn
from tser_kd.utils import MetricMeter
from tser_kd.dataset import Encoder


def run_train(
    epoch: int,
    data_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    encoder: Encoder = None
) -> tuple:
    """Trains the model for one epoch over the provided data loader and tracks loss and accuracy.

    This function uses PyTorch AMP to perform model's predictions.

    Args:
        epoch: Zero‐based index of the current training epoch.
        data_loader: PyTorch DataLoader providing batches for training.
        model: The neural network model to train.
        criterion: The loss function used to compute training loss.
        optimizer: The optimizer used to update model parameters.
        device: The device on which training computations will be performed.
        scaler: The scaler used with PyTorch AMP.
        encoder: Encoder used to convert images into spike trains.

    Returns:
        tuple: A 4-tuple containing:
            - train_loss.avg: Average training loss per example over this epoch.
            - train_acc.avg: Average top‐1 training accuracy (%) over this epoch.
            - epoch_time: Total elapsed wall‐clock time (in seconds) to complete this epoch.
            - batch_time.avg: Average time (in seconds) to process one batch during this epoch.
    """
    # Device check
    if device == 'cpu':
        device_type = 'cpu'
    else:
        device_type = 'cuda'

    # Puts the model in training mode
    model.train()

    # Metrics recorders
    batch_time = MetricMeter()
    loss = MetricMeter()
    top1 = MetricMeter()    # Train accuracy

    # Starts epoch timer
    start = time.time()

    # Dynamic bar
    with tqdm(
            total=len(data_loader),
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| Batch {n_fmt}/{total_fmt} {postfix}",
    ) as pbar:
        # Process the batches in the dataset
        for idx, (inputs, targets) in enumerate(data_loader):
            # Starts the timer
            ref_time = time.time()

            # Offload the inputs and targets to the desired device with asynchronous operation
            # inputs.shape: [B, C, H, W]
            inputs = inputs.to(device, non_blocking=True)

            # targets.shape: [B]
            targets = targets.to(device, non_blocking=True)

            # Resets the gradients (set_to_none speeds up the operation)
            optimizer.zero_grad(set_to_none=True)

            # CUDA automatic mixed precision
            with torch.amp.autocast(device_type=device_type):
                # Checks if the model is an ANN or a SNN
                if encoder is not None:
                    # SNN
                    # Encodes the data with the specified encoder type, inputs.shape: [T, B, C, H, W]
                    inputs = encoder(inputs)

                    # Resets LIF neurons' hidden states
                    utils.reset(model)

                    # Computes the model's logits, logits.shape: [T, B, K]
                    logits = model(inputs)

                    # Computes the loss value between logits and targets
                    loss_val = criterion(logits, targets)
                else:
                    # ANN
                    # Computes the model's logits, logits.shape: [B, K]
                    logits = model(inputs)

                    # Computes the loss value between logits and targets
                    loss_val = criterion(logits, targets)

            # Scales AMP loss and apply backprop
            scaler.scale(loss_val).backward()

            # optimizer.step() is called automatically
            scaler.step(optimizer)

            # Updates the scale for the next iteration
            scaler.update()

            # Checks if the model is an ANN or a SNN
            if encoder is not None:
                # SNN
                # Computes the accuracy of the model
                acc1, = accuracy_snn(logits=logits, targets=targets, top_k=(1,))
            else:
                # ANN
                # Computes accuracy of the model over the batch
                acc1, = accuracy(logits=logits, targets=targets, top_k=(1,))

            # Retrieves the batch-size
            B = targets.size(0)

            # Updates the metrics
            loss.update(val=loss_val.item(), n=B)
            top1.update(val=acc1.item(), n=B)
            batch_time.update(val=time.time() - ref_time)

            # Bar update
            pbar.set_postfix(
                loss=f"{loss.avg:.4f}",
                acc=f"{top1.avg:.2f}%",
                refresh=False
            )
            pbar.update(1)

    return loss.avg, top1.avg, time.time() - start, batch_time.avg


def run_kd_train(
    epoch: int,
    data_loader: DataLoader,
    s_model: nn.Module,
    t_model: nn.Module,
    criterion: nn.Module,
    optimizer: optim,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    encoder: Encoder
) -> tuple:
    # Device check
    if device == 'cpu':
        device_type = 'cpu'
    else:
        device_type = 'cuda'

    # Puts the student model in training mode and the teacher in evaluation mode
    s_model.train()
    t_model.eval()

    # Metrics recorders
    batch_time = MetricMeter()
    total_loss = MetricMeter()
    ce_loss = MetricMeter()
    kl_loss = MetricMeter()
    e_reg = MetricMeter()
    top1 = MetricMeter()    # Train accuracy

    # Starts epoch timer
    start = time.time()

    # Dynamic bar
    with tqdm(
            total=len(data_loader),
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| Batch {n_fmt}/{total_fmt} {postfix}",
    ) as pbar:
        # Process the batches in the dataset
        for idx, (t_inputs, targets) in enumerate(data_loader):
            # Starts the timer
            ref_time = time.time()

            # Offload the inputs and targets to the desired device with asynchronous operation
            # t_inputs.shape: [B, C, H, W]
            t_inputs = t_inputs.to(device, non_blocking=True)

            # Encodes the data with the specified encoder type, s_inputs.shape: [T, B, C, H, W]
            s_inputs = encoder(t_inputs)

            # targets.shape: [B]
            targets = targets.to(device, non_blocking=True)

            # Resets the gradients (set_to_none speeds up the operation)
            optimizer.zero_grad(set_to_none=True)

            # CUDA automatic mixed precision
            with torch.amp.autocast(device_type=device_type):
                # Resets LIF neurons' hidden states
                utils.reset(s_model)

                # Computes the student model's logits, s_logits.shape: [T, B, K]
                s_logits = s_model(s_inputs)

                with torch.no_grad():
                    # Computes the teacher model's logits, t_logits.shape: [B, K]
                    t_logits = t_model(t_inputs)

                # Computes the loss values using knowledge distillation with entropy regularization
                total_val, ce_val, kl_val, e_val = criterion(t_logits, s_logits, targets)

            # Scales AMP loss and apply backprop
            scaler.scale(total_val).backward()

            # optimizer.step() is called automatically
            scaler.step(optimizer)

            # Updates the scale for the next iteration
            scaler.update()

            # Computes the accuracy of the model
            acc1, = accuracy_snn(logits=s_logits, targets=targets, top_k=(1,))

            # Retrieves the batch-size
            B = targets.size(0)

            # Updates the metrics
            total_loss.update(val=total_val.item(), n=B)
            ce_loss.update(val=ce_val.item(), n=B)
            kl_loss.update(val=kl_val.item(), n=B)
            e_reg.update(val=e_val.item(), n=B)
            top1.update(val=acc1.item(), n=B)
            batch_time.update(val=time.time() - ref_time)

            # Bar update
            pbar.set_postfix(
                total_loss=f"{total_loss.avg:.4f}",
                ce_loss=f"{ce_loss.avg:.4f}",
                kl_loss=f"{kl_loss.avg:.4f}",
                e_reg=f"{e_reg.avg:.4f}",
                acc=f"{top1.avg:.2f}%",
                refresh=False
            )
            pbar.update(1)

    return total_loss.avg, ce_loss.avg, kl_loss.avg, e_reg.avg, top1.avg, time.time() - start, batch_time.avg
