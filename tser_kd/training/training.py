import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tser_kd.eval import MetricMeter, accuracy


def run_train(
        epoch: int,
        data_loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim,
        device: torch.device
) -> tuple:
    """Trains the model for one epoch over the provided data loader and tracks loss and accuracy.

    Args:
        epoch: Zero‐based index of the current training epoch.
        data_loader: PyTorch DataLoader providing batches for training.
        model: The neural network model to train.
        criterion: The loss function used to compute training loss.
        optimizer: The optimizer used to update model parameters.
        device: The device on which training computations will be performed.

    Returns:
        tuple: A 4-tuple containing:
            - train_loss.avg: Average training loss per example over this epoch.
            - train_acc.avg: Average top‐1 training accuracy (%) over this epoch.
            - epoch_time: Total elapsed wall‐clock time (in seconds) to complete this epoch.
            - batch_time.avg: Average time (in seconds) to process one batch during this epoch.
    """
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

            # Offload the inputs and targets to the desired device
            inputs = inputs.to(device)  # non-blocking=True should be tested for performance
            targets = targets.to(device)  # non-blocking=True should be tested for performance

            # Resets the gradients
            optimizer.zero_grad()   # Check set_to_none=True

            # Computes the model's predictions
            logits = model(inputs)      # Check CUDA mixed precision

            # Computes the loss value between predictions and targets
            loss_val = criterion(logits, targets)

            # Backprop and gradient update
            loss_val.backward()
            optimizer.step()

            # Computes accuracy of the model over the batch
            acc1, = accuracy(predictions=logits, targets=targets, top_k=(1,))

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
                batch_time=f"{batch_time.avg:.2f}s",
                refresh=False
            )
            pbar.update(1)

    return loss.avg, top1.avg, time.time() - start, batch_time.avg
