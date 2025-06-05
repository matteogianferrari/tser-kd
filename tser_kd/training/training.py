import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from snntorch import utils
from tqdm import tqdm
from tser_kd.eval import MetricMeter, accuracy


def forward_pass(snn_model: nn.Module, input_spikes: torch.Tensor) -> tuple:
    """Performs a time‐stepped forward pass through a SNN model.

    This function resets all hidden LIF neuron states in the 'snn_model' before iterating
    over 'num_steps' time steps. At each time step, it feeds the corresponding slice of
    'input_spikes' into the model, obtains the output spike tensor and membrane potential
    tensor, and appends them to recording lists.

    Args:
        snn_model: A spiking neural network model.
        input_spikes: A tensor of shape [T, B, C, H, W] containing the input spikes for each time step.

    Returns:
        tuple: A 2‐tuple containing:
            - spk_rec: A tensor of shape [T, K] where each slice is the output spikes from the model.
            - mem_rec: A tensor of shape [T, K] where each slice  is the membrane potential from the model.
    """
    # Membranes potentials through time
    mem_rec = []

    # Neurons spikes through time
    spk_rec = []

    # Resets the hidden states for all LIF neurons in the network
    utils.reset(snn_model)

    # Retrieves the time steps
    T = input_spikes.shape(0)

    # Iterates through time steps
    for t in range(T):  # Check LIF parallel to avoid loop
        # Model's predictions
        spk_out, mem_out = snn_model(input_spikes[t])

        # Records spikes and membrane potentials for time step t
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    # Converts the lists into tensors
    spk_rec = torch.stack(spk_rec)
    mem_rec = torch.stack(mem_rec)

    return spk_rec, mem_rec


def run_train(
        epoch: int,
        data_loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim,
        device: torch.device,
        scaler: torch.amp.GradScaler
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
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Resets the gradients (set_to_none speeds up the operation)
            optimizer.zero_grad(set_to_none=True)

            # CUDA automatic mixed precision
            with torch.amp.autocast(device_type=device_type):
                # Computes the model's predictions
                logits = model(inputs)

                # Computes the loss value between predictions and targets
                loss_val = criterion(logits, targets)

            # Scales AMP loss and apply backprop
            scaler.scale(loss_val).backward()

            # optimizer.step() is called automatically
            scaler.step(optimizer)

            # Updates the scale for the next iteration
            scaler.update()

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
                refresh=False
            )
            pbar.update(1)

    return loss.avg, top1.avg, time.time() - start, batch_time.avg
