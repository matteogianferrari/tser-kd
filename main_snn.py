import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb
import snntorch as snn
from snntorch import surrogate
from tser_kd.utils import setup_seed, AccuracyMonitor
from tser_kd.dataset import load_cifar10_data, load_mnist_data, StaticEncoder
from tser_kd.model.student import make_student_model
from tser_kd.model import TSCELoss
from tser_kd.eval import run_eval
from tser_kd.training import run_train
from config_snn import args, args_dict


# Setups the seed for reproducibility
setup_seed(42)


def initialize_wandb(config: dict):
    import time
    """
    Initializes Weights and Biases (wandb) with the given configuration.

    Args:
        configuration: Configuration parameters for the run.
    """
    # Name the run using current time and configuration name
    run_complete_name = f"{time.strftime('%Y%m%d%H%M%S')}-{args.run_name}"

    return wandb.init(project="tser-kd", name=run_complete_name, config=dict(config), group='student')


if __name__ == '__main__':
    # Selects the device for the experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset creation
    if args.dataset == 'cifar10':
        train_dataset, val_dataset, num_classes = load_cifar10_data(auto_aug=args.auto_aug, cutout=args.cutout)
        in_channels = 3
    elif args.dataset == 'mnist':
        train_dataset, val_dataset, num_classes = load_mnist_data()
        in_channels = 1

    # Creates the train and test DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Selects the surrogate gradient
    if args.snn_grad == 'atan':
        spike_grad = surrogate.atan()

    # Check if an existing state_dict must be loaded
    s_state_dict = None

    if args.s_weight is not None:
        s_state_dict = torch.load(args.s_weight, map_location="cpu")

    # Creates the model architecture
    s_model = make_student_model(
        arch=args.student_arch,
        in_channels=in_channels,
        num_classes=num_classes,
        beta=args.beta,
        threshold=args.v_th,
        spike_grad=spike_grad,
        device=device,
        learn_beta=args.learn_beta,
        learn_threshold=args.learn_threshold,
        state_dict=s_state_dict
    )

    # Creates the optimizer
    if args.optimizer == 'adamw':
        opt = optim.AdamW(s_model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        opt = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # Creates the scheduler
    if args.scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=args.lr_patience, factor=args.lr_factor)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Creates the loss function
    train_criterion = TSCELoss()
    eval_criterion = nn.CrossEntropyLoss()

    # Creates the encoder
    encoder = StaticEncoder(num_steps=args.t_steps)

    # Crates the scaler
    scaler = torch.amp.GradScaler(device='cuda')

    # Accuracy monitor
    acc_monitor = AccuracyMonitor(path=args.model_path)

    # Initializes wandb
    run = initialize_wandb(args_dict)

    # Training
    epoch_i = 0
    curr_lr = opt.param_groups[0]["lr"]

    for epoch_i in range(args.epochs):
        # Forward pass
        train_loss, train_acc, epoch_time, train_batch_time = run_train(
            epoch_i, train_loader, s_model, train_criterion, opt, device, scaler, encoder
        )

        val_loss, val_acc1, val_acc5, val_batch_time = run_eval(val_loader, s_model, eval_criterion, device, encoder)

        # Updates the LR
        if args.scheduler == 'reduce':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        curr_lr = opt.param_groups[0]["lr"]

        # Logging
        print(
            f"Time: {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc1: {val_acc1:.2f}% | Val Acc5: {val_acc5:.2f}% | LR: {curr_lr:.6f}"
        )

        run.log({
            "Train Loss": train_loss, "Train Accuracy": train_acc,
            "Validation Loss": val_loss, "Validation Accuracy 1": val_acc1, "Validation Accuracy 5": val_acc5,
            "Epoch Time": epoch_time, "Learning Rate": curr_lr,
            "Train Batch Time": train_batch_time, "Validation Batch Time": val_batch_time
        })

        # Accuracy monitor
        acc_monitor(val_acc1, epoch_i, s_model)

    s_model.load_state_dict(torch.load(args.model_path))
    test_ce_loss, test_acc1, test_acc5, _ = run_eval(val_loader, s_model, eval_criterion, device, encoder)
    run.log({"Test CE Loss": test_ce_loss, "Test Accuracy 1": test_acc1, "Test Accuracy 5": test_acc5})

    # Ends the run
    run.finish()
