import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb
from tser_kd.utils import setup_seed, AccuracyMonitor
from tser_kd.dataset import load_cifar10_data, load_mnist_data
from tser_kd.model.teacher import make_teacher_model
from tser_kd.model import transfer_weights_resnet18_resnet19
from tser_kd.eval import run_eval
from tser_kd.training import run_train
from config_ann import args, args_dict


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

    return wandb.init(project="tser-kd", name=run_complete_name, config=dict(config), group='teacher')


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

    # Check if an existing state_dict must be loaded
    t_state_dict = None

    if args.t_weight is not None:
        t_state_dict = torch.load(args.t_weight, map_location="cpu")

    # Creates the model architecture
    t_model = make_teacher_model(
        arch=args.teacher_arch,
        in_channels=in_channels,
        num_classes=num_classes,
        device=device,
        state_dict=t_state_dict
    )

    # Checks if transfer learning from another model
    if args.transfer and args.teacher_arch == 'resnet-19':
        # Loads the state dict of the trained model
        transfer_state_dict = torch.load(args.transfer_weight, map_location="cpu")

        # Creates the model to perform transfer
        r18_model = make_teacher_model(
            arch='resnet-18',
            in_channels=in_channels,
            num_classes=num_classes,
            device=device,
            state_dict=transfer_state_dict
        )

        # Performs transfer learning
        transfer_weights_resnet18_resnet19(r18=r18_model, r19=t_model, trainable=args.trainable_weights)

        # Remove the model and its weights
        del r18_model, transfer_state_dict

    # Creates the optimizer
    if args.optimizer == 'adamw':
        opt = optim.AdamW(t_model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        opt = optim.SGD(t_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # Creates the scheduler
    if args.scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=args.lr_patience, factor=args.lr_factor)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Creates the loss function
    criterion = nn.CrossEntropyLoss()

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
            epoch_i, train_loader, t_model, criterion, opt, device, scaler
        )

        val_loss, val_acc1, val_acc5, val_batch_time = run_eval(val_loader, t_model, criterion, device)

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
        acc_monitor(val_acc1, epoch_i, t_model)

    t_model.load_state_dict(torch.load(args.model_path))
    test_loss, test_acc1, test_acc5, _ = run_eval(val_loader, t_model, criterion, device)
    run.log({"Test Loss": test_loss, "Test Accuracy 1": test_acc1, "Test Accuracy 5": test_acc5})

    # Ends the run
    run.finish()
