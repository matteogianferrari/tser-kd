import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tser_kd.utils import setup_seed, AccuracyMonitor
from tser_kd.dataset import load_cifar10_data, load_mnist_data
from tser_kd.model.teacher import make_teacher_model
from tser_kd.eval import run_eval
from tser_kd.training import run_train
from config import args


# Setups the seed for reproducibility
setup_seed(42)


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

    # Creates the loss
    if args.experiment_type == 'ann':
        criterion = nn.CrossEntropyLoss()

    # Crates the scaler
    scaler = torch.amp.GradScaler(device='cuda')

    # Accuracy monitor
    acc_monitor = AccuracyMonitor(path=args.model_path)

    # Training
    epoch_i = 0
    curr_lr = opt.param_groups[0]["lr"]

    for epoch_i in range(args.epochs):
        # Forward pass
        train_loss, train_acc, epoch_time, train_batch_time = run_train(
            epoch_i, train_loader, t_model, criterion, opt, device, scaler
        )

        val_loss, val_acc1, val_acc5, val_batch_time = run_eval(val_loader, t_model, criterion, device)

        # Logging
        print(
            f"Time: {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc1: {val_acc1:.2f}% | Val Acc5: {val_acc5:.2f}% | LR: {curr_lr:.6f}"
        )

        # Updates the LR
        if args.scheduler == 'reduce':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        curr_lr = opt.param_groups[0]["lr"]

        # Accuracy monitor
        acc_monitor(val_acc1, epoch_i, t_model)
