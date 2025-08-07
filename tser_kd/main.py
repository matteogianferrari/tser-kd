import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from snntorch import surrogate
from tser_kd.utils import setup_seed, AccuracyMonitor
from tser_kd.dataset import load_cifar10_data, load_mnist_data
from tser_kd.model.teacher import make_teacher_model
from tser_kd.model.student import make_student_model
from tser_kd.model import TSERKDLoss, TSCELoss
from tser_kd.dataset import StaticEncoder
from tser_kd.eval import run_eval
from tser_kd.training import run_train, run_kd_train, EarlyStopping
from config import args


# Setups the seed for reproducibility
setup_seed(42)


if __name__ == '__main__':
    # Selects the device for the experiment
    device = torch.device(args.dev_name if torch.cuda.is_available() else "cpu")

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

    # Initializes the models based on the experiment type
    # When 'experiment_type' is equal to 'kd', both if are executed
    if args.experiment_type == 'ann' or args.experiment_type == 'kd':
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
    if args.experiment_type == 'snn' or args.experiment_type == 'kd':
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

        # TODO: transfer from ann to snn

    if args.experiment_type == 'ann':
        # Sets a reference to the teacher model for the optimizer during training
        model_ref = t_model
    elif args.experiment_type == 'snn' or args.experiment_type == 'kd':
        # Sets a reference to the teacher model for the optimizer during training
        model_ref = s_model

    # Creates the optimizer
    if args.optimizer == 'adamw':
        opt = optim.AdamW(model_ref.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        opt = optim.SGD(model_ref.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # Creates the scheduler
    if args.scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=args.lr_patience, factor=args.lr_factor)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Creates the loss
    if args.experiment_type == 'ann':
        criterion = nn.CrossEntropyLoss()
    elif args.experiment_type == 'snn':
        train_criterion = TSCELoss()
        eval_criterion = nn.CrossEntropyLoss()
    elif args.experiment_type == 'kd':
        train_criterion = TSERKDLoss(alpha=args.alpha, gamma=args.gamma, tau=args.tau)
        eval_criterion = nn.CrossEntropyLoss()

    # Crates the scaler
    scaler = torch.amp.GradScaler(device='cuda')

    # Creates the encoder
    encoder = StaticEncoder(num_steps=args.t_steps)

    # Accuracy monitor
    acc_monitor = AccuracyMonitor(path=args.model_path)

    # Training
    epoch_i = 0
    curr_lr = opt.param_groups[0]["lr"]

    for epoch_i in range(args.epochs):
        # Forward pass
        if args.experiment_type == 'ann':
            train_loss, train_acc, epoch_time, train_batch_time = run_train(
                epoch_i, train_loader, t_model, criterion, opt, device, scaler
            )

            val_loss, val_acc1, val_acc5, val_batch_time = run_eval(val_loader, t_model, criterion, device)

            # Logging
            print(
                f"Time: {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc1: {val_acc1:.2f}% | Val Acc5: {val_acc5:.2f}% | LR: {curr_lr:.6f}"
            )
        elif args.experiment_type == 'snn':
            train_loss, train_acc, epoch_time, train_batch_time = run_train(
                epoch_i, train_loader, s_model, train_criterion, opt, device, scaler, encoder
            )

            val_loss, val_acc1, val_acc5, val_batch_time = run_eval(val_loader, s_model, eval_criterion, device, encoder)

            # Logging
            print(
                f"Time: {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc1: {val_acc1:.2f}% | Val Acc5: {val_acc5:.2f}% | LR: {curr_lr:.6f}"
            )
        elif args.experiment_type == 'kd':
            train_total_loss, train_ce_loss, train_kl_loss, train_e_reg, train_acc, epoch_time, train_batch_time = run_kd_train(
                epoch_i, train_loader, s_model, t_model, train_criterion, opt, device, scaler, encoder
            )

            val_loss, val_acc1, val_acc5, val_batch_time = run_eval(val_loader, s_model, eval_criterion, device, encoder)

            print(
                f"Time: {epoch_time:.1f}s | Train Total Loss: {train_total_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc1: {val_acc1:.2f}% | Val Acc5: {val_acc5:.2f}% | LR: {curr_lr:.6f}"
            )

        # Updates the LR
        if scheduler == 'reduce':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        curr_lr = opt.param_groups[0]["lr"]

        if args.experiment_type == 'ann':
            # Sets a reference to the teacher model for the optimizer during training
            model_ref = t_model
        elif args.experiment_type == 'snn' or args.experiment_type == 'kd':
            # Sets a reference to the teacher model for the optimizer during training
            model_ref = s_model

        # Accuracy monitor
        acc_monitor(val_acc1, epoch_i, model_ref)
